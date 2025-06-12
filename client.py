import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
from tqdm import tqdm

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. 모델 정의 (간단한 CNN)
class Net(nn.Module):
    """데이터셋을 위한 간단한 CNN 모델"""
    def __init__(self, in_channels=3, num_classes=9):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data(data_path, client_id, num_clients):
    """
    데이터를 로드하고 IID 방식으로 분할합니다.
    Dirichlet 분포를 사용해 클라이언트별 랜덤 크기 할당
    """
    data = np.load(data_path)
    x_train, y_train = data['train_images'], data['train_labels'].squeeze()

    # 데이터 정규화 및 텐서 변환을 위한 Transform 정의
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 1. 데이터 무작위 셔플 (IID)
    shuffled_indices = np.random.permutation(len(y_train))
    x_train, y_train = x_train[shuffled_indices], y_train[shuffled_indices]

    # 2. 클라이언트별 랜덤 데이터 크기 할당
    client_proportions = np.random.dirichlet(np.ones(num_clients))
    client_sizes = (client_proportions * len(x_train)).astype(int)
    client_sizes[-1] += len(x_train) - np.sum(client_sizes)  # 나머지 처리

    split_indices = np.cumsum(client_sizes)[:-1]
    x_train_clients = np.split(x_train, split_indices)
    y_train_clients = np.split(y_train, split_indices)

    x_train_client = x_train_clients[client_id]
    y_train_client = y_train_clients[client_id]

    # PyTorch TensorDataset 및 DataLoader 생성
    x_train_tensor = torch.stack([transform(img) for img in x_train_client])
    y_train_tensor = torch.tensor(y_train_client, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(f"클라이언트 {client_id}: 학습 데이터 {len(train_dataset)}개 로드 완료.")

    # 평가 데이터는 모든 클라이언트가 동일하게 전체를 사용
    x_test, y_test = data['test_images'], data['test_labels'].squeeze()
    x_test_tensor = torch.stack([transform(img) for img in x_test])
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader

# 3. 학습 및 평가 함수
def train(net, trainloader, epochs):
    """모델 학습"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """모델 평가"""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy

# 4. Flower NumPyClient 구현
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        """로컬 모델의 파라미터를 NumPy ndarray 리스트로 반환"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """서버로부터 받은 파라미터로 로컬 모델 업데이트"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """로컬 모델 학습"""
        self.set_parameters(parameters)
        print("로컬 모델 학습 시작...")
        train(self.model, self.trainloader, epochs=1)
        print("학습 완료.")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """로컬 모델 평가"""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client-id", type=int, required=True, help="클라이언트 ID")
    parser.add_argument("--num-clients", type=int, default=3, help="전체 클라이언트 수")
    args = parser.parse_args()

    # 모델과 데이터를 로드하고 Flower 클라이언트 실행
    net = Net().to(DEVICE)
    trainloader, testloader = load_data("./data/pathmnist.npz", args.client_id, args.num_clients)
    
    client = FlowerClient(net, trainloader, testloader)

    # 서버 주소는 docker-compose에서 정의한 서비스 이름(server)을 사용
    fl.client.start_numpy_client(server_address="server:8080", client=client)

if __name__ == "__main__":
    main()
