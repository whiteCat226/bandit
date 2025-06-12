import torch
import pickle
from collections import OrderedDict
from client import Net, load_data, test

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. 모델 인스턴스 생성
model = Net().to(DEVICE)

# 2. 저장된 파라미터 로딩
with open("final_model_parameters.pkl", "rb") as f:
    parameters = pickle.load(f)

# 3. 파라미터를 state_dict 형태로 변환 및 적용
params_dict = zip(model.state_dict().keys(), parameters)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
model.load_state_dict(state_dict, strict=True)
model.eval()

# 4. 전체 테스트 데이터 로딩 (client_id=0으로 전체 사용)
_, testloader = load_data("./data/pathmnist.npz", client_id=0, num_clients=1)

# 5. 평가 수행
loss, acc = test(model, testloader)
print("\n================ 최종 모델 성능 =================")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {acc:.4f}")
print("==============================================\n")

# 6. 전체 모델 저장 (.pth 파일)
torch.save(model, "final_model.pth")
print("✅ 전체 모델 저장 완료: final_model.pth")


