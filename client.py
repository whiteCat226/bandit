## client.py
import flwr as fl
import torch
from torch.utils.data import DataLoader
from model import Classifier
from dataset import PathMNISTDataset, split_dataset
import sys

hospital_id = int(sys.argv[1])  # pass 0, 1, 2 from docker-compose

dataset = PathMNISTDataset("pathmnist.npz", split="train")
hospitals = split_dataset(dataset, num_clients=3)
trainloader = DataLoader(hospitals[hospital_id], batch_size=32, shuffle=True)

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), parameters):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()
        for epoch in range(1):
            for images, labels in trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return get_parameters(self.model), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        return 0.0, len(trainloader.dataset), {}

model = Classifier()
fl.client.start_client(server_address="server:8080", client=FlowerClient(model).to_client())

