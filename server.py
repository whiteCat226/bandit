import flwr as fl
import torch
import numpy as np
import pickle
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict

# client.py에서 정의한 모델과 데이터 로더를 가져옵니다.
from client import Net, load_data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_evaluate_fn(data_path: str):
    """서버 측에서 전역 모델을 평가하기 위한 함수"""
    _, testloader = load_data(data_path, client_id=0, num_clients=1)

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        net = Net().to(DEVICE)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        from client import test
        loss, accuracy = test(net, testloader)

        # 마지막 라운드일 경우 파라미터 저장
        if server_round == 5:
            with open("final_model_parameters.pkl", "wb") as f:
                pickle.dump(parameters, f)

                model_path = "final_model.pth"
                torch.save(net.state_dict(), model_path)
                print(f"✅ PyTorch 모델 저장 완료: {model_path}")

        print(f"서버 측 전역 모델 평가 (라운드 {server_round}): Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    return evaluate

def save_model_at_final_round(server_round, parameters, **kwargs):
    if server_round == 5:  # 마지막 라운드에서 저장
        with open("final_model_parameters.pkl", "wb") as f:
            pickle.dump(parameters, f)
        print("✅ 최종 모델 파라미터 저장 완료: final_model_parameters.pkl")

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn("./data/pathmnist.npz"),
    )

    print("연합학습 서버를 시작합니다.")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
