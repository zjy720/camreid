# import torch
# from train import train
# from evaluate import evaluate, ablation_study
# from models.reid_model import ReIDModel
# from config import config
# import itertools
# import pandas as pd
#
#
# def tune_hyperparameters():
#     lambda_configs = [
#         {'lambda1': 0.5, 'lambda2': 0.3, 'lambda3': 1.0},
#         {'lambda1': 0.3, 'lambda2': 0.5, 'lambda3': 1.0},
#         {'lambda1': 0.0, 'lambda2': 0.3, 'lambda3': 1.0},  # No MMD
#         {'lambda1': 0.5, 'lambda2': 0.0, 'lambda3': 1.0},  # No align
#     ]
#     param_configs = [
#         {'k': 3, 'g': 4, 'K': 5},
#         {'k': 1, 'g': 4, 'K': 5},
#         {'k': 3, 'g': 1, 'K': 5},
#         {'k': 3, 'g': 4, 'K': 1},
#     ]
#     results = []
#     for lambda_cfg, param_cfg in itertools.product(lambda_configs, param_configs):
#         print(f"Tuning with lambda1={lambda_cfg['lambda1']}, lambda2={lambda_cfg['lambda2']}, "
#               f"lambda3={lambda_cfg['lambda3']}, k={param_cfg['k']}, g={param_cfg['g']}, K={param_cfg['K']}")
#
#         model = ReIDModel(
#             k=param_cfg['k'],
#             g=param_cfg['g'],
#         ).to(config.DEVICE)
#         train(**lambda_cfg)
#         rank1, mAP, flops = evaluate(model, k=param_cfg['k'], g=param_cfg['g'], K=param_cfg['K'])
#         results.append({
#             'lambda1': lambda_cfg['lambda1'],
#             'lambda2': lambda_cfg['lambda2'],
#             'lambda3': lambda_cfg['lambda3'],
#             'k': param_cfg['k'],
#             'g': param_cfg['g'],
#             'K': param_cfg['K'],
#             'Rank-1': rank1,
#             'mAP': mAP,
#             'FLOPs': flops
#         })
#
#     df = pd.DataFrame(results)
#     df.to_csv("tuning_results.csv", index=False)
#     print("Tuning Results:")
#     print(df)
#     return results
#
#
# if __name__ == "__main__":
#     # Run default training and evaluation
#     model = train()
#     evaluate(model)
#     ablation_study()
#     # Run hyperparameter tuning
#     tune_hyperparameters()
import torch
from train import train
from evaluate import evaluate, ablation_study
from models.reid_model import ReIDModel
from config import config
import itertools
import pandas as pd

def tune_hyperparameters():
    lambda_configs = [
        {'lambda1': 0.5, 'lambda2': 0.3, 'lambda3': 1.0},
        {'lambda1': 0.3, 'lambda2': 0.5, 'lambda3': 1.0},
        {'lambda1': 0.0, 'lambda2': 0.3, 'lambda3': 1.0},  # No MMD
        {'lambda1': 0.5, 'lambda2': 0.0, 'lambda3': 1.0},  # No align
    ]
    param_configs = [
        {'k': 3, 'g': 4, 'K': 5},
        {'k': 1, 'g': 4, 'K': 5},
        {'k': 3, 'g': 1, 'K': 5},
        {'k': 3, 'g': 4, 'K': 1},
    ]
    results = []
    for lambda_cfg, param_cfg in itertools.product(lambda_configs, param_configs):
        print(f"Tuning with lambda1={lambda_cfg['lambda1']}, lambda2={lambda_cfg['lambda2']}, "
              f"lambda3={lambda_cfg['lambda3']}, k={param_cfg['k']}, g={param_cfg['g']}, K={param_cfg['K']}")

        model = ReIDModel(k=param_cfg['k'], g=param_cfg['g']).to(config.DEVICE)
        model = train(**lambda_cfg)  # 假设 train 返回模型
        rank1, mAP, flops = evaluate(model, k=param_cfg['k'], g=param_cfg['g'], K=param_cfg['K'], test_mode="all", trial=0)
        results.append({
            'lambda1': lambda_cfg['lambda1'],
            'lambda2': lambda_cfg['lambda2'],
            'lambda3': lambda_cfg['lambda3'],
            'k': param_cfg['k'],
            'g': param_cfg['g'],
            'K': param_cfg['K'],
            'Rank-1': rank1,
            'mAP': mAP,
            'FLOPs': flops
        })

    df = pd.DataFrame(results)
    df.to_csv("tuning_results.csv", index=False)
    print("Tuning Results:")
    print(df)
    return results

if __name__ == "__main__":
    # Run default training and evaluation
    model = train()
    evaluate(model, test_mode="all", trial=0)
    ablation_study()
    # Run hyperparameter tuning
    tune_hyperparameters()