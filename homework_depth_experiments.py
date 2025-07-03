import torch
import logging
from utils.experiment_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import FullyConnectedModel, train_model, count_parameters
from utils.visualization_utils import plot_training_history, results_dict_to_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENTS = [
    {"name": "1_layer", "hidden_sizes": []},
    {"name": "2_layers", "hidden_sizes": [128]},
    {"name": "3_layers", "hidden_sizes": [256, 128]},
    {"name": "5_layers", "hidden_sizes": [512, 256, 128, 64]},
    {"name": "7_layers", "hidden_sizes": [512, 256, 128, 64, 32, 16]},
]

def build_layers(hidden_sizes: list,
                 use_bn_dropout: bool):
    """
    Конструктор слоев модели
    :param hidden_sizes: размеры скрытых слоев
    :param use_bn_dropout: булево значение, добавлять ли BatchNorm и Dropout
    :return:
    """
    layers = []
    for i, hidden in enumerate(hidden_sizes):
        layers.append({"type": "linear", "size": hidden})
        if use_bn_dropout:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if use_bn_dropout:
            rate = max(0.1, 0.3 * (1 - i / len(hidden_sizes)))
            layers.append({"type": "dropout", "rate": rate})

    return layers

def run_depth_experiments(dataset: str, batch_size: int = 64, epochs: int = 10):
    """
    Запуск экспериментов с глубиной
    :param dataset: имя датасета mnist или cifar
    :param batch_size: размер батча
    :param epochs: число эпох
    """
    if dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        input_size = 784
    elif dataset == "cifar":
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        input_size = 3072
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores = {}

    # bool значение, добавлять ли BatchNorm и Dropout
    for use_extra in [False, True]:
        suffix = "with_bn_do" if use_extra else "base"
        for exp in EXPERIMENTS:
            exp_name = f"{exp['name']}_{dataset}_{suffix}"
            # создание слоев конструктором
            layers = build_layers(
                hidden_sizes=exp["hidden_sizes"],
                use_bn_dropout=use_extra
            )
            model = FullyConnectedModel(
                input_size=input_size,
                num_classes=10,
                layers=layers
            ).to(device)

            params = count_parameters(model)
            logger.info(f"Running {exp_name}: params={params}")

            history = train_model(
                model, train_loader, test_loader,
                epochs=epochs, device=str(device)
            )
            # создание и сохранение графиков
            plot_training_history(history, f"depth_experiments/{exp_name}.png")

            train_acc = history['train_accs'][-1]
            test_acc = history['test_accs'][-1]
            scores[exp_name] = (train_acc, test_acc)

    # преобразование результатов в таблицу и ее сохранение
    save_path = f"results/depth_experiments/{dataset}.csv"
    results_dict_to_table(scores, save_path)


if __name__ == '__main__':
    for ds in ['mnist', 'cifar']:
        run_depth_experiments(ds)
