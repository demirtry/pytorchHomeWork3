import torch
import logging
from utils.experiment_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import FullyConnectedModel, train_model, count_parameters
from utils.visualization_utils import plot_training_history, results_dict_to_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ширинные конфигурации (3 скрытых слоя)
WIDTH_EXPERIMENTS = [
    {"name": "narrow", "hidden_sizes": [64, 32, 16]},
    {"name": "medium", "hidden_sizes": [256, 128, 64]},
    {"name": "wide",   "hidden_sizes": [1024, 512, 256]},
    {"name": "very_wide", "hidden_sizes": [2048, 1024, 512]},
]


def build_layers(hidden_sizes: list[int]):
    """
    Конструктор слоёв для ширинных экспериментов.
    :param hidden_sizes: размеры слоев
    :return: список конфигураций слоёв
    """
    w1, w2, w3 = hidden_sizes
    layers = [
        {'type': 'linear', 'size': w1},
        {'type': 'relu'},

        {'type': 'linear', 'size': w2},
        {'type': 'relu'},

        {'type': 'linear', 'size': w3},
        {'type': 'relu'}
    ]
    return layers


def run_width_experiments(dataset: str, batch_size: int = 64, epochs: int = 2):
    """
    Запуск экспериментов с различной шириной скрытых слоёв
    :param dataset: 'mnist' или 'cifar'
    :param batch_size: размер батча
    :param epochs: число эпох обучения
    """
    # Загрузка датасета
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

    # Эксперименты: без и с BatchNorm+Dropout
    for exp in WIDTH_EXPERIMENTS:
        exp_name = f"{exp['name']}_{dataset}"
        layers = build_layers(
            hidden_sizes=exp['hidden_sizes'],
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
        plot_training_history(history, f"width_experiments/{exp_name}.png")

        train_acc = history['train_accs'][-1]
        test_acc = history['test_accs'][-1]
        scores[exp_name] = (train_acc, test_acc)
        scores[exp_name] = (train_acc, test_acc)

    save_path = f"results/width_experiments/{dataset}.csv"
    results_dict_to_table(scores, save_path)


if __name__ == '__main__':
    for ds in ['mnist', 'cifar']:
        run_width_experiments(ds)
