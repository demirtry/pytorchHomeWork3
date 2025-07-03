import torch
import logging
from utils.experiment_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import FullyConnectedModel, train_model, count_parameters
from utils.visualization_utils import plot_training_history, results_dict_to_table, heatmap_from_data

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


def load_data(dataset: str, batch_size: int = 64):
    """
    Загрузка датасета и определение input_size
    :param dataset: 'mnist' или 'cifar'
    :param batch_size: размер батча
    :return: возвращает train_loader, test_loader, input_size
    """
    if dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        input_size = 784
    elif dataset == "cifar":
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        input_size = 3072
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_loader, test_loader, input_size

def run_width_experiments(dataset: str, batch_size: int = 64, epochs: int = 10):
    """
    Запуск экспериментов с различной шириной скрытых слоёв
    :param dataset: 'mnist' или 'cifar'
    :param batch_size: размер батча
    :param epochs: число эпох обучения
    """
    train_loader, test_loader, input_size = load_data(dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores = {}

    # Эксперименты с BatchNorm+Dropout и без
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


# grid search по схемам изменения ширины слоёв
def run_width_grid_search(dataset: str, batch_size: int = 64, epochs: int = 5):
    """
    Grid search по различным схемам ширины скрытых слоёв и визуализация heatmap.
    Схемы:
      - constant: [w, w, w]
      - expanding: [w/4, w/2, w]
      - narrowing: [w, w/2, w/4]
    Используются базовые ширины: 64, 128, 256, 512
    :param dataset: 'mnist' или 'cifar'
    :param batch_size: размер батча
    :param epochs: число эпох обучения
    """
    train_loader, test_loader, input_size = load_data(dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Базовые значения
    base_widths = [64, 128, 256, 512]
    grid_records = []

    for w in base_widths:
        schemes = {
            f"constant_{w}": [w, w, w],
            f"expanding_{w//4}_{w//2}_{w}": [w//4, w//2, w],
            f"narrowing_{w}_{w//2}_{w//4}": [w, w//2, w//4]
        }
        for name, hidden_sizes in schemes.items():
            exp_name = f"grid_{name}_{dataset}"
            layers = build_layers(hidden_sizes)
            model = FullyConnectedModel(
                input_size=input_size,
                num_classes=10,
                layers=layers
            ).to(device)
            params = count_parameters(model)
            logger.info(f"Running {exp_name}: params={params}")
            history = train_model(model, train_loader, test_loader, epochs=epochs, device=str(device))
            train_acc = history['train_accs'][-1]
            test_acc = history['test_accs'][-1]
            grid_records.append({
                'scheme': name,
                'base_width': w,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'params': params
            })

    heatmap_from_data(grid_records, dataset)


if __name__ == '__main__':
    for ds in ['mnist', 'cifar']:
        run_width_experiments(ds)

    # for ds in ['mnist', 'cifar']:
    #     run_width_grid_search(ds)
