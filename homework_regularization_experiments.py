import torch
import logging
from utils.experiment_utils import load_data
from utils.model_utils import FullyConnectedModel, train_model, count_parameters
from utils.visualization_utils import plot_training_history, results_dict_to_table, plot_weight_distribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Фиксированная архитектура: 3 скрытых слоя
FIXED_HIDDEN_SIZES = [256, 128, 64]

REGULARIZATION_EXPERIMENTS = [
    {"name": "no_reg", "use_bn": False, "dropout": None, "weight_decay": 0.0},
    {"name": "dropout_0.1", "use_bn": False, "dropout": 0.1, "weight_decay": 0.0},
    {"name": "dropout_0.3", "use_bn": False, "dropout": 0.3, "weight_decay": 0.0},
    {"name": "dropout_0.5", "use_bn": False, "dropout": 0.5, "weight_decay": 0.0},
    {"name": "batch_norm", "use_bn": True, "dropout": None, "weight_decay": 0.0},
    {"name": "bn_dropout", "use_bn": True, "dropout": 0.3, "weight_decay": 0.0},
    {"name": "l2_reg", "use_bn": False, "dropout": None, "weight_decay": 1e-4},
]

def build_layers(hidden_sizes, use_bn=False, dropout=None):
    """
    Конструктор слоев для регуляризационных экспериментов
    :param hidden_sizes: размеры скрытых слоев
    :param use_bn: булево значение, добавлять ли BatchNorm
    :param dropout: булево значение, добавлять ли Dropout
    :return: список слоев
    """
    layers = []
    for i, hidden in enumerate(hidden_sizes):
        layers.append({'type': 'linear', 'size': hidden})
        if use_bn:
            layers.append({'type': 'batch_norm'})
        layers.append({'type': 'relu'})
        if dropout is not None:
            layers.append({'type': 'dropout', 'rate': dropout})
    return layers

def run_regularization_experiments(dataset: str, batch_size: int = 64, epochs: int = 10):
    """
    Запуск экспериментов с регуляризацией
    :param dataset:  имя датасета (mnist или cifar)
    :param batch_size: размер батча
    :param epochs: число эпох
    """
    train_loader, test_loader, input_size = load_data(dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores = {}

    for config in REGULARIZATION_EXPERIMENTS:
        name = f"{config['name']}_{dataset}"
        layers = build_layers(
            hidden_sizes=FIXED_HIDDEN_SIZES,
            use_bn=config["use_bn"],
            dropout=config["dropout"]
        )

        model = FullyConnectedModel(
            input_size=input_size,
            num_classes=10,
            layers=layers
        ).to(device)

        logger.info(f"Running {name}")
        params = count_parameters(model)
        logger.info(f"Params: {params}")

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            device=str(device),
            weight_decay=config["weight_decay"]  # L2
        )

        # Сохраняем графики
        plot_training_history(history, f"regularization_experiments/train_history/{name}_history.png")
        plot_weight_distribution(model, f"plots/regularization_experiments/weights/{name}_weights.png")

        # Финальные метрики
        train_acc = history['train_accs'][-1]
        test_acc = history['test_accs'][-1]
        scores[name] = (train_acc, test_acc)

    # Таблица с результатами
    save_path = f"results/regularization_experiments/{dataset}.csv"
    results_dict_to_table(scores, save_path)


if __name__ == '__main__':
    for ds in ['mnist', 'cifar']:
        run_regularization_experiments(ds)
