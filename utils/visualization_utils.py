import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_training_history(history, plot_name: str):
    """Визуализирует и сохраняет историю обучения"""
    os.makedirs('plots', exist_ok=True)
    os.makedirs('plots/depth_experiments', exist_ok=True)
    os.makedirs('results/width_experiments', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()

    plt.savefig(f'plots/{plot_name}')
    plt.close()


def results_dict_to_table(results_dict: dict, save_path: str):
    """
    Принимает словарь вида {'exp_name': (train_acc, test_acc)}
    и сохраняет pandas DataFrame с результатами:
    """
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/depth_experiments', exist_ok=True)
    os.makedirs('results/width_experiments', exist_ok=True)
    os.makedirs('results/regularization_experiments', exist_ok=True)

    data = []
    for name, (train_acc, test_acc) in results_dict.items():
        data.append({
            "Эксперимент": name,
            "Train Acc": round(train_acc * 100, 2),
            "Test Acc": round(test_acc * 100, 2)
        })

    df = pd.DataFrame(data)
    df.to_csv(save_path, sep=";", index=False)


def heatmap_from_data(grid_records: list[dict], dataset: str):
    df_grid = pd.DataFrame(grid_records)

    # Добавляем тип схемы для группировки по Y
    df_grid['scheme_type'] = df_grid['scheme'].apply(lambda s: s.split('_')[0])

    # Создаём сводную таблицу для тепловой карты
    heatmap_data = df_grid.pivot(index='scheme_type', columns='base_width', values='test_acc')

    # Рисуем тепловую карту
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='magma', linewidths=0.5)
    plt.title(f"Grid Search Test Acc Heatmap ({dataset})")
    plt.xlabel('Base Width')
    plt.ylabel('Scheme')
    plt.tight_layout()

    # Сохраняем изображение
    os.makedirs("plots", exist_ok=True)
    heatmap_path = f"plots/grid_search/heatmap_{dataset}.png"
    plt.savefig(heatmap_path)
    plt.close()

    # Сохраняем CSV
    os.makedirs("results/grid_search", exist_ok=True)
    save_csv = f"results/grid_search/width_grid_{dataset}.csv"
    df_grid.to_csv(save_csv, index=False)


def plot_weight_distribution(model, save_path: str):
    weights = []
    for param in model.parameters():
        if param.dim() > 1:
            weights.append(param.detach().cpu().flatten())

    all_weights = torch.cat(weights).numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(all_weights, bins=50, color='skyblue', edgecolor='black')
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

