import os
import pandas as pd
import matplotlib.pyplot as plt


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

    data = []
    for name, (train_acc, test_acc) in results_dict.items():
        data.append({
            "Эксперимент": name,
            "Train Acc": round(train_acc * 100, 2),
            "Test Acc": round(test_acc * 100, 2)
        })

    df = pd.DataFrame(data)
    df.to_csv(save_path, sep=";", index=False)
