import matplotlib.pyplot as plt
import numpy as np


def plot_results(epochs: int, y: list, title: str, y_label: str) -> None:

    x = list(range(epochs))

    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.ylim([0, 1])
    plt.margins(x=0)
    plt.margins(y=0)
    plt.title(title)
    plt.plot(x, y, label=y_label)
    plt.savefig(f'./figures/{y_label}.png')
    plt.show()