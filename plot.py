import matplotlib.pyplot as plt

def plot_results(epochs: int, y: list, title: str, y_label: str) -> None:

    x = list(range(epochs))

    plt.margins(x=0)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x, y, label=y_label)
    plt.show()