import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import seaborn as sns
# Import math Library
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def graph_loss(history):
    print(history[0])
    history = np.array(history)
    history_w = history[:, 0]
    history_loss = history[:, 1]

    MESH_SIZE = 20
    weights = np.linspace(np.min(history_w) - 10, np.max(history_w) + 10,
                          MESH_SIZE)

    W, B = np.meshgrid(weights, history_loss)

    # Plot surface
    sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_zticklabels(())
    ax.set_xlabel("Weight", labelpad=20, fontsize=30)
    ax.set_ylabel("Bias", labelpad=20, fontsize=30)
    ax.set_zlabel("Loss", labelpad=5, fontsize=30)
    ax.plot_surface(W, B, cmap=cm.Blues, linewidth=0, antialiased=True)

    # Display plot
    plt.show()
