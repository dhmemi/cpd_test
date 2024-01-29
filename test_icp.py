import icp
import numpy as np
import matplotlib.pyplot as plt


def visualize(X, Y, ax):
    plt.cla()
    ax.plot(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.plot(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():

    X = np.loadtxt('data/cpd-target.txt')
    Y = np.loadtxt('data/cpd-source-1.txt')
    _, Y_out = icp.icp(X, Y, 10000, 3)

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    visualize(X, Y_out, fig.axes[0])
    plt.show()


if __name__ == '__main__':
    main()