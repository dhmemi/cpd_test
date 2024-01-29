from functools import partial
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import numpy as np
import time
import cv2


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def main(true_rigid=True):
    X = np.loadtxt('data/cpd-target.txt')
    Y = np.loadtxt('data/cpd-source-0.txt')
    if true_rigid is True:
        theta = np.pi / 6.0
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = np.array([0.5, 1.0])
        Y = np.dot(X, R) + t

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = RigidRegistration(**{'X': X, 'Y': Y})
    reg.register(callback)
    plt.show()


def draw_binary(points, shape):
    img = np.zeros(shape, np.uint8)
    contours = np.array(points.reshape((-1, 2)).astype(np.int32))
    cv2.drawContours(img, [contours], -1, [240], -1)
    return img

def erode_and_dilate(img, size):
    if size  == 0:
        return img
    kernel = np.ones((size, size), dtype=np.uint8)
    out = cv2.erode(img, kernel)
    out = cv2.dilate(out, kernel)
    return out


if __name__ == '__main__':
    x, y = main(true_rigid=False)
    # x_img = draw_binary(x)
    # y_img = draw_binary(y)
    # diff = cv2.bitwise_xor(x_img, y_img)
    # cv2.namedWindow("diff", cv2.WINDOW_FREERATIO)
    # cv2.imshow("diff", erode_and_dilate(diff, 2))
    # cv2.waitKey(0)
