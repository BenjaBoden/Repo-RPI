import cv2
import numpy as np
import matplotlib.pyplot as plt


im = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]


def m(r, s, coord):
    return np.sum(coord[:, 0] ** r * coord[:, 1] ** s)


def mu(r, s, coord):
    i, j = cm(coord)
    return m(r, s, coord - np.array([i, j]))


def cm(coord):
    m00 = m(0, 0, coord)
    m10 = m(1, 0, coord)
    m01 = m(0, 1, coord)
    return m10 / m00, m01 / m00


def trs(r, s):
    return ((r + s) / 2) + 1


def eta(r, s, coord):
    t = trs(r, s)
    return mu(r, s, coord) / (mu(0, 0, coord) ** t)


im = np.array(im)
i, j = cm(im)
coord = np.argwhere(im == 1)
print(m(0, 0, coord))
print(mu(2, 1, coord))
print(eta(2, 0, coord) - eta(0, 2, coord))

# plt.imshow(im, cmap="gray")
# plt.scatter(j, i, marker="x")
# plt.show()
