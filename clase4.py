# import numpy as np
# import matplotlib.pyplot as plt

# t = np.linspace(0, 1, num=1000)
# y = 0.5 * np.sin(2 * np.pi * t * 25)

# plt.plot(t, y)
# plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# imagen inventada
bw = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

# transformamos los datos a uint8
bw = np.array(bw, dtype="uint8")

cnt, h = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
coords = np.vstack(cnt[0])
x, y = zip(*coords)

signal = np.array(x) + 1j * np.array(y)

fft = np.fft.fft(signal)
plt.bar(np.arange(len(signal)), np.abs(fft))

np.sort(np.abs(fft))[::-1]

# plt.plot(np.abs(fft))
# plt.show()

# plt.figure()
# plt.scatter(x, y)
# plt.imshow(bw, cmap="gray")
# plt.show()

im = cv2.imread("FotoLluvia.png")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
bw = (gray < 128) * 1

plt.imshow(bw)
plt.show()
