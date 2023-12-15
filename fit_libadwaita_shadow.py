#!/usr/bin/python

import argparse
from PIL import Image
import numpy as np
import scipy
import matplotlib.pyplot as plt

START_X_OFFSET = 74
GRADIENT_PIXELS = 74
SCALE_FACTOR = 1


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


parser = argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

image = Image.open(args.image)
assert START_X_OFFSET - GRADIENT_PIXELS == 0

pixels = image.load()
x_values = []
y_values = []
for i, xoff in enumerate(reversed(range(0, START_X_OFFSET))):
    r, g, b, a = pixels[xoff, image.height / 2]
    print(i, xoff, a)
    x_values.append((i + 0.5) / SCALE_FACTOR)
    y_values.append(a / 255)


x = np.array(x_values)
y = np.array(y_values)

print(x)

popt, *_ = scipy.optimize.curve_fit(func, x, y)

shadow_size = 0
while round(func(shadow_size, *popt) * 255) > 0:
    shadow_size += 1

print(f"a = {popt[0]}")
print(f"b = {popt[1]}")
print(f"c = {popt[2]}")
print(f"shadow size = {shadow_size}")

if args.plot:
    plt.figure()
    plt.plot(x, y, 'ko', label="Original Data")
    plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
    plt.legend()
    plt.title(args.image)
    plt.show()
