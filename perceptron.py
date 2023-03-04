# Simply perceptron algorithm visualization.
# We'll learn to classify 3 distinct digits, (28*28=784 inputs, 784*3=2352 weights, logistic function, one-hot encoding)

import copy
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Window size
window_w = 1720
window_h = 1080
np.set_printoptions(suppress=True)

# Tkinter Setup
root = Tk()
root.title("Simulator")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = tkinter.Canvas(root, width=window_w, height=window_h)
w.pack()


# Coordinate Shift
def A(v):
    return np.array([v[0] + window_w/2, -v[1] + window_h/2])


# Animation params
t = 0
dt = 0.001
keys = []


# Get keyframe number
def frame(t_):
    global keys
    for i in range(len(keys)-1):
        if keys[i] <= t_ < keys[i+1]:
            return i


# Squash t parameter to keys
def squash(t_):
    global keys
    for i in range(len(keys)-1):
        if keys[i] <= t_ < keys[i+1]:
            return (t_ - keys[i]) / (keys[i+1] - keys[i])


# Ease functions (for animation)
def ease_in(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


def ease_out(t_):
    return np.power(t_ - 1, 2) / (np.power(t_, 2) + np.power(t_ - 1, 2))


def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


# Learning functions
weights = [np.ones(28 * 28) * 1, np.ones(28 * 28) * 1, np.ones(28 * 28) * 1]  # 3 sets of weights for the 3 classes
bias = 0
digits = [1, 7, 8]
def relu(x):
    return 0.01 * x if x < 0 else x  # leaky relu


def reluprime(x):
    return 0.01 if x <= 0 else 1


# def softmax(raw_probs):
#     assert len(raw_probs) == 3
#     return np.exp(raw_probs) / sum(np.exp(raw_probs))


def one_hot(probs):
    global digits
    assert len(probs) == 3  # and sum(probs) - 1 < 0.000001
    return digits[np.argmax(probs)]


def predict(x):
    '''
        returns: 1) Raw (un-softmaxed) probability for each of the 3 classes, and 2) The single, one-hot class.
    '''
    global weights, bias
    assert len(x) == 28 * 28
    raw_probs = [0, 0, 0]
    for i in range(3):
        raw_probs[i] = relu(np.dot(x, weights[i]) + bias)
    return raw_probs, one_hot(raw_probs)  # one_hot(softmax(raw_probs))


def loss(x, y):
    global digits
    assert len(x) == 28 * 28 and y in digits
    expected = np.array([1 if y_ == y else 0 for y_ in digits])
    predicted, _ = predict(x)
    diff = predicted - expected
    return 0.5 * np.dot(diff, diff)


def update_weights(x, y, alpha):
    global weights, bias, digits
    assert len(x) == 28 * 28 and y in digits
    expected = np.array([1 if y_ == y else 0 for y_ in digits])
    for i in range(3):
        xTw = np.dot(x, weights[i])
        bias_gradient = (relu(xTw + bias) - expected[i]) * reluprime(xTw + bias)
        weight_gradient = bias_gradient * x
        bias -= alpha * bias_gradient
        weights[i] -= alpha * weight_gradient


# Color ramp for weights
def color_ramp(x):
    radius = 0.5
    return ease_in((x + radius) / (2 * radius)) * 255


# Parameters
n_images = 2000
input_img = None
weight_images = [None, None, None]


# Main function
def run():
    global n_images, input_img, weight_images, bias, digits
    w.configure(background='black')

    # Read and store input digits
    data = []
    labels = []
    file = open('digits.txt', 'r')
    i = 0
    while i < n_images:
        vals = np.array(file.readline().rstrip().rsplit(',')).astype(float)
        if int(vals[0]) not in digits:
            continue
        data.append(vals[1:])
        labels.append(vals[0])
        i += 1

    # Train!
    pts = []
    for i in range(n_images):
        w.delete('all')

        # View rectangle
        width, height = window_w * 0.3, window_h * 0.9
        w.create_rectangle(*A(np.array([-width / 2, height / 2])),
                           *A(np.array([width / 2, -height / 2])),
                           fill='', outline='white', width=6)

        # Input digit square
        width, height = window_w * 0.1, window_w * 0.1
        y_shift = window_h * 0.25
        w.create_rectangle(*A(np.array([-width / 2, (height / 2) + y_shift])),
                           *A(np.array([width / 2, (-height / 2) + y_shift])),
                           fill='', outline='white', width=6)

        # Display current datum as image
        pixels = np.reshape(data[i], (28, 28))
        input_img = ImageTk.PhotoImage(Image.fromarray(pixels).resize((int(width), int(height))))
        w.create_image(*A(np.array([-width / 2, (height / 2) + y_shift])), image=input_img, anchor=NW)

        # Display current weights as images in squares
        width, height = window_w * 0.1, window_w * 0.1
        for k in range(3):
            w.create_rectangle(*A(np.array([(-1.5 * width) + (k * width), height / 2])),
                               *A(np.array([(-0.5 * width) + (k * width), -height / 2])),
                               fill='', outline='white', width=6)

            # TODO: Figure out a good measure to extract out the differences
            pixels = np.reshape(color_ramp(weights[k] - weights[(k+1) % 3]), (28, 28))
            weight_images[k] = ImageTk.PhotoImage(Image.fromarray(pixels).resize((int(width), int(height))))
            w.create_image(*A(np.array([(-1.5 * width) + (k * width), height / 2])), image=weight_images[k], anchor=NW)

        # Make a prediction (forward pass)
        x, y = data[i] / 255., labels[i]
        prediction = predict(x)
        cost = loss(x, y)
        pts.append(cost)
        print('Cost:', cost, 'Predicted:', prediction, 'Expected:', y)
        # print()
        # print()
        # print()
        # print()
        # for p in range(28*28):
        #     if abs(weights[0][p] - weights[1][p]) > 0.000001:
        #         print(weights[0][p], weights[1][p], 'diff:', weights[0][p] - weights[1][p])
        # print(weights[0][0])

        update_weights(x, y, alpha=0.001)

        w.update()
        # time.sleep(1)

    plt.plot(np.arange(len(pts)), pts)
    plt.ylim((0, 5))
    plt.show()


# From https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


# Main function
if __name__ == '__main__':
    run()

# Necessary line for Tkinter
mainloop()
