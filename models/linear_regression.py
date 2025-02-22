import numpy as np
import matplotlib.pyplot as plt
import random

x_train = [2, 4, 6, 9, 11]
y_train = [2, 4, 6, 8, 10]
plt.scatter(x_train, y_train)


# Once the model has been created, this
# function is called to produce a prediction (y_hat)
def predict_value(x, w, b):
    return w * x + b


def plot_line_from_slope_intercept(slope, y_intercept):
    x_range = (min(x_train) - 3, max(x_train) + 3)
    x = np.linspace(x_range[0], x_range[1], 100)
    y = slope * x + y_intercept
    color = (random.random(), random.random(), random.random())
    plt.plot(x, y, color=color)


def configure_plot_settings():
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)


# The cost is just the average of all of the squared
# deltas from the current line to the actual value
# and then multiplied by 2
def calculate_cost(w, b):
    errors = []
    for x, y in zip(x_train, y_train):
        y_hat = w * x + b
        delta = y_hat - y
        errors.append(delta**2)
    return sum(errors) / (len(errors) * 2)


def calculate_derivative_w(w, b):
    deltas = []
    for x, y in zip(x_train, y_train):
        deltas.append(((w * x + b - y) * x))
    return sum(deltas) / len(deltas)


def calculate_derivative_b(w, b):
    deltas = []
    for x, y in zip(x_train, y_train):
        deltas.append(((w * x + b - y)))
    return sum(deltas) / len(deltas)


# Gradient descent attempts to find the optimal localized
# lowest cost value, thus producting the best approximation
# of a line that will produce accurate reults.
def gradient_descent(w, b, a, iterations=1000, tolerance=1e-6):
    i = 0
    while True:
        d_w = calculate_derivative_w(w, b)
        d_b = calculate_derivative_b(w, b)

        if abs(d_w) < tolerance and abs(d_b) < tolerance:
            break

        w = w - a * d_w
        b = b - a * d_b

        i += 1
        if i >= iterations:
            break

    return (w, b)


slope = 3
bias = 1
alpha = 0.0075

configure_plot_settings()

w, b = gradient_descent(w=slope, b=bias, a=alpha)

cost = calculate_cost(w, b)

print(f"Predicted value for 5: {predict_value(5, w, b)}")
print(f"Cost: {cost}")

plot_line_from_slope_intercept(slope=w, y_intercept=b)
plt.show()
