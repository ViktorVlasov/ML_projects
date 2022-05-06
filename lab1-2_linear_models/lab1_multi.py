import numpy as np
from matplotlib import pyplot as plt


def compute_hypothesis(X, theta):
    return X @ theta


def compute_cost(X, y, theta):
    m = X.shape[0]  # количество примеров в выборке
    # ВАШ КОД ЗДЕСЬ
    return 1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2)
    # ==============


def get_gradient(X, y, y_pred):
    grad = (y_pred - y)[:, np.newaxis] * X
    grad = grad.mean(axis=0)
    return grad


def gradient_descend(X, y, theta, alpha, num_iter):
    history = list()
    m = X.shape[0]  # количество примеров в выборке
    n = X.shape[1]  # количество признаков с фиктивным
    for i in range(num_iter):

        # ВАШ КОД ЗДЕСЬ
        y_pred = compute_hypothesis(X, theta)
        grad = get_gradient(X, y, y_pred)
        theta -= alpha * grad
        # =====================

        history.append(compute_cost(X, y, theta))
    return history, theta


def scale_features(X):
    # ВАШ КОД ЗДЕСЬ
    # меняет оригинальный X
    # X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)

    Z = X.copy()
    Z[:, 1:] = (Z[:, 1:] - Z[:, 1:].mean(axis=0)) / Z[:, 1:].std(axis=0)
    return (Z)
    # =====================


def normal_equation(X, y):
    # ВАШ КОД ЗДЕСь
    return np.linalg.pinv(X.T @ X) @ X.T @ y
    # =====================


def load_data(data_file_path):
    with open(data_file_path) as input_file:
        X = list()
        y = list()
        for line in input_file:
            *row, label = map(float, line.split(','))
            X.append([1] + row)
            y.append(label)
        return np.array(X, float), np.array(y, float)


X, y = load_data('lab1data2.txt')

# num_iter = 1500 - слишком много без
history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 10)

plt.title('График изменения функции стоимости от номера итерации до стандартизации')
plt.plot(range(len(history)), history)
plt.show()

#print(X)
X = scale_features(X)
# print(X)

history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

plt.title('График изменения функции стоимости от номера итерации после стандартизации')
plt.plot(range(len(history)), history)
plt.show()
#
theta_solution = normal_equation(X, y)
print(f'theta, посчитанные через градиентный спуск: {theta}, через нормальное уравнение: {theta_solution}')
