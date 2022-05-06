import numpy as np

y_pred = np.array([1, 2], float)
print("y_pred =", y_pred)

y = np.array([1.5, 2.5], float)
print("y =", y)

X = np.array([[1, 1], [10, 20], [40, 50]]).T
print("X =", X)

# grad = (y_pred - y)[:, np.newaxis] * X
# grad = (y_pred - y)[np.newaxis, :] * X

print((y_pred - y).shape, (y_pred - y))
print((y_pred - y)[np.newaxis, :].shape, (y_pred - y)[np.newaxis, :])
print((y_pred - y)[:, np.newaxis].shape, (y_pred - y)[:, np.newaxis] * X)


#print((y_pred - y)[np.newaxis, :], "|", (y_pred - y)[np.newaxis, :] @ X)
#print((y_pred - y)[np.newaxis, :])
#print((y_pred - y)[np.newaxis, :].shape)
#print(X.shape)

#print(grad)


#print("grad =", grad)

#grad = grad.mean(axis=0)
#print("grad after mean =", grad)

def calc_gradient(X, y, y_pred):
        grad = (y_pred - y)[:, np.newaxis] * X
        grad = grad.mean(axis=0)
        return grad