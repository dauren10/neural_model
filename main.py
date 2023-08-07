import numpy as np

def sigmoid(x, der=False):
    if der == True:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

# набор входных данных
x = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
])

y = np.array([[0, 0, 1, 1]]).T
# сделаем случайные числа более определенными
np.random.seed(1)
# инициализируем веса случайным образом со средним 0
syn0 = 2 * np.random.random((3, 1)) - 1

l1 = []

for iter in range(10000):
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))
    l1_error = y - l1
    l1_delta = l1_error * sigmoid(l1, True)

    syn0 += np.dot(l0.T, l1_delta)

print('Выходные данные после тренировки')
print(l1)

new_one = np.array([1, 1, 1])
l1_new = sigmoid(np.dot(new_one, syn0))  # Using the sigmoid function here
print('Новые данные')
print(l1_new)
