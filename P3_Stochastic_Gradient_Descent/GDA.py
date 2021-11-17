import numpy as np
from matplotlib import pyplot as plt

# config配置区
learningRate = 0.001
epoch = 100

actually_a = 2
actually_b = -1
actually_c = 0.5

a = 0
b = 0
c = 0

data_size = 50

x_data = []
y_data = []

# 按照实际模型添加随机数 构造数据
for x in np.arange(-4, 4, 10.0 / data_size):
    x_data.append(x)
    y_data.append(actually_a * x ** 2 + actually_b * x + +actually_c + np.random.rand() / 10)


def forward(x):
    return a * x * x + b * x + c;


def cost(x_set, y_set):
    cost_val = 0
    for x, y in zip(x_set, y_set):
        cost_val += (forward(x) - y) ** 2
    return cost_val / len(x_set)


def grad_a(x_set, y_set):
    grad = 0;
    for x, y in zip(x_set, y_set):
        grad += (x * x) * (a * x * x + b * x + c - y)
    return 2 * grad / len(x_set)


def grad_b(x_set, y_set):
    grad = 0;
    for x, y in zip(x_set, y_set):
        grad += x * (a * x * x + b * x + c - y)
    return 2 * grad / len(x_set)


def grad_c(x_set, y_set):
    grad = 0;
    for x, y in zip(x_set, y_set):
        grad += a * x * x + b * x + c - y
    return 2 * grad / len(x_set)


cost_list = []

for epoch in range(epoch):
    cost_val = cost(x_data, y_data)
    cost_list.append(cost_val)
    a -= learningRate * grad_a(x_data, y_data);
    b -= learningRate * grad_b(x_data, y_data);
    c -= learningRate * grad_c(x_data, y_data);
    print(f"epoch=%d a=%f b=%f c=%f loss=%f" % (epoch, a, b, c, cost_val))
print("Predict (after training)", 4, forward(4))

plt.plot(np.arange(epoch + 1), np.array(cost_list))
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()
