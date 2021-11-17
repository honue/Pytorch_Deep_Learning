# linear model
import numpy
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# config配置区
actually_w = 2
actually_b = -0.5

data_size = 50

x_data = []
y_data = []

# 按照实际模型添加随机数 构造数据
for i in np.arange(-5, 5, 10.0 / data_size):
    x_data.append(i)
    y_data.append(i * actually_w + actually_b + np.random.rand() / 10)


def forward(x):
    y = w * x + b
    return y


def loss(x, y):
    y_pred_val = forward(x)
    return (y_pred_val - y) * (y_pred_val - y)


w_list = []
b_list = []
mse_list = []

# w b 在这两个范围内尝试  将每一组尝试的结果添加到列表
for w in np.arange(0, 4, 0.01):
    for b in np.arange(-2, 2, 0.01):
        sum_loss_val = 0

        for x, y in zip(x_data, y_data):
            sum_loss_val += loss(x, y)

        mse = sum_loss_val / data_size
        # print(f"w=%f b=%f mse=%f\n" % (w, b, mse))

        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

# 显示3维图
fig = plt.figure()
ax = Axes3D(fig)

w = np.arange(0, 4, 0.01)
b = np.arange(-2, 2, 0.01)

index = mse_list.index(min(mse_list))

print(f"minMSE: w=%f b=%f MSE=%f" % (w_list[index], b_list[index], mse_list[index]))

# 生成网格数据
X, Y = np.meshgrid(w, b)

Z = np.reshape(mse_list, (w.__len__(), b.__len__()))

ax.set_xlabel("b")
ax.set_ylabel("w")
ax.set_zlabel("mse")

ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()
