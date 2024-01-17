import paddle
from paddle.vision.transforms import Compose, Normalize
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.static import InputSpec
import cv2

class LeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

localimage = cv2.imread('a8.jpg')
# 单通道，归一化
localimage = cv2.cvtColor(localimage, cv2.COLOR_BGR2GRAY)
localimage = cv2.resize(localimage, (28, 28))
# 反相，要黑底白字
localimage = 255 - localimage
plt.figure(figsize=(2,2))
plt.imshow(localimage, cmap=plt.cm.binary)
localimage = (localimage.astype(np.float32) - 127.5) / 127.5

# 缩放到28*28

localimage = np.reshape(localimage, [-1, 1, 1, 28, 28])

model = paddle.Model(LeNet())   # 用Model封装模型
model.load('model/minst')
model.prepare(metrics=paddle.metric.Accuracy())
# 参数 [-1, 1, 1, 28, 28] 指定了新的形状。在这个形状中，28, 28 是图像的高度和宽度，
# 1 是图像的通道数（因为我们假设这是一个灰度图像，所以通道数为1），另一个 1 是批量大小（因为我们只有一个图像），
# -1 是一个特殊的值，表示该维度的大小应该自动计算以保证总元素数量不变。
img = np.reshape(localimage, [-1, 1, 1, 28, 28])
result = model.predict(img)
print(result)

# 取[(array([[ 2.555041  , -7.8529673 ,  8.123733  ,  1.5727088 , -1.6028267 ,-5.274915  , -5.6212573 , -3.1298928 ,  6.5407977 ,  0.08192763]],dtype=float32),)]中的最大值
result = np.argmax(result[0][0])

# 打印预测结果
print(result)


