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
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')


train_data0, train_label_0 = train_dataset[2][0],train_dataset[2][1]
train_data0 = train_data0.reshape([28,28])
plt.figure(figsize=(2,2))
plt.imshow(train_data0, cmap=plt.cm.binary)
print('train_data0 label is: ' + str(train_label_0))

localimage = cv2.imread('a11.jpg')
localimage = cv2.resize(localimage, (28, 28))
localimage = localimage.astype(np.float32)

# 缩放到28*28

localimage = np.reshape(localimage, [-1, 1, 1, 28, 28])

# input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
model = paddle.Model(LeNet())   # 用Model封装模型
model.load('model/minst')
# 加载测试数据集
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, drop_last=True)
model.prepare(metrics=paddle.metric.Accuracy())
img = np.reshape(localimage, [-1, 1, 1, 28, 28])
# img = np.reshape(train_dataset[2][0], [-1, 1, 1, 28, 28])
result = model.predict(img)
# 将该模型及其所有子层设置为预测模式
model.eval()
# predictions = model.predict(SingleSampleDataset(train_dataset))
predictions = model(test_loader[0])

# 打印预测结果
print(predictions)


