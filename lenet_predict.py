import paddle
from paddle.vision.transforms import Compose, Normalize
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.static import InputSpec
import cv2
import tools
from tools import LeNet

localimage = cv2.imread('dataset/test/0/0_1.jpg')
# 单通道，归一化
# localimage = cv2.cvtColor(localimage, cv2.COLOR_BGR2GRAY)
localimage = cv2.resize(localimage, (64, 64))
# 反相，要黑底白字
localimage = tools.saveRed(localimage)
cv2.imwrite('out.jpg', localimage)
plt.figure(figsize=(2,2))
plt.imshow(localimage)
localimage = (localimage.astype(np.float32) - 127.5) / 127.5


model = paddle.Model(LeNet())   # 用Model封装模型
model.load('model/minst')
model.prepare(metrics=paddle.metric.Accuracy())
# 参数 [-1, 1, 1, 64, 64] 指定了新的形状。在这个形状中，64, 64 是图像的高度和宽度，
# 1 是图像的通道数（因为我们假设这是一个灰度图像，所以通道数为1），另一个 1 是批量大小（因为我们只有一个图像），
# -1 是一个特殊的值，表示该维度的大小应该自动计算以保证总元素数量不变。
img = np.reshape(localimage, [-1, 1, 1, 64, 64])
result = model.predict(img)
print(result)

# 取[(array([[ 2.555041  , -7.8529673 ,  8.123733  ,  1.5727088 , -1.6064267 ,-5.274915  , -5.6212573 , -3.1298964 ,  6.5407977 ,  0.08192763]],dtype=float32),)]中的最大值
result = np.argmax(result[0][0])

# 打印预测结果
print(result)


