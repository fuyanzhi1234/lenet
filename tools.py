# 输出一个文件夹下面的所有子文件
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn.functional as F

class LeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*14*14, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=2)

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


def print_directory_contents(sPath):
    for sChild in os.listdir(sPath):
        sChildPath = os.path.join(sPath,sChild)
        if os.path.isdir(sChildPath):
            print_directory_contents(sChildPath)
        else:
            print(sChildPath + ' 1')


def readTxt(txt_path):
    fh = open(txt_path, 'r')
    imgs = []
    for line in fh:
        line = line.rstrip()
        words = line.split()
        imgs.append((words[0], int(words[1])))
    print(imgs)
    return imgs

def saveRed(image):
    """
    只保留红色
    :param image:
    :return:
    """

    # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
    grid_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 从RGB色彩空间转换到HSV色彩空间
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    lower1 = np.array([0, 30, 70])
    upper1 = np.array([7, 255, 255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
    # res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

    # H、S、V范围二：
    lower2 = np.array([150, 30, 70])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    # res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

    # 将两个二值图像结果 相加
    mask3 = mask1 + mask2
    
    # 去除左右两边各1个像素的红色，因为开运算无法处理边缘噪声
    # image_h, image_w = mask3.shape
    # mask3[0:image_h, 0:1] = 0
    # mask3[0:image_h, image_w - 1:image_w] = 0
    
    # 获取卷积核，这里使用矩阵的方式
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1, 3))
    # 进行开运算,对于横着的红色噪声，1-2行，可通过卷积核（1,3）去除掉，且对老师手动坚画的红线影响较小
    # mask3 = cv2.morphologyEx(src=mask3, op=cv2.MORPH_OPEN, kernel=kernel, iterations=1)

    return mask3

    
# print_directory_contents('dataset/train/1')
# imgs = readTxt('dataset/train/train.txt')

# image = cv2.imread(imgs[0][0])
# plt.figure(figsize=(2,2))
# plt.imshow(image)
# plt.waitforbuttonpress()