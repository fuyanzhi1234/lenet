import paddle
from paddle.vision.transforms import Compose, Normalize, Resize
import numpy as np
import matplotlib.pyplot as plt
import paddle
import cv2
from paddle.metric import Accuracy
from paddle.static import InputSpec
import tools
from tools import LeNet

class MyDataset(paddle.io.Dataset):
    def __init__(self, txt_path, transform = None):
        super().__init__()
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs 
        self.transform = transform
            
    def __getitem__(self, index):
        imagePath, label = self.imgs[index]
        img = cv2.imread(imagePath)
        img = cv2.resize(img, (64, 64))
        img = tools.saveRed(img)
        img = img.astype('float16')

        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    
    def __len__(self):
        return len(self.imgs)


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
# 使用transform对数据集做归一化
print('download training data and load training data')
# train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
# test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
# 加载自定义数据
train_data = MyDataset('dataset/train/train.txt', transform=transform)
test_data = MyDataset('dataset/test/test.txt', transform=transform)

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载

train_dataloader = paddle.io.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
test_dataloader = paddle.io.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
print('load finished')
for i, (img, labels) in enumerate(train_dataloader): 
    print(img, labels)

train_data0, train_label_0 = train_data[0][0],train_data[0][1]
train_data0 = train_data0.reshape([64,64])
plt.figure(figsize=(2,2))
plt.imshow(train_data0, cmap=plt.cm.binary)
print('train_data0 label is: ' + str(train_label_0))


model = paddle.Model(LeNet())   # 用Model封装模型
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())




# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )

# 训练模型
model.fit(train_dataloader,
        epochs=10,
        batch_size=64,
        verbose=1
        )

model.evaluate(test_dataloader, batch_size=64, verbose=1)

# 保存模型参数
model.save('model/minst')


