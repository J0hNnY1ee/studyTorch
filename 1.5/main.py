import torch
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库
from torch import optim  # 导入 PyTorch 的优化器模块
from torch.autograd import Variable  #导入 Variable 类，用于自动微分
from torch.utils.data import DataLoader  # 导入 DataLoader 类，用于数据加载
from torchvision.datasets import mnist  # 导入 MNIST 数据集
from torchvision import transforms  # 导入 transforms 模块，用于数据预处理
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, in_c=784, out_c=10):
        super(Net, self).__init__()
        # 定义第一个全连接层，输入维度为 784（28x28 图像展平），输出维度为 512
        self.fc1 = nn.Linear(in_c, 512)
        # 定义第一个激活函数 ReLU
        self.act1 = nn.ReLU(inplace=True)
        
        # 定义第二个全连接层，输入维度为 512，输出维度为 256
        self.fc2 = nn.Linear(512, 256)
        # 定义第二个激活函数 ReLU
        self.act2 = nn.ReLU(inplace=True)
        
        # 定义第三个全连接层，输入维度为 256，输出维度为 128
        self.fc3 = nn.Linear(256, 128)
        # 定义第三个激活函数 ReLU
        self.act3 = nn.ReLU(inplace=True)
        
        # 定义第四个全连接层，输入维度为 128，输出维度为 10（对应 10 个数字分类）
        self.fc4 = nn.Linear(128, out_c)
        
    def forward(self, x):
        # 前向传播过程
        x = self.act1(self.fc1(x))  # 通过第一个全连接层和 ReLU 激活函数
        x = self.act2(self.fc2(x))  # 通过第二个全连接层和 ReLU 激活函数
        x = self.act3(self.fc3(x))  # 通过第三个全连接层和 ReLU 激活函数
        x = self.fc4(x)  # 通过第四个全连接层（输出层）
        return x  # 返回输出结果
    
# 实例化模型
net = Net()

# 准备数据集
# 加载训练集，数据将下载到 './data' 目录，并转换为张量格式
train_set = mnist.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
# 加载测试集，数据将下载到 './data' 目录，并转换为张量格式
test_set = mnist.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
# 创建训练集数据加载器，批量大小为 64，数据顺序将被打乱
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
# 创建测试集数据加载器，批量大小为 128，数据顺序不变
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# 可视化数据
import random  # 导入 random 模块
for i in range(4):  # 随机选择 4 个样本进行可视化
    ax = plt.subplot(2, 2, i+1)  # 创建 2x2 子图
    idx = random.randint(0, len(train_set))  # 随机选择一个样本索引
    digit_0 = train_set[idx][0].numpy()  # 获取样本图像并转换为 NumPy 数组
    digit_0_image = digit_0.reshape(28, 28)  # 将图像展平为 28x28
    ax.imshow(digit_0_image, interpolation="nearest")  # 显示图像
    ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')  # 设置标题为标签
plt.savefig("./1.5/data.png")  # 保存图像为 data.png

# 定义损失函数--交叉熵
criterion = nn.CrossEntropyLoss()

# 定义优化器---随机梯度下降，学习率为 0.01，权重衰减为 0.0005
optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)

# 开始训练
# 记录训练损失
losses = []
# 记录训练精度
acces = []
# 记录测试损失
eval_losses = []
# 记录测试精度
eval_acces = []
# 设置迭代次数为 20
nums_epoch = 20
for epoch in range(nums_epoch):
    train_loss = 0  # 初始化训练损失
    train_acc = 0  # 初始化训练精度
    net = net.train()  # 设置模型为训练模式
    for batch, (img, label) in enumerate(train_data):
        img = img.reshape(img.size(0), -1)  # 将图像展平为一维向量
        img = Variable(img)  # 将图像转换为 Variable
        label = Variable(label)  # 将标签转换为 Variable

        # 前向传播
        out = net(img)  # 通过网络获取输出
        loss = criterion(out, label)  # 计算损失
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)  # 获取预测结果
        num_correct = (pred == label).sum().item()  # 计算正确预测的数量
        acc = num_correct / img.shape[0]  # 计算准确率

        if (batch + 1) % 200 == 0:  # 每 200 个批次打印一次信息
            print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                         batch + 1,
                                                                                         loss.item(),
                                                                                         acc))
        train_acc += acc  # 累加每个批次的准确率

    losses.append(train_loss / len(train_data))  # 记录每个 epoch 的平均训练损失
    acces.append(train_acc / len(train_data))  # 记录每个 epoch 的平均训练准确率

    eval_loss = 0  # 初始化测试损失
    eval_acc = 0  # 初始化测试精度
    net = net.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在评估时不计算梯度
        for img, label in test_data:
            img = img.reshape(img.size(0), -1)  # 将图像展平为一维向量
            img = Variable(img)  # 将图像转换为 Variable
            label = Variable(label)  # 将标签转换为 Variable

            out = net(img)  # 通过网络获取输出
            loss = criterion(out, label)  # 计算损失
            # 记录误差
            eval_loss += loss.item()

            _, pred = out.max(1)  # 获取预测结果
            num_correct = (pred == label).sum().item()  # 计算正确预测的数量
            acc = num_correct / img.shape[0]  # 计算准确率

            eval_acc += acc  # 累加每个批次的准确率
    eval_losses.append(eval_loss / len(test_data))  # 记录每个 epoch 的平均测试损失
    eval_acces.append(eval_acc / len(test_data))  # 记录每个 epoch 的平均测试准确率

    print('[INFO] Epoch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f} | Test: Loss-{:.4f}, Accuracy-{:.4f}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
        eval_acc / len(test_data)))  # 打印每个 epoch 的训练和测试结果

# 绘制训练和测试的损失及准确率
plt.figure()
plt.suptitle('Test', fontsize=12)
ax1 = plt.subplot(1, 2, 1)
ax1.plot(eval_losses, color='r')  # 绘制测试损失曲线
ax1.plot(losses, color='b')  # 绘制训练损失曲线
ax1.set_title('Loss', fontsize=10, color='black')
ax2 = plt.subplot(1, 2, 2)
ax2.plot(eval_acces, color='r')  # 绘制测试准确率曲线
ax2.plot(acces, color='b')  # 绘制训练准确率曲线
ax2.set_title('Acc', fontsize=10, color='black')
plt.savefig("./1.5/result.svg")  # 保存结果图像为 result.svg
