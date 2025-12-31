import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt  # For plotting

from CWRUDataset import load_and_process_files
from MyLSTM import MyLSTM

"""
原先的LSTM故障诊断代码，用来测试数据集效果，并且后续需要验证生成数据效果也是用这个看效果

训练效果非常好0.98
"""


# Training parameters
epochs = 300
learning_rate = 0.001

# Model parameters
input_size = 1
hidden_size = 32
num_layers = 2
num_classes = 10
seq_length = 1024


batch_size = 32
# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 配置参数
DATA_DIR = "./data/preprocess_bearing/0HP"
SEGMENT_LENGTH = 1024  # 信号段长度
TRAIN_SAMPLES_PER_CLASS = 20  # 每类训练样本数
TEST_SAMPLES_PER_CLASS = 100 # 每类测试样本数
OVERLAP = True  # 启用样本重叠
STEP_SIZE = 1024
"""
20个样本最好是82.2
"""
train_set, test_set = load_and_process_files(
    data_dir=DATA_DIR,
    segment_length=SEGMENT_LENGTH,
    train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
    test_samples_per_class=TEST_SAMPLES_PER_CLASS,
    overlap=OVERLAP,
    step_size=STEP_SIZE
)
trainDataloader = DataLoader(train_set, batch_size=10, shuffle=True)
testDataloader = DataLoader(test_set, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = MyLSTM(input_size, hidden_size, num_layers, num_classes).cuda()
loss_func = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Store losses and accuracies
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

maxn = 100
minn = 0
# Training loop
for epoch in range(epochs):
    print(f'Epoch [{epoch + 1}/{epochs}]')
    model.train()
    total_train_loss, total_train_correct = 0, 0


    for data in trainDataloader:
        inputs, labels = data
        inputs = inputs.reshape(-1, seq_length, input_size).cuda()

        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_correct += (outputs.argmax(1) == labels).sum().item()

    # Calculate and store train loss and accuracy
    train_loss = total_train_loss / len(trainDataloader) / 10
    train_accuracy = total_train_correct / len(trainDataloader) / 32
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluation on test set
    model.eval()
    total_test_loss, total_test_correct = 0, 0

    with torch.no_grad():
        for data in testDataloader:
            inputs, labels = data
            inputs = inputs.reshape(-1, seq_length, input_size).cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            total_test_loss += loss.item()
            total_test_correct += (outputs.argmax(1) == labels).sum().item()

    # Calculate and store test loss and accuracy
    test_loss = total_test_loss / len(testDataloader) / batch_size
    test_accuracy = total_test_correct / len(testDataloader) / batch_size
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(f'Train Loss={train_loss:.4f}, Train Accuracy={train_accuracy:.4f}')
    print(f'Test Loss={test_loss:.4f}, Test Accuracy={test_accuracy:.4f}')
    if test_accuracy > minn:
        # 保存模型参数
        torch.save(model.state_dict(), 'LSTM_model200.pth')
        print("模型参数已保存到 'LSTM_model200.pth'")
        minn = test_accuracy
        maxn = test_loss

print(f'Finished Training,最优准确率:{minn},最优损失：{maxn}')

# Plotting loss and accuracy curves
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()