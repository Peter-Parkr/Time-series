import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch.optim as optim

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = torch.flatten(out, 1)
        out = out.unsqueeze(0)  # 添加batch维度
        out, _ = self.lstm(out)
        out = self.fc1(out[:, -1, :])
        out = F.relu(out)
        out = self.fc2(out)
        return out


# 超参数
input_size = 40000  # 输入特征维度（视频帧大小）
hidden_size = 256  # LSTM隐藏层大小
num_layers = 50  # LSTM层数
num_classes = 100  # 输出特征维度（视频帧大小）
num_epochs = 50
batch_size = 16
learning_rate = 0.001

# 加载视频并将每一帧的像素值转换为张量
video_capture = cv2.VideoCapture(r'D:\pythonProject\laser-dec\data\laserdata0426-3s\de\de13.mp4', cv2.CAP_FFMPEG)
frames = []
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (100, 100))
    frames.append(resized_frame)


video_capture.release()
frames_array = np.array(frames)
frames_tensor = torch.from_numpy(frames_array).float()

# 划分训练集和测试集
X_train, X_test = frames_tensor[:-1], frames_tensor[1:]

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# 训练模型
# 训练模型
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        frames_batch = X_train[i:i + batch_size].unsqueeze(1).to(device)  # 添加通道维度
        labels_batch = X_train[i + 1:i + batch_size + 1].unsqueeze(1).to(device)

        # 前向传播
        outputs = model(frames_batch)
        loss = criterion(outputs, labels_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    epoch_loss /= len(X_train) / batch_size
    losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    # 更新学习率
    scheduler.step(epoch_loss)

# 绘制损失曲线
plt.plot(losses, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# 使用模型进行预测
with torch.no_grad():
    X_test = X_test.unsqueeze(1).to(device)
    predicted_frames = model(X_test).cpu().numpy()

# 显示预测结果
# for i in range(len(predicted_frames)):
#     plt.plot(X_test[i].cpu().numpy().flatten(), label='Original')
#     plt.plot(predicted_frames[i].flatten(), label='Predicted')
#     plt.xlabel('Pixel')
#     plt.ylabel('Value')
#     plt.title('Prediction vs. Original')
#     plt.legend()
#     plt.show()

# 显示预测结果
# 显示预测结果
for i in range(len(predicted_frames)):
    plt.figure()
    plt.plot(predicted_frames[i].flatten(), label='Predicted')
    plt.xlabel('Pixel')
    plt.ylabel('Value')
    plt.title(f'Predicted Frame (Sample {i+1})')
    plt.legend()
    plt.show()

for i in range(len(predicted_frames)):
    plt.figure()
    plt.plot(X_test[i].cpu().numpy().flatten(), label='Original')
    plt.xlabel('Pixel')
    plt.ylabel('Value')
    plt.title('Original')
    plt.legend()
    plt.show()

