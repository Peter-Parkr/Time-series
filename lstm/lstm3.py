import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.optim as optim

# 构建视频数据
# video_capture = cv2.VideoCapture(r'D:\pythonProject\laser-dec\data\laserdata0426-3s\de\de13.mp4', cv2.CAP_FFMPEG)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.fc(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# 准备训练数据
def prepare_data(video_file):
    # 读取视频文件
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 从视频中提取帧并将像素值归一化
    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # 调整尺寸和通道顺序
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 归一化像素值
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    return frames


# 数据预处理
def preprocess_data(frames, seq_length):
    X = []
    y = []
    for i in range(len(frames) - seq_length):
        seq = frames[i:i+seq_length]
        target = frames[i+seq_length]
        # 将序列中的帧连接起来形成输入序列
        input_seq = torch.tensor(seq).permute(0, 3, 1, 2)  # 调整维度顺序
        X.append(input_seq)
        # 将目标帧作为输出
        y.append(torch.tensor(target).unsqueeze(0).permute(0, 3, 1, 2))
    return torch.stack(X), torch.stack(y)



# 设置超参数
input_size = 3  # RGB通道数
hidden_size = 64
output_size = 3  # 同样是RGB通道数
seq_length = 10  # 序列长度，即多少帧视频
learning_rate = 0.001
epochs = 10

# 加载数据
video_file = r'D:\pythonProject\laser-dec\data\laserdata0426-3s\de\de13.mp4'

frames = prepare_data(video_file)

# 划分数据集
X, y = preprocess_data(frames, seq_length)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 在验证集上评估模型
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val)
    print(f"Validation Loss: {val_loss.item()}")

# 保存模型
torch.save(model.state_dict(), "lstm_model.pth")
