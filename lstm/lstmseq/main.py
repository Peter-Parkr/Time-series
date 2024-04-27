import numpy as np


def normalise_windows(window_data):  # 数据全部除以最开始的数据再减一
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def no_normalise_windows(window_data):  # 数据全部除以最开始的数据再减一
    normalised_data = []
    for window in window_data:
        normalised_window = [float(p) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()  # 读取文件中的数据
    data = f.split('\n')  # split() 方法用于把一个字符串分割成字符串数组，这里就是换行分割
    sequence_lenghth = seq_len + 1  # #得到长度为seq_len+1的向量，最后一个作为label
    result = []
    for index in range(len(data) - sequence_lenghth):
        result.append(data[index: index + sequence_lenghth])  # 制作数据集，从data里面分割数据
    if normalise_window:
        result = normalise_windows(result)
    else:
        result = no_normalise_windows(result)
    result = np.array(result)  # shape (4121,51) 4121代表行，51是seq_len+1
    row = round(0.9 * result.shape[0])  # round() 方法返回浮点数x的四舍五入值
    train = result[:int(row), :]  # 取前90%
    np.random.shuffle(train)  # shuffle() 方法将序列的所有元素随机排序。
    x_train = train[:, :-1]  # 取前50列，作为训练数据
    y_train = train[:, -1]  # 取最后一列作为标签
    x_test = result[int(row):, :-1]  # 取后10% 的前50列作为测试集
    y_test = result[int(row):, -1]  # 取后10% 的最后一列作为标签
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # 最后一个维度1代表一个数据的维度
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return [x_train, y_train, x_test, y_test]


# x_train, y_train, x_test, y_test = load_data('./video_pixels_li_test.csv', 50, False)
x_train, y_train, x_test, y_test = load_data('./video_pixels_li_test_normalized.csv', 50, False)


# x_train, y_train, x_test, y_test = load_data('./video_pixels_du_de1.csv', 50, True)
# x_train, y_train, x_test, y_test = load_data('./sp500.csv', 50, True)
print('shape_x_train', np.array(x_train).shape)  # shape_x_train (3709, 50, 1)
print('shape_y_train', np.array(y_train).shape)  # shape_y_train (3709,)
print('shape_x_test', np.array(x_test).shape)  # shape_x_test (412, 50, 1)
print('shape_y_test', np.array(y_test).shape)  # shape_y_test (412,)

print("初始化训练参数")
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time

model = Sequential()
# model.add(LSTM(input_dim = 1, output_dim=50, return_sequences=True))
model.add(LSTM(50, input_dim=1, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('linear'))
start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time : ', time.time() - start)

# 训练
print("开始训练")
model.fit(x_train, y_train, batch_size=512, epochs=5, validation_split=0.05)

# 预测
print("开始预测")
import warnings

warnings.filterwarnings("ignore")


def predict_point_by_point(model, data):
    predicted = model.predict(data)  # 输入测试集的全部数据进行全部预测，（412，1）
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


# predictions = predict_point_by_point(model, x_test)
predictions = predict_point_by_point(model, x_train)

import matplotlib.pylab as plt


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


plot_results(predictions, y_train)
