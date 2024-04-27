import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Flatten, Bidirectional
from tensorflow.keras.layers import Conv1D, TimeDistributed, MaxPooling1D
from tensorflow.keras.layers import ConvLSTM2D


class UnivariateModels:
    '''
    单变量时间序列LSTM模型
    '''

    def __init__(self, sequence, test_seq, n_seq, n_steps, sw_width, features, epochs_num, verbose_set, flag):
        self.sequence = sequence
        self.test_seq = test_seq
        self.sw_width = sw_width
        self.features = features
        self.epochs_num = epochs_num
        self.verbose_set = verbose_set
        self.flag = flag
        self.X, self.y = [], []

        self.n_seq = n_seq
        self.n_steps = n_steps

    def split_sequence(self):
        for i in range(len(self.sequence)):
            # 找到最后一个元素的索引
            end_index = i + self.sw_width
            # 如果最后一个滑动窗口中的最后一个元素的索引大于序列中最后一个元素的索引则丢弃该样本
            if end_index > len(self.sequence) - 1:
                break

            # 实现以滑动步长为1（因为是for循环），窗口宽度为self.sw_width的滑动步长取值
            seq_x, seq_y = self.sequence[i:end_index], self.sequence[end_index]
            self.X.append(seq_x)
            self.y.append(seq_y)

        self.X, self.y = np.array(self.X), np.array(self.y)

        for i in range(len(self.X)):
            print(self.X[i], self.y[i])

        if self.flag == 1:
            self.X = self.X.reshape((self.X.shape[0], self.n_seq, self.n_steps, self.features))
        elif self.flag == 2:
            self.X = self.X.reshape((self.X.shape[0], self.n_seq, 1, self.n_steps, self.features))
        else:
            self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.features))

        print('X:\n{}\ny:\n{}\n'.format(self.X, self.y))
        print('X.shape:{}, y.shape:{}\n'.format(self.X.shape, self.y.shape))
        return self.X, self.y

    def vanilla_lstm(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(self.sw_width, self.features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        print(model.summary())

        history = model.fit(self.X, self.y, epochs=self.epochs_num, verbose=self.verbose_set)
        print('\ntrain_acc:%s' % np.mean(history.history['accuracy']),
              '\ntrain_loss:%s' % np.mean(history.history['loss']))
        print('yhat:%s' % (model.predict(self.test_seq)), '\n-----------------------------')

    #         model = Sequential()
    #         model.add(LSTM(50, activation='relu', input_shape=(self.sw_width, self.features),
    #                        # 其它参数配置
    #                        recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
    #                        recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
    #                        recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    #                        bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2))

    #         model.add(Dense(units=1,
    #                         # 其它参数配置
    #                         activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    #         model.compile(optimizer='adam', loss='mse',
    #                       # 其它参数配置
    #                       metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    #         print(model.summary())

    #         history = model.fit(self.X, self.y, self.epochs_num, self.verbose_set,
    #                   # 其它参数配置
    #                   callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None,
    #                   initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

    #         model.predict(self.test_seq, verbose=self.verbose_set,
    #                       # 其它参数配置
    #                       steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    def stacked_lstm(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True,
                       input_shape=(self.sw_width, self.features)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        history = model.fit(self.X, self.y, epochs=self.epochs_num, verbose=self.verbose_set)
        print('\ntrain_acc:%s' % np.mean(history.history['accuracy']),
              '\ntrain_loss:%s' % np.mean(history.history['loss']))
        print('yhat:%s' % (model.predict(self.test_seq)), '\n-----------------------------')

    def bidirectional_lstm(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'),
                                input_shape=(self.sw_width, self.features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        history = model.fit(self.X, self.y, epochs=self.epochs_num, verbose=self.verbose_set)
        print('\ntrain_acc:%s' % np.mean(history.history['accuracy']),
              '\ntrain_loss:%s' % np.mean(history.history['loss']))
        print('yhat:%s' % (model.predict(self.test_seq)), '\n-----------------------------')

    def cnn_lstm(self):
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                  input_shape=(None, self.n_steps, self.features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        history = model.fit(self.X, self.y, epochs=self.epochs_num, verbose=self.verbose_set)
        print('\ntrain_acc:%s' % np.mean(history.history['accuracy']),
              '\ntrain_loss:%s' % np.mean(history.history['loss']))
        print('yhat:%s' % (model.predict(self.test_seq)), '\n-----------------------------')

    def conv_lstm(self):
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu',
                             input_shape=(self.n_seq, 1, self.n_steps, self.features)))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        history = model.fit(self.X, self.y, epochs=self.epochs_num, verbose=self.verbose_set)
        print('\ntrain_acc:%s' % np.mean(history.history['accuracy']),
              '\ntrain_loss:%s' % np.mean(history.history['loss']))
        print('yhat:%s' % (model.predict(self.test_seq)), '\n-----------------------------')


if __name__ == '__main__':
    single_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    sw_width = 3
    features = 1

    n_seq = 2
    n_steps = 2
    epochs = 300
    verbose = 0

    test_seq = np.array([70, 80, 90])
    test_seq = test_seq.reshape((1, sw_width, features))

    UnivariateLSTM = UnivariateModels(single_seq, test_seq, n_seq, n_steps, sw_width, features, epochs, verbose, flag=0)
    UnivariateLSTM.split_sequence()
    UnivariateLSTM.vanilla_lstm()
    UnivariateLSTM.stacked_lstm()
    UnivariateLSTM.bidirectional_lstm()
