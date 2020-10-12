import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import keras.backend as K

import matplotlib.pyplot as plt


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
class net:
    def __init__(self):
        self.model = Sequential()
        self.model_name = 'data.h5'

    def init_model(self):
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        return 0

    def train(self,train_x,train_y):
        history = self.model.fit(train_x, train_y, batch_size=32, epochs=20)
        self.model.save(self.model_name)
        print(history.history.keys())
        # # 绘制训练 & 验证的准确率值
        # plt.plot(history.history['accuracy'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

        # 绘制训练 & 验证的损失值
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        # score = self.model.evaluate(train_x[:20], train_y[:20], batch_size=32)

        # print(score)
        return 0
