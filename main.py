from opencv import appliaction
import numpy as np
from load_file import img_file
from net import net
from net import net
import keras
import matplotlib.pyplot as plt
def show(x_train,y_train):
    plt.imshow(x_train)
    plt.xlabel(str(y_train))
    plt.show()
if __name__ == "__main__":
    # loador = img_file()
    # loador.load('F:\Dataset')
    # cnn = net()
    # # 生成虚拟数据
    # [x, y] = loador.get_data()
    # x_train = np.array(x)
    # y_train = np.array(y)
    # loador.show_image(1201)
    # cnn.init_model()
    # cnn.train(x_train,y_train)
    appliaction()