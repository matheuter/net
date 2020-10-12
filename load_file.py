from PIL import Image
from glob import glob
import cv2
import matplotlib.pyplot as plt
import keras
import os
import numpy as np
class img_file:
    def __init__(self):
        self.train_x = []
        self.train_y = []

    # 读取目录下所有的jpg图片
    def load(self,file_path):
        for i in range(6):
            self.load_image(file_path,i)
        return 0

    def load_image(self,dirname,label):
        file_name = glob(dirname+'\\'+str(label)+ '\\*')
        for file in file_name:
            im = Image.open(os.path.join(dirname, file))
            # im = skinMask(im)
            im = im.resize((100, 100), Image.ANTIALIAS)

            image_array = np.array(im)
            image_array = image_array.astype(np.float32) / 255
            self.train_y.append(image_array)
            temp_x= keras.utils.to_categorical(label, num_classes= 6)
            self.train_x.append(temp_x)
        print('a')
        return 0
    def show_image(self,index):
        plt.imshow(self.train_y[index])
        plt.xlabel(str(self.train_x[index]))
        plt.show()
    def add_turnned_images(self):
        for i in range(len(self.train_y)):
            self.train_y.append(cv2.flip(self.train_y[i],1))
        for i in range(len(self.train_x)):
            self.train_x.append(self.train_x[i])
    def get_data(self):
        self.add_turnned_images()
        return [self.train_y,self.train_x]
    def get_test(self):
        return [self.train_y[1000:], self.train_x[1000:]]

