import cv2
import os
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from load_file import img_file
text_item = {'pos':(0,0),'color':(0, 255, 0),'type': 4}
retangle_item = {'pos':(100,100),'pos_end':(400,400),'color':(0, 255, 0),'type': 4,'thickness':1}

# create the subtractor

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=100, detectShadows=False)


def getPerson(frame, opt=1):
    # get the front mask
    mask = fgbg.apply(frame)
    # eliminate the noise
    line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)
    # find the max area contours
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    out, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area < 150:
            continue
        rect = cv2.minAreaRect(contours[c])
        cv2.ellipse(image, rect, (0, 255, 0), 2, 8)
        cv2.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
    return image, mask


def process_image(img):
    # img = cv2.GaussianBlur(img,(5,5),0)
    frame = cv2.resize(img, (100, 100))
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # downsize it to reduce processing time
    # cv2.imshow("original",frame)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert from RGB to HSV
    # print(frame.shape)
    # tuned settings
    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    #    print(skinMask.shape)
    #    print(type(skinMask))
    #    cv2.imshow("masked",skinMask)
    # img = cv2.medianBlur(img, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(skinMask, kernel)
    img = cv2.erode(img, kernel)
    edges = cv2.Canny(img, 30, 70)  # canny边缘检测

    return edges
def skinMask(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
    (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5,5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处
    res = cv2.bitwise_and(roi,roi, mask = skin)
    kernel = np.ones((3, 3), np.uint8)  # 设置卷积核
    erosion = cv2.erode(res, kernel)  # 腐蚀操作
    dilation = cv2.dilate(erosion, kernel)  # 膨胀操作
    return res

def appliaction():
    loador = img_file()
    loador.load('F:\Dataset')
    # 生成虚拟数据
    [x, y] = loador.get_data()
    x_train = np.array(x)
    y_train = np.array(y)
    mod = load_model(r'./data.h5')
    capture = cv2.VideoCapture(0)
    capture.set(3, 480)
    while capture.isOpened():
        # 摄像头打开，读取图像
        flag, image = capture.read()
        cv2.rectangle(image, retangle_item['pos'],retangle_item['pos_end'], retangle_item['color'],thickness= retangle_item['thickness'],lineType= retangle_item['type'])

        img_pre = image[100:400,100:400]
        img_pre = skinMask(img_pre)
        img_pre = cv2.resize(img_pre, (100, 100))
        img_pre = img_pre.astype(np.float32) / 255
        img_pre = img_pre.reshape((1,100,100,3))
        img_pre = np.array(img_pre)

        print(img_pre.shape)
        preds = mod.predict(img_pre)
        pred = np.argmax(preds, axis=1)
        print(preds)
        cv2.putText(img_pre[0], 'pred:' + str(pred), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("image", img_pre[0])
        k = cv2.waitKey(1)
        if k == ord('s'):
            cv2.imwrite("test.jpg", img_pre)
        elif k == 0x1b:
            break
    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()