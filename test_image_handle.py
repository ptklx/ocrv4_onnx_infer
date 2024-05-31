
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

#对图像做锐化增强   
def sharpen_image(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

#膨胀腐蚀增强
def dilation_erode_image(img):
    kernel = np.ones((5,5),np.uint8)
    # 
    dst = cv2.erode(img,kernel,iterations = 1)
    kernel = np.ones((3,3),np.uint8)
    dst = cv2.erode(dst,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.dilate(dst,kernel,iterations = 1)
    kernel = np.ones((3,3),np.uint8)
    dst = cv2.dilate(dst,kernel,iterations = 1)
    return dst

#对图像分块自适应二值实现
def adaptive_threshold_image(img, block_size=11, C=2):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 应用自适应阈值
    # 注意：OpenCV中的adaptiveThreshold函数使用的是1减去C，所以这里使用-C

    thresh_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, block_size, -C
    )
    
    # 返回二值化后的图像
    return thresh_img

if __name__ == '__main__':
    
    image_path=r"D:\file\ocr_detect\snapshot\20240511\rec_img"
    path_list = os.listdir(image_path)

    for path in path_list:
        # image_p = os.path.join(image_path,path)
        image_p = os.path.join(image_path,"pot1.png")


        img = cv2.imread(image_p)
        
        # 使用filter2D函数进行锐化处理
        # out_img = sharpen_image(img)
        out_img=dilation_erode_image(img)

        out_img1=dilation_erode_image(out_img)
        # thresh_img = adaptive_threshold_image(img)

        # 显示原始图像、锐化后的图像和原始与锐化后的对比
        titles = ['img_o', 'img_o1','out_img', 'Comparison']
        images = [ out_img,out_img1,img, np.hstack((img, out_img))]

        for title, image in zip(titles, images):
            cv2.imshow(title, image)
            cv2.waitKey(0)