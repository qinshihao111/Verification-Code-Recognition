import cv2
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
import os
import shutil
from PIL import Image
import random
import numpy as np
Number = ['0','1','2','3','4','5','6','7','8','9']
Alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
alpha = list(''.join(Alpha).lower())
def generate_txt(char = None, size = 4, num = 10 ):
    txt = []
    for i in range(num):
        label = [random.sample(char,1)[0] for _ in range(size)]
        label = ''.join(label)
        txt.append(label)
    return txt
def generate_image(Verification_Code_txt,save_path):
    imgs = []
    for i in Verification_Code_txt:
        txt = ImageCaptcha().generate(i)
        img = Image.open(txt)
        img= np.array(img)
        #print(img.shape)
        imgs.append(img)
        #img.save(save_path + i + '.jpg')

    return imgs
def create_dataset(path,trainval_percent,val_percent):
    fa = open('./trainval.txt','w')
    fb = open('./val.txt','w')
    fc = open('./test.txt','w')
    images = os.listdir(path)
    train_list = random.sample(images,int(trainval_percent*len(images)))
    val_list = random.sample(train_list,int(val_percent*len(train_list)))
    for image in images:
        if image in train_list:
            shutil.copy(path + image , './trainval/')
            fa.write( image + '\n')
            if image in val_list:
                shutil.copy(path + image , './val/')
                fb.write( image + '\n')
        else:
            shutil.copy(path + image , './test/')
            fc.write(image + '\n')
    fa.close()
    fb.close()
    fc.close()
#调整图片的大小
def image_resize(image,out_width,out_height,is_color = True):
    out_image = cv2.resize(image,(out_width,out_height))
    if is_color == True and image.ndim ==2:
        out_image = cv2.cvtColor(out_image,cv2.COLOR_GRAY2BGR)
    elif is_color ==False and image.ndim ==3:
        out_image = cv2.cvtColor(out_image,cv2.COLOR_BGR2GRAY)
    return out_image
#获取图片镜像，用来增加数据集
# def image_mirror(image):
#     return cv2.flip(image,1)
def get_images_and_labels(path):
    with open (path,'r') as f:
        lines = f.readlines()
        image_list = [line for line in lines]
        label_list = [line.split('.')[0] for line in lines]
    return image_list,label_list
#train_images,train_labels = get_images_and_labels('./dataset/trainval.txt')
def gen_batch(batch_size):
    label_list = Number #+ alpha + Alpha
    txt = generate_txt(char=label_list, num=batch_size)
    images = generate_image(txt, './image/')

    for i in range(len(txt)):
        temps = list(txt[i])
        txt[i] = [label_list.index(temp) for temp in temps]
    labels = [list(label) for label in txt]
    labels = np.array(labels)
    images = np.array(images)
    return images,labels
# batch_images,batch_labels = gen_batch(batch_size= 10)
# print(batch_labels)
# image = cv2.imread('./dataset/image/yi2s.jpg')
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
# binary = cv2.line(binary,(30,0),(30,60),[0,0,255],1)
# binary = cv2.line(binary,(60,0),(60,60),[0,0,255],1)
# binary = cv2.line(binary,(95,0),(95,60),[0,0,255],1)
# binary = cv2.line(binary,(140,0),(140,60),[0,0,255],1)
# #cv2.imwrite('1.jpg',gray)
# cv2.imwrite('2.jpg',binary)
# #cv2.imshow('1',)