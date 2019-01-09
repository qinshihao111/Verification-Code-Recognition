import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import model
import read_dataset
from read_dataset import *
batch_size = 1
test_num = 10
x = tf.placeholder(tf.float32,[batch_size,60,160,3])
y = tf.placeholder(tf.int32,[batch_size,4])
keep_prob =tf.placeholder(tf.float32)
pool5 = model.vgg16_with_bn_model(x,is_training=False)
#pool5 = model.vgg16_model(x)
logit1,logit2,logit3,logit4 = model.inference(pool5,keep_prob)
accu = model.evaluation(logit1,logit2,logit3,logit4,y)
logs_train_dir = './62classes_with_bn_layer/'
saver = tf.train.Saver()
def visualization(label_list,x_batchs,y_batchs,predictions):
    #label_list = Number + alpha + Alpha
    labels = [''.join([label_list[i] for i in label]) for label in y_batchs ]
    preds = [''.join([label_list[i] for i in p]) for p in predictions]
    for i in range(len(x_batchs)):
        cv2.putText(x_batchs[i],'True: {}'.format(labels[i]),(0,10),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,255),1)
        cv2.putText(x_batchs[i],'Pred: {}'.format(preds[i]),(0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1)
        cv2.imshow('1',x_batchs[i])
        cv2.imwrite(logs_train_dir + 'test' + str(j) + '.jpg',x_batchs[i])
        #cv2.waitKey(2000)
        #plt.show()

with tf.Session() as sess:
    print ("Reading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    #print(ckpt)
    cout = 0
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
    accuarcy = []
    label_list = Number + alpha + Alpha
    for j in range (test_num):
        x_batchs, y_batchs = read_dataset.gen_batch(batch_size)
        _,pre1,pre2,pre3,pre4,acc= sess.run([pool5,logit1,logit2,logit3,logit4,accu], feed_dict={x: x_batchs,y: y_batchs,keep_prob:1.0})
        predictions = np.transpose(np.vstack((np.argmax(pre1,1),np.argmax(pre2,1),np.argmax(pre3,1),np.argmax(pre4,1))))
        #visualization(label_list,x_batchs,y_batchs,predictions)
        accuarcy.append(acc)
        if acc==1:
            cout +=1
    print('共测试{}张图片，'.format(test_num * batch_size))
    print('每一位字符的准确率为： ',np.mean(accuarcy))
    print('平均准确率： ',cout/test_num)
#print(cout/test_num)