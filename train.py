import tensorflow as tf
import read_dataset
import model
import time
import os
import numpy as np
width = 160
height = 60
num_label = 4
batch_size = 32
lr = 0.0001
x_holder = tf.placeholder(tf.float32,[batch_size,height,width,3])
y_holder = tf.placeholder(tf.int32,[batch_size,num_label])
keep_prob = tf.placeholder(tf.float32)
log_dir = './10classes/'
pool5 = model.vgg16_with_bn_model(x_holder,is_training=True)
#pool5 = model.vgg16_model(x_holder)
fc21,fc22,fc23,fc24 = model.inference(pool5,keep_prob)
loss1,loss2,loss3,loss4 = model.losses(fc21,fc22,fc23,fc24,y_holder)
train_loss1,train_loss2,train_loss3,train_loss4 = model.trainning(loss1,loss2,loss3,loss4,lr)
accuarcy = model.evaluation(fc21,fc22,fc23,fc24,y_holder)
images = tf.summary.image('images',x_holder)
loss = loss1 + loss2 + loss3 + loss4
loss = tf.summary.scalar('loss',loss)
summary_all = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(log_dir,sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
for step in range(30000):
    batch_images, batch_labels = read_dataset.gen_batch(batch_size)
    #print(batch_images.shape,batch_labels.shape)
    time1 = time.time()
    feed_dict = {x_holder:batch_images,y_holder:batch_labels,keep_prob:0.5}
    #pre1, pre2, pre3, pre4 = sess.run([fc21,fc22,fc23,fc24],feed_dict = feed_dict)
    #lo1,lo2,lo3,lo4 = sess.run(loss1,loss2,loss3,loss4)
    _,_,_,_,lo1,lo2,lo3,lo4,accu,summ = sess.run([train_loss1,train_loss2,train_loss3,train_loss4,loss1,loss2,loss3,loss4,accuarcy,summary_all],feed_dict)
    train_writer.add_summary(summ,step)
    time2 = time.time()
    duration = time2 - time1
    all_loss = lo1 + lo2 + lo3 + lo4
    if step % 100 ==0:
        # prediction = np.reshape(np.array([pre1, pre2, pre3, pre4]), [-1, 62])
        # max_index = np.argmax(prediction, axis=1)
        # labels = np.reshape(np.transpose(batch_labels), [-1])
        #max_index = sess.run(tf.transpose(tf.reshape(max_index, [-1, batch_size])))
        sec_per_batch = float(duration)
        print('Step %d,train_loss = %.4f,accuarcy = %.4f,sec/batch=%.3f' % (step, all_loss,accu, sec_per_batch))
    if (step+1) % 10000 == 0 :
        checkpoint_path = os.path.join(log_dir, 'model.ckpt')
        saver = tf.train.Saver()
        saver.save(sess, checkpoint_path, global_step = step)
sess.close()





