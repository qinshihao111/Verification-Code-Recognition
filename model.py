import tensorflow as tf
import numpy as np
#定义卷积层
number_classes = 10
def conv_layer(name,input,kernel_size,out_num):
    in_num = input.shape[-1]
    with tf.name_scope(name):
        weights = tf.get_variable(name =  name + '_weights',
                                  shape = [kernel_size,kernel_size,in_num,out_num],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias = tf.get_variable(name = name + '_bias',
                               shape = [out_num],
                               dtype = tf.float32,
                               initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input,weights,strides = [1,1,1,1],padding='SAME')
        conv = tf.nn.bias_add(conv,bias)
        conv = tf.nn.relu(tf.nn.bias_add(conv,bias))
        return conv
#定义最大池化层
def max_pooling(name,input):
        pool = tf.nn.max_pool(input,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME',name = name + '_pooling')
        return pool
#定义batch_norm层
def bn_layer(inputs,is_training,decay = 0.9,eps = 1e-5,name = None):
    shape = inputs.shape[-1]
    with tf.name_scope(name):
        gamma = tf.get_variable(name = name + '_gamma',
                                shape = shape,
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(1))
        beta = tf.get_variable( name = name + '_beta',
                                shape = shape,
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0))
    #计算均值和方差
    axes = list(range(len(inputs.shape)-1))
    mean,var = tf.nn.moments(inputs,axes)
    #滑动平均更新均值与方差
    ema = tf.train.ExponentialMovingAverage(decay)
    def update(mean,var):
        ema_apply_op = ema.apply([mean,var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean),tf.identity(var)
    #训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
    mean,var = tf.cond(tf.equal(is_training,True),lambda: update(mean,var),
                       lambda:(ema.average(mean),ema.average(var)))
    # if is_training == True:
    #     mean,var = update(mean,var)
    # else:
    #     mean = ema.average(mean)
    #     var = ema.average(var)
    #print(type(mean),type(var))
    bn = tf.nn.batch_normalization(inputs,mean,var,beta,gamma,eps,name)
    return bn
#卷积+bn+relu
def conv_bn_layer(inputs,k_size,out_num,is_training,is_act = True,is_bn = True,name = None):
    conv = conv_layer(name,inputs,k_size,out_num)
    if is_bn ==True:
        conv = bn_layer(conv,is_training,name = name + '_bn')
    if is_act == True:
        conv = tf.nn.relu(conv, name = name + '_relu')
    return conv

def vgg16_model(images):
    with tf.name_scope('conv1'):
        conv1_1 = conv_layer(name = 'conv1_1',input=images,kernel_size=3,out_num=16)
        conv1_2 = conv_layer(name= 'conv1_2', input = conv1_1,kernel_size=3,out_num=16)
        pool1 = max_pooling(name = 'pool1',input = conv1_2)
    with tf.name_scope('conv2'):
        conv2_1 = conv_layer(name='conv2_1', input=pool1, kernel_size=3, out_num=32)
        conv2_2 = conv_layer(name='conv2_2', input=conv2_1, kernel_size=3, out_num=32)
        pool2 = max_pooling(name='pool2', input=conv2_2)
    with tf.name_scope('conv3'):
        conv3_1 = conv_layer(name='conv3_1', input=pool2, kernel_size=3, out_num=64)
        conv3_2 = conv_layer(name='conv3_2', input=conv3_1, kernel_size=3, out_num=64)
        conv3_3 = conv_layer(name='conv3_3', input=conv3_2, kernel_size=3, out_num=64)
        pool3 = max_pooling(name='pool3', input=conv3_3)
    with tf.name_scope('conv4'):
        conv4_1 = conv_layer(name='conv4_1', input=pool3, kernel_size=3, out_num=128)
        conv4_2 = conv_layer(name='conv4_2', input=conv4_1, kernel_size=3, out_num=128)
        conv4_3 = conv_layer(name='conv4_3', input=conv4_2, kernel_size=3, out_num=128)
        pool4 = max_pooling(name='pool4', input=conv4_3)
    # with tf.name_scope('conv5'):
    #     conv5_1 = conv_layer(name='conv5_1', input=pool4, kernel_size=3, out_num=128)
    #     conv5_2 = conv_layer(name='conv5_2', input=conv5_1, kernel_size=3, out_num=128)
    #     conv5_3 = conv_layer(name='conv5_3', input=conv5_2, kernel_size=3, out_num=128)
    #     pool5 = max_pooling(name='conv5', input=conv5_3)
    return pool4


def vgg16_with_bn_model(images,is_training):
    with tf.name_scope('conv1'):
        conv1_1 = conv_bn_layer(inputs = images,k_size=3,out_num=16,is_training = is_training,name = 'conv1_1')
        conv1_2 = conv_bn_layer(inputs = conv1_1,k_size=3,out_num=16,is_training = is_training,name = 'conv1_2')
        pool1 = max_pooling(name = 'pool1',input = conv1_2)
    with tf.name_scope('conv2'):
        conv2_1 = conv_bn_layer(inputs = pool1,k_size=3,out_num=32,is_training = is_training,name = 'conv2_1')
        conv2_2 = conv_bn_layer(inputs = conv2_1,k_size=3,out_num=32,is_training = is_training,name = 'conv2_2')
        pool2 = max_pooling(name = 'pool2',input = conv2_2)
    with tf.name_scope('conv3'):
        conv3_1 = conv_bn_layer(inputs = pool2,k_size=3,out_num=64,is_training = is_training,name = 'conv3_1')
        conv3_2 = conv_bn_layer(inputs = conv3_1,k_size=3,out_num=64,is_training = is_training,name = 'conv3_2')
        conv3_3 = conv_bn_layer(inputs=conv3_2, k_size=3, out_num=64, is_training=is_training, name='conv3_3')
        pool3 = max_pooling(name = 'pool3',input = conv3_3)
    with tf.name_scope('conv4'):
        conv4_1 = conv_bn_layer(inputs = pool3,k_size=3,out_num=128,is_training = is_training,name = 'conv4_1')
        conv4_2 = conv_bn_layer(inputs = conv4_1,k_size=3,out_num=128,is_training = is_training,name = 'conv4_2')
        conv4_3 = conv_bn_layer(inputs=conv4_2, k_size=3, out_num=128, is_training=is_training, name='conv4_3')
        pool4 = max_pooling(name = 'pool4',input = conv4_3)
    with tf.name_scope('conv5'):
        conv5_1 = conv_bn_layer(inputs = pool4,k_size=3,out_num=128,is_training = is_training,name = 'conv5_1')
        conv5_2 = conv_bn_layer(inputs = conv5_1,k_size=3,out_num=128,is_training = is_training,name = 'conv5_2')
        conv5_3 = conv_bn_layer(inputs=conv5_2, k_size=3, out_num=128, is_training=is_training, name='conv5_3')
        pool5 = max_pooling(name = 'pool5',input = conv5_3)
    return pool5
def fc(name, input, shape ):
    with tf.variable_scope(name):
        weights = tf.get_variable( name = 'weights',
                                  shape = shape,
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape = [shape[1]],
                                 dtype=tf.float32,
                                 initializer = tf.truncated_normal_initializer(0.1)
                                 )
        fc = tf.matmul(input,weights)+biases
    return fc
def inference(pool5,keep_prob):
    with tf.name_scope('fc1'):
        shape = pool5.shape[1]*pool5.shape[2]*pool5.shape[3]
        fc1 = tf.reshape(pool5,[-1,shape])
        fc1 = tf.nn.dropout(fc1,keep_prob,name = 'fc1_dropout')
    fc21 = fc(name = 'fc21', input = fc1, shape = [shape, number_classes])
    fc22 = fc(name = 'fc22', input = fc1, shape = [shape, number_classes])
    fc23 = fc(name = 'fc23', input = fc1, shape = [shape, number_classes])
    fc24 = fc(name = 'fc24', input = fc1, shape = [shape, number_classes])
    return fc21,fc22,fc23,fc24    #shape = [7,batch_size,65]
def losses(fc21,fc22,fc23,fc24,labels):
    labels = tf.convert_to_tensor(labels,tf.int32)
    # for i in range(4):
    #     name = 'loss' + str(i)
    #     with tf.variable_scope(name) as scope:
    #         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc21, labels=labels[:,0], name='xentropy_per_example')
    #         loss = tf.reduce_mean(cross_entropy, name=name)
    #         tf.summary.scalar(scope.name+'/' + name ,loss)
    with tf.variable_scope('loss1') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc21, labels=labels[:,0], name='xentropy_per_example')
        loss1 = tf.reduce_mean(cross_entropy, name='loss1')
        tf.summary.scalar(scope.name+'/loss1' ,loss1)
    with tf.variable_scope('loss2') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc22, labels=labels[:,1], name='xentropy_per_example')
        loss2 = tf.reduce_mean(cross_entropy, name='loss2')
        tf.summary.scalar(scope.name+'/loss2', loss2)

    with tf.variable_scope('loss3') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc23, labels=labels[:,2], name='xentropy_per_example')
        loss3 = tf.reduce_mean(cross_entropy, name='loss3')
        tf.summary.scalar(scope.name+'/loss3', loss3)

    with tf.variable_scope('loss4') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc24, labels=labels[:,3], name='xentropy_per_example')
        loss4 = tf.reduce_mean(cross_entropy, name='loss4')
        tf.summary.scalar(scope.name+'/loss4', loss4)
    return loss1,loss2,loss3,loss4

def trainning( loss1,loss2,loss3,loss4, learning_rate):
    with tf.name_scope('optimizer1'):
        train_loss1 = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss1)
    with tf.name_scope('optimizer2'):
        train_loss2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss2)
    with tf.name_scope('optimizer3'):
        train_loss3 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss3)
    with tf.name_scope('optimizer4'):
        train_loss4 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss4)
    return train_loss1,train_loss2,train_loss3,train_loss4

def evaluation(fc21,fc22,fc23,fc24,labels):
    logits_all = tf.concat([fc21,fc22,fc23,fc24],0)
    labels = tf.convert_to_tensor(labels,tf.int32)
    labels_all = tf.reshape(tf.transpose(labels),[-1])
    with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits_all, labels_all, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy














