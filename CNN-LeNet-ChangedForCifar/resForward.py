import  tensorflow as tf
import  numpy as np


IMAGE_SIZE=224
NUM_CHANNELS=3

def get_weight(shape,regularizer):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer !=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))

    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))

    return b

def conv2d(x,w,stride):
    return tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding='VALID')


def max_pool(x,k,s):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,s,s,1],padding='VALID')

def resblock(x,preout,w1,w2,w3,regularizer,s):
    conv1_w=get_weight([1,1,preout,w1],regularizer=regularizer)
    conv1_b=get_bias(w1)
    conv1=conv2d(x,conv1_w,s)
    batch1=tf.nn.batch_normalization(conv1)
    relu1=tf.nn.relu(tf.nn.bias_add(batch1,conv1_b))
    relu1=relu1+x

    conv2_w=get_weight([3,3,w1,w2],regularizer=regularizer)
    conv2_b=get_bias(w2)
    conv2=conv2d(relu1,conv2_w,s)
    batch2=tf.nn.batch_normalization(conv2)
    relu2=tf.nn.relu(tf.nn.bias_add(batch2,conv2_b))
    relu2=relu1+relu2

    conv3_w=get_weight([1,1,w2,w3],regularizer=regularizer)
    conv3_b=get_bias(w3)
    conv3=conv2d(relu2,conv3_w,s)
    batch3=tf.nn.batch_normalization(conv3)
    relu3=tf.nn.relu(tf.nn.bias_add(batch3,conv3_b))
    relu3=relu2+relu3

    return relu3

def forward(x , regularizer):
    conv1_w=get_weight([7,7,NUM_CHANNELS,64],regularizer)
    conv1_b=get_bias(64)
    conv1=conv2d(x,conv1_w,2)
    batch1=tf.nn.batch_normalization(conv1)
    relu1=tf.nn.relu(tf.nn.bias_add(batch1,conv1_b))
    pool1=max_pool(relu1,3,2)
    #padding要看一下
    bottleneck1=resblock(pool1,64,64,64,256,regularizer,1)
    #从开始到第一个bottleneck结束里面维度的变化看一下

    return y












