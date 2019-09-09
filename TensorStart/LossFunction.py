import tensorflow as tf
import numpy as np

#增加了bias 和 activation function

#自定义损失函数

#Cross Entropy 用于求两个概率分布之间的距离
'''
比如我输出的一个二元分类问题的yhat是(0.6,0.4)，而标准答案是(1,0)
则交叉熵为  -(1*log0.6+0*log0.4)=0.222
'''


'''
指数衰减学习率
global step=tf.variavble(0,trainable=False)
learning rate=tf.train.exponential_decay(...)

'''


learning_rate_base=0.1
learning_rate_decay=0.99
learning_rate_step=2  #每两次训练更新一次alpha

global_step=tf.Variable(0,trainable=False)

learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,learning_rate_step,learning_rate_decay,staircase=True)

w=tf.Variable(tf.constant(5,dtype=tf.float32))
loss=tf.square(w+1)

train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        print("%f  %f %f %f "%(sess.run(global_step),sess.run(w),sess.run(learning_rate),sess.run(loss)))

