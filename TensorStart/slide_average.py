import tensorflow as tf
import numpy as np


#超参
BATCH_SIZE = 8
seed = 23455
LR = 0.1
STEPS = 40
MOVING_AVERAGE_DECAY = 0.99         # 滑动平均衰减率

W1 = tf.Variable(0,dtype=tf.float32)

global_step = tf.Variable(0,trainable = False)

ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

#ema.apply 后的括号里面是更新列表，每次运行sess.run(ema_op)时，对更新列表中的元素求滑动平均值
#在实际应用中，会使用tf.trainable_variables()自动获取所有参数
#ema_op = ema.apply([W1])
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #用ema.average(W1)获取W1滑动平均值(要运行多个节点，作为列表中的元素列出，写在 sess.run中)
    print(sess.run(global_step))
    print(sess.run([W1,ema.average(W1)]))

    #赋初值为1
    print(sess.run(global_step))
    sess.run(tf.assign(W1,1))
    sess.run(ema_op)
    print(sess.run([W1,ema.average(W1)]))

    # 模拟100轮后， W1 变成 10
    sess.run(tf.assign(global_step,100))
    sess.run(tf.assign(W1,10))
    sess.run(ema_op)
    print(sess.run(global_step))
    print(sess.run([W1,ema.average(W1)]))

    # 每次更新一次
    sess.run(ema_op)
    print(sess.run(global_step))
    print(sess.run([W1,ema.average(W1)]))


    sess.run(ema_op)
    print(sess.run(global_step))
    print(sess.run([W1,ema.average(W1)]))

    sess.run(ema_op)
    print(sess.run(global_step))
    print(sess.run([W1,ema.average(W1)]))
#运行一次sess.run(ema_op)就更新一次，但是global_step不会随他变
