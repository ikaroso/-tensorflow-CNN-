import tensorflow as tf
import numpy as np
#损失函数一般使用均方误差

BATCH_SIZE=8
seed=23455

rng = np.random.RandomState(seed)

X=rng.rand(3200,2)
Y=[[int(x0+x1<1)] for (x0,x1) in X] #32*1的有标签数据，x1+x2小于1时，Y是1

print(X)
print(Y)

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#LOSS
loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print("UnTrained")
    print(sess.run(w1))
    print(sess.run(w2))
    print()
    steps=15000
    for i in range (steps):
        start=(i*BATCH_SIZE)%3200
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i% 500 ==0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d times, loss is %g" % (i,total_loss))

    print()
    print("w1: "+str(sess.run(w1))) #查看生成的参数
    print("w2: "+str(sess.run(w2)))
