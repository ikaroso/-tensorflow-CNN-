import tensorflow as tf
a=tf.constant([1.0,2.0])
b=tf.constant([3.0,4.0])
result=a+b
print(result)  #计算图只描述运算过程，不运算

x=tf.constant([[1.0,2.0]])
w=tf.constant([[3.0],[4.0]])
y=tf.matmul(x,w)
print(y)

#会话执行运算

with tf.Session() as sess:
    print(sess.run(y))
