import  tensorflow as tf
import  numpy as np
import cifarTest
import forward
import os
from tensorflow.examples.tutorials.mnist import  input_data

BATCH_SIZE=100
LEARNING_RATE_BASE=0.005
LEARNING_RATE_DECAY=0.99
REGULARIZER=0.0001
STEPS=500000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnist_model"


def backward(inputx,labely):
    x=tf.placeholder(tf.float32,[BATCH_SIZE,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE])
    y=forward.forward(x,True,REGULARIZER) #true说明在训练时开启dropout
    global_step=tf.Variable(0,trainable=False)

    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,50000/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    '''
    爆我显存？cnm

    '''
    with tf.Session(config=config) as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(STEPS):

            start=(i*BATCH_SIZE)%50000
            end=start+BATCH_SIZE
            if end >50000:
                continue

            xs=inputx[start:end]
            ys=labely[start:end]

            #reshaped_xs=np.reshape(xs,(BATCH_SIZE,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.NUM_CHANNELS))

            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%100 ==0:
                print("After %d steps, loss is %g" %(step,loss_value))
                #saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
            if i%25000==0:
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    tf.reset_default_graph()
    x,y=cifarTest.getdata()
    x=x/255
    '''这一步数据归一化非常重要!'''
    backward(x,y)

if __name__ == "__main__":
    main()
