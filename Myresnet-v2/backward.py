import tensorflow as tf
import forward
import  numpy as np
import os
import cifarTest
import app

BATCH_SIZE=200
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DECAY=0.99
REGULARIZER=0.0001
STEPS=600000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="cifar10_model"


def backward(inputx,labely):
    x=tf.placeholder(tf.float32,[BATCH_SIZE,32,32,3])
    y_ = tf.placeholder(tf.float32,[None,10])
    y=forward.forward1(x,forward.ResNet_demo["layer_50"],True) #true说明在训练时开启dropout
    global_step=tf.Variable(0,trainable=False)

    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem=tf.reduce_mean(ce)
    loss=cem




    train_step=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE).minimize(loss,global_step)


    saver=tf.train.Saver()

    with tf.Session() as sess:
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

            _,loss_value,step=sess.run([train_step,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%100 ==0 and i!=0:
                print("After %d steps, loss is %g" %(step,loss_value))
            if i%5000000==0 and i!=0:
                app.main()
            if i%10000==0 and i !=0:
                print("saved")
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    tf.reset_default_graph()
    x,y=cifarTest.getdata()
    #print(x.shape)
    #print(y.shape)
    backward(x,y)

if __name__ == "__main__":
    main()
