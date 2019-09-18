import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import backward
import numpy as np
#这里是验证一下实际测试时候的准确率
TEST_INTERVAL_SECS = 5


def test(mnist):
    with tf.Graph().as_default() as g: #这是绘图
        x = tf.placeholder(tf.float32, [mnist.test.num_examples, forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forward(x, False,None)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #一个cast直接把true和false映射成1 0
        #求平均值，相当于求了准确率

        while True:

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_xs=np.reshape(mnist.test.images,(mnist.test.num_examples,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.NUM_CHANNELS))
                    #从文件名里面直接获得global step
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: mnist.test.labels})
                    print("After %s training steps, test accuracy = %g" %(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()

'''

'''
