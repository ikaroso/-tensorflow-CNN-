import tensorflow as tf
import numpy as np
from PIL import Image
import backward
import forward
from matplotlib import pyplot as plt
import pylab

def restore_model(test_pic_arr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [1, forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.NUM_CHANNELS])
        y = forward.forward(x,False,None)
        pre_value = tf.arg_max(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                pre_value = sess.run(pre_value, feed_dict={x: test_pic_arr})
                return pre_value
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(pic_name):
    '''
    with tf.Session() as sess:

        image_raw_data_jpg = Image.open("a.JPG")
        out = image_raw_data_jpg.resize((28, 28))
        out.save("a.JPG")
        plt.imshow(out,cmap ='gray')
        pylab.show()


        image_raw_data_jpg = tf.gfile.FastGFile('a.jpg', 'rb').read()
        image_data = tf.image.decode_jpeg(image_raw_data_jpg)
        image_data = sess.run(tf.image.rgb_to_grayscale(image_data))
        print (image_data.shape)
        plt.imshow(image_data[:,:,0],cmap ='gray')
        plt.show()
        这个办法好像不行
        '''

    #reshaped_xs=np.reshape(xs,(BATCH_SIZE,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.NUM_CHANNELS))


    img = Image.open("a.JPG")
    re_im = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(re_im.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    plt.imshow(im_arr,cmap ='gray')
    plt.show()

    nm_arr = im_arr.reshape([1, 28,28,1])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready




def application():
    #test_num = int(input("input the num of the test pictures"))
    #for i in range(5):
    #test_pic = input("the path of test picture:")
    test_pic_arr = pre_pic("asd")
    pre_value = restore_model(test_pic_arr)
    print("the prediction num is:", pre_value)


def main():
    application()


if __name__ == "__main__":
    main()
