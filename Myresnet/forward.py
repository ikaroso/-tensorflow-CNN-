import tensorflow as tf
import tensorflow.contrib.slim as slim


ResNet_demo = {
    "layer_50":[{"depth": 256,"num_class": 3},
                {"depth": 512,"num_class": 4},
                {"depth": 1024,"num_class": 6},
                {"depth": 2048,"num_class": 3}],

    "layer_101": [{"depth": 256, "num_class": 3},
                  {"depth": 512, "num_class": 4},
                  {"depth": 1024, "num_class": 23},
                  {"depth": 2048, "num_class": 3}],

    "layer_152": [{"depth": 256, "num_class": 3},
                  {"depth": 512, "num_class": 8},
                  {"depth": 1024, "num_class": 36},
                  {"depth": 2048, "num_class": 3}]
               }

def resblock(input,outdepth,stride):
    x=input
    conv1=slim.conv2d(input,outdepth/4,)
    conv1=tf.layers.batch_normalization(conv1)


    conv2=slim.conv2d(conv1)
    conv2=tf.layers.batch_normalization(conv2)


    conv3=slim.conv2d(conv2)
    conv3=tf.layers.batch_normalization(conv3)


    if  tf.shape(x).as_list(3)!=outdepth:
        x=slim.conv2d(x,outdepth,1,1,activation_fn=None)

    conv3=conv3+x
    result=tf.nn.relu(conv3)
    return result

def build(x,demos):
    conv1=slim.conv2d(x,64,7,2,'SAME',activation_fn=None)
    maxpool1=slim.max_pool2d(conv1,3,2)
    layer_counter=0
    resBlockResult=maxpool1
    for demo in demos:
        name="Layer"+str(layer_counter)
        print(name)
        for i in range(demo["num_class"]):
            print("the  "+str(i)+" times")
            stride=1  #stride是多少??????
            resBlockResult=resblock(resBlockResult,int(demo["depth"]),stride)

    averagepool=slim.avg_pool2d(resBlockResult,2)
    data_shape = averagepool.get_shape().as_list()
    nodes = data_shape[1] * data_shape[2] * data_shape[3]
    averagepool = tf.reshape(averagepool, [-1, nodes])
    fc=slim.fully_connected(averagepool,1000)
    softmax=slim.softmax(fc)

    return softmax

    '''data=averagepool
    #最后全连接层
    data = slim.conv2d(data,num_output,1,activation_fn=None,scope='final_conv')

    data_shape = data.get_shape().as_list()
    nodes = data_shape[1] * data_shape[2] * data_shape[3]
    data = tf.reshape(data, [-1, nodes])'''
