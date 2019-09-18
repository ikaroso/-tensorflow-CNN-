import tensorflow as tf
import tensorflow.contrib.slim as slim
WEIGHT_DECAY=0.0001
IMAGE_SIZE=32
NUM_CHANNELS=3
OUTPUT_NODE=10



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

def sampling(input,ksize=1,stride=2):
    data= input
    if stride  >1 :#什么时候会有sride=1呢
        data=slim.max_pool2d(data,ksize,stride=stride)
    return data

'''
我们希望的场景是，pad的多少仅仅根据kernel size来决定，比如说，kernel size是7，那么padding就是[3,3]不要改变了，多pad出来的，没用到也就不要了
'''

#而且在卷积时，使用了weight——decay
def conv2d_same(input_tensor,num_outputs,kernel_size,stride,is_train = True,activation_fn=tf.nn.relu,normalizer_fn = True,scope = None):
    data = input_tensor
    if stride is 1:
        data = slim.conv2d(inputs = data,num_outputs = num_outputs,kernel_size = kernel_size,stride = stride,
                           weights_regularizer=None,activation_fn=None,padding='SAME')
    else:

        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        #使用padding进行填充
        data = tf.pad(data,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

        data = slim.conv2d(inputs = data,num_outputs = num_outputs,kernel_size = kernel_size,stride = stride,
                           weights_regularizer=None,activation_fn=None,padding='VALID',scope=scope)

    #print("Conv ",kernel_size, "depth = ", num_outputs, "  stride = ", stride)
    if normalizer_fn:
        data = tf.layers.batch_normalization(data, training=is_train)
        #print("batch_norm")
    if activation_fn is not None:
        data = activation_fn(data)

        #print("Relu")

    return data
#好像还需要设置全局的is—training
def resblock(input,outdepth,stride,is_train):
    x=input
    conv1=conv2d_same(input,outdepth//4,1,1,is_train,scope="conv1_1x1")
    #conv1=tf.layers.batch_normalization(conv1)


    conv2=conv2d_same(conv1,outdepth//4,3,stride,is_train,scope="conv2_3x3")
    #conv2=tf.layers.batch_normalization(conv2)


    conv3=conv2d_same(conv2,outdepth,1,1,is_train,activation_fn=None,normalizer_fn=False,scope="conv3_1x1")
    #conv3=tf.layers.batch_normalization(conv3)
    #最后一个卷积层为什么不需要activate和normalizer


    if  input.get_shape().as_list()[3]!=outdepth:
        x=slim.conv2d(x,outdepth,1,1,activation_fn=None)
        x=tf.layers.batch_normalization(x,training=is_train)#应该要使用batchnormalization



    else:
        x=sampling(input,stride=stride) #这里的stride是外面传进来的

    conv3=conv3+x
    result=tf.nn.relu(conv3)
    return result

def forward1(x,demos,istrain):
    #这里也用conv2dsame，resnet里所有的卷积层都要用这个
    #conv1=slim.conv2d(x,64,7,2,'SAME',activation_fn=None)
    conv1=conv2d_same(x,64,7,2,istrain,None,False,scope="conv1")#为什么这里不需要激活函数和归一化呢

    #maxpool1=slim.max_pool2d(conv1,3,2,scope="pool1")
    maxpool1=conv1

    layer_counter=0
    resBlockResult=maxpool1

    with tf.variable_scope("resnet") as scope:
        for demo in demos:
            layer_counter=layer_counter+1
            name="Layer"+str(layer_counter)
            #print(name)
            with tf.variable_scope("num_"+str(layer_counter)):
                for i in range(demo["num_class"]):
                    if layer_counter==4:
                        stride=1
                    else:
                        if i==demo["num_class"]-1:
                            stride=2
                        else:
                            stride=1

                    #print("the  "+str(i)+" times")
                    #stride=1  #stride是多少??????
                    with tf.variable_scope("bottleneck"+str(i+1)):
                        resBlockResult=resblock(resBlockResult,int(demo["depth"]),stride,istrain)

    #增加一个batch normalization和一个relu
    resBlockResult=tf.layers.batch_normalization(resBlockResult)
    resBlockResult=tf.nn.relu(resBlockResult)

    #print(resBlockResult.get_shape().as_list())
    averagepool=slim.avg_pool2d(resBlockResult,2,scope="AVGpool")


    data_shape = averagepool.get_shape().as_list()

    nodes = data_shape[1] * data_shape[2] * data_shape[3]

    averagepool = tf.reshape(averagepool, [-1, nodes])
    fc=tf.layers.dense(averagepool,10,activation=tf.nn.relu,trainable=istrain)
    return fc
'''
def forward2():


    fc=slim.fully_connected(averagepool,1000,scope="FC")

    softmax=slim.softmax(fc)

    return softmax
'''

'''
    data=averagepool
    #最后全连接层
    data = slim.conv2d(data,num_output,1,activation_fn=None,scope='final_conv')

    data_shape = data.get_shape().as_list()
    nodes = data_shape[1] * data_shape[2] * data_shape[3]
    data = tf.reshape(data, [-1, nodes])

'''
print("ok")
