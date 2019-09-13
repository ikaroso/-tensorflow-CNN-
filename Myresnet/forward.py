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

def sampling(input,ksize=1,stride=2):
    data= input
    if stride  >1 :
        data=slim.max_pool2d(data,ksize,stride=stride)
    return data

#这个conv2d里面存在自主进行的pad0填充，暂时没懂
#而且在卷积时，使用了weight——decay
def conv2d_same(input_tensor,num_outputs,kernel_size,stride,is_train = True,activation_fn=tf.nn.relu,normalizer_fn = True,scope = None):
    data = input_tensor
    if stride is 1:
        data = slim.conv2d(inputs = data,num_outputs = num_outputs,kernel_size = kernel_size,stride = stride,
                           weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None,padding='SAME',scope=scope)
    else:

        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        data = tf.pad(data,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        data = slim.conv2d(inputs = data,num_outputs = num_outputs,kernel_size = kernel_size,stride = stride,
                           weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None,padding='VALID',scope=scope)

    print("Conv ",kernel_size, "depth = ", num_outputs, "  stride = ", stride)
    if normalizer_fn:
        data = tf.layers.batch_normalization(data, training=is_train)
        print("batch_norm")
    if activation_fn is not None:
        data = activation_fn(data)
        print("Relu")

    return data
#好像还需要设置全局的is—training
def resblock(input,outdepth,stride):
    x=input
    conv1=slim.conv2d(input,outdepth//4,)
    conv1=tf.layers.batch_normalization(conv1)


    conv2=slim.conv2d(conv1)
    conv2=tf.layers.batch_normalization(conv2)


    conv3=slim.conv2d(conv2)
    conv3=tf.layers.batch_normalization(conv3)


    if  tf.shape(x).as_list(3)!=outdepth:
        x=slim.conv2d(x,outdepth,1,1,activation_fn=None)
        x=tf.layers.batch_normalization(x)#应该要使用batchnormalization

    else:
        x=sampling(input,stride=stride) #这里的stride是外面传进来的，暂时我不知怎么用

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

    #增加一个batch normalization和一个relu
    resBlockResult=tf.layers.batch_normalization(resBlockResult)
    resBlockResult=tf.nn.relu(resBlockResult)


    averagepool=slim.avg_pool2d(resBlockResult,2)

    data_shape = averagepool.get_shape().as_list()

    nodes = data_shape[1] * data_shape[2] * data_shape[3]

    averagepool = tf.reshape(averagepool, [-1, nodes])

    fc=slim.fully_connected(averagepool,1000)

    softmax=slim.softmax(fc)

    return softmax

'''
    data=averagepool
    #最后全连接层
    data = slim.conv2d(data,num_output,1,activation_fn=None,scope='final_conv')

    data_shape = data.get_shape().as_list()
    nodes = data_shape[1] * data_shape[2] * data_shape[3]
    data = tf.reshape(data, [-1, nodes])

'''
