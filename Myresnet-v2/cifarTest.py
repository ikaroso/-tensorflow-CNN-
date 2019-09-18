import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def test():
    filename="cifar/data_batch_1"

    keys = list(unpickle(filename).keys())
    key = list(unpickle(filename).keys())[2]
    #print(key)
    #print(keys)
    #print(unpickle(filename)[key].shape)

    #有一万行数据，每一行数据是3072, 32*32*3
    number=230
    X=unpickle(filename)[key]
    X=X.reshape(10000, 3, 32, 32)#变成3*32*32是为了方便取出每个通道的数据
    imgs = X[number]
    img0 = imgs[0]
    img1 = imgs[1]
    img2 = imgs[2]
    i0 = Image.fromarray(img0)  # 从数据，生成image对象
    i1 = Image.fromarray(img1)
    i2 = Image.fromarray(img2)
    img = Image.merge("RGB", (i0, i1, i2))
    plt.imshow(img)

    plt.legend()
    plt.show()
    label=unpickle(filename)[b'labels'][number]
    #print(label)


def getdata():

    y1=np.array(unpickle("cifar/data_batch_1")[b'labels'])
    y2=np.array(unpickle("cifar/data_batch_2")[b'labels'])
    y3=np.array(unpickle("cifar/data_batch_3")[b'labels'])
    y4=np.array(unpickle("cifar/data_batch_4")[b'labels'])
    y5=np.array(unpickle("cifar/data_batch_5")[b'labels'])

    y=np.concatenate((y1,y2,y3,y4,y5))
    #print(len(y))
    #y=np.array(y)
    y=dense_to_one_hot(y,10)


    x1=unpickle("cifar/data_batch_1")[b'data']
    x2=unpickle("cifar/data_batch_2")[b'data']
    x3=unpickle("cifar/data_batch_3")[b'data']
    x4=unpickle("cifar/data_batch_4")[b'data']
    x5=unpickle("cifar/data_batch_5")[b'data']
    x=np.concatenate((x1,x2,x3,x4,x5))


    x = x.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")

    #print(x.shape)


    return x,y
def getdata2():
    x1=unpickle("cifar/test_batch")[b'data']
    y1=list(unpickle("cifar/test_batch")[b'labels'])
    y1=np.array(y1)
    y1=dense_to_one_hot(y1,10)
    print(y1.shape)
    return x1,y1
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

if __name__ == '__main__':
    getdata()
