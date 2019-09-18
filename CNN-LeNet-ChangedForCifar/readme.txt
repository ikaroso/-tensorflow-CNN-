  Find out and solve a bunch of questions.

  First,there is some problems in the origin-myresnet, so i seek help from the LeNet. The loss of the origin-myresnet dont decrease at all, i Guess that there is some 
problems in the structure of the network. And it turns out that there are actually probelms of the network itself, since the cifar-10 data runs pretty well in the LEnet
structure.

  Second, in the pre-process of the le-net data, there are some points needed to be remember, and may be used later.
1. We changed it into 3*32*32 at first, then use transpose to make it 32*32*3
2.Before training, we need to divide it by 255, to do the normalization
3. Make sure that you know the difference of ONE-HOT vector and normal result and the function: arg_max()   ----it returns the index,which is a integer


However, problems exist.

1.DownSampling
2.change the channles when use residual blocks  h(x)=x+f(x)
3.How to easily use regularizer in tensorflow.


Improvement of residual blocks

1. I just add 3 residual blocks for test, after 50000 steps with batchsize of 100, using the cifar-10 dataset. The accuracy rise from 61% to 72%. A big improvement.
