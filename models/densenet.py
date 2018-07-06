import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import math
def Conv2D(input,filter,kernel_size,name,padd='VALID',stride=1):
    return tf.layers.conv2d(inputs=input,filters=filter,kernel_size=kernel_size,strides=stride,padding=padd,use_bias=False,kernel_initializer=tf.contrib.layers.xavier_initializer(),name=name)


class DenseNet():
    def __init__(self,is_train):
        super(DenseNet,self).__init__()
        self.growthRate=12
        self.reduction=0.5
        self.is_train=is_train
        self.n_classes=10

    def Bottleneck(self,x,name):
        with tf.variable_scope(name) as scope:
            out=tf.layers.batch_normalization(x,training=self.is_train)
            out=tf.nn.relu(out)
            out=Conv2D(out,filter=4*self.growthRate,kernel_size=1,name='conv1')

            out=tf.layers.batch_normalization(out,training=self.is_train)
            out=tf.nn.relu(out)
            out=Conv2D(out,filter=self.growthRate,kernel_size=3,padd='SAME',name='conv2')

            out=tf.concat([out,x],axis=3)

            return out

    def Transition(self,x,out_channels,name):
        with tf.variable_scope(name) as scope:
            out = tf.layers.batch_normalization(x, training=self.is_train)
            out = tf.nn.relu(out)
            out = Conv2D(out, filter=out_channels, kernel_size=1, name='conv1')
            out=tf.layers.average_pooling2d(out,pool_size=(2,2),strides=(2,2))

            return out

    def densenet(self,x,n_blocks):
        self.n_block=n_blocks
        in_channels=2*self.growthRate

        out=Conv2D(x,filter=in_channels,kernel_size=3,padd='SAME',name='conv1')

        with tf.variable_scope('block_1') as scope:
            for i in range(n_blocks[0]):
                out=self.Bottleneck(out,'dense_layer.{}'.format(i))
            in_channels+=n_blocks[0]*self.growthRate
            out_channels=int(math.floor(in_channels*self.reduction))
            out=self.Transition(out,out_channels,name='trans_1')
            in_channels=out_channels

        with tf.variable_scope('block_2') as scope:
            for i in range(n_blocks[1]):
                out=self.Bottleneck(out,'dense_layer.{}'.format(i))
            in_channels+=n_blocks[1]*self.growthRate
            out_channels=int(math.floor(in_channels*self.reduction))
            out=self.Transition(out,out_channels,name='trans_2')
            in_channels=out_channels

        with tf.variable_scope('block_3') as scope:
            for i in range(n_blocks[2]):
                out=self.Bottleneck(out,'dense_layer.{}'.format(i))
            in_channels+=n_blocks[2]*self.growthRate

        out=tf.layers.batch_normalization(out,training=self.is_train)
        out=tf.nn.relu(out)
        out=tf.layers.average_pooling2d(out,pool_size=(8,8),strides=(8,8),padding='VALID')
        out=slim.flatten(out)
        out = tf.layers.dense(out, units=self.n_classes,use_bias=False)

        return out

def test_graph(logs_dir):
    input_tensor=tf.constant(np.ones([64,32,32,3]),dtype=tf.float32)
    result=DenseNet(is_train=False).densenet(input_tensor,n_blocks=[3,3,3])
    init=tf.initialize_all_variables()
    sess=tf.Session()
    sess.run(init)
    sess.run(result)
    summary_writer=tf.summary.FileWriter(logs_dir,sess.graph)
    summary_writer.close()
    sess.close()
test_graph(logs_dir='graph')








