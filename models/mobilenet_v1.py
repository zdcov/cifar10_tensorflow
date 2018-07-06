import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class MobileNet_V1:
    def __init__(self,x,is_train):
        self.is_train=is_train
        self.model=self.forward(x)

    def depthwise_conv2d(self,x,chl_multi,name,stride=1):
        with tf.variable_scope(name):
            in_channels=x.get_shape().as_list()[3]
            filter=tf.get_variable(name='depthwise_weight',shape=(3,3,in_channels,chl_multi),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            x=tf.nn.depthwise_conv2d_native(x,filter,strides=[1,stride,stride,1],padding='SAME',name='conv2d')
            return x

    def pointwise_conv2d(self,x,out_channels,name):
        with tf.variable_scope(name):
            x=tf.layers.conv2d(x,out_channels,(1,1),strides=(1,1),padding='valid',name='point_wise')
            return x

    def conv2d_bn_relu(self,x,out_channels,is_train,stride,name):
        with tf.variable_scope(name):
            x=self.depthwise_conv2d(x,1,'depthwise_conv2d',stride=stride)
            x=tf.layers.batch_normalization(x,training=self.is_train,name='bn1')
            x=tf.nn.relu(x)
            x=self.pointwise_conv2d(x,out_channels,name='point_wise')
            x=tf.layers.batch_normalization(x,training=self.is_train,name='bn2')
            x = tf.nn.relu(x)

            return x

    def forward(self,x):
        with tf.variable_scope('inference'):

            x=tf.layers.conv2d(x,filters=32,kernel_size=3,strides=(1,1),padding='SAME',use_bias=False,kernel_initializer=tf.contrib.layers.xavier_initializer())
            x=tf.layers.batch_normalization(x,training=self.is_train)
            x=tf.nn.relu(x)

            x=self.conv2d_bn_relu(x,64,is_train=self.is_train,stride=1,name='block1')
            x = self.conv2d_bn_relu(x, 128, is_train=self.is_train, stride=2, name='block2')
            x = self.conv2d_bn_relu(x, 128, is_train=self.is_train, stride=1, name='block3')
            x = self.conv2d_bn_relu(x, 256, is_train=self.is_train, stride=2, name='block4')
            x = self.conv2d_bn_relu(x, 256, is_train=self.is_train, stride=1, name='block5')
            x = self.conv2d_bn_relu(x, 512, is_train=self.is_train, stride=2, name='block6')
            x = self.conv2d_bn_relu(x, 512, is_train=self.is_train, stride=1, name='block7')
            x = self.conv2d_bn_relu(x, 512, is_train=self.is_train, stride=1, name='block8')
            x = self.conv2d_bn_relu(x, 1024, is_train=self.is_train, stride=2, name='block9')
            x = self.conv2d_bn_relu(x, 1024, is_train=self.is_train, stride=1, name='block10')

            x=tf.layers.average_pooling2d(x,pool_size=(2,2),strides=(2,2),name='avg')

            x=slim.flatten(x,scope='flatten')

            logits=tf.layers.dense(x,units=10,use_bias=False,name='logits')

            return logits

# def test_graph(logs_dir):
#     input_tensor=tf.constant(np.ones([128,32,32,3]),dtype=tf.float32)
#     result=MobileNet_V1().forward(input_tensor,is_train=False)
#     init=tf.initialize_all_variables()
#     sess=tf.Session()
#     sess.run(init)
#     sess.run(result)
#     summary_writer=tf.summary.FileWriter(logs_dir,sess.graph)
#     summary_writer.close()
#     sess.close()
# test_graph(logs_dir='graph')






