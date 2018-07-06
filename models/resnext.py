import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

depth=64
cardinality=8
def Conv2D(input,filter,kernel_size,stride,padding,name):
    return tf.layers.conv2d(inputs=input,filters=filter,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=False,kernel_initializer=tf.contrib.layers.xavier_initializer(),name=name)


class ResNext():
    def __init__(self,x,is_train):
        self.is_train=is_train
        self.model=self.Build_ResNext(x)

    def first_layer(self,x,scope):
        with tf.variable_scope(scope):
            x=Conv2D(x,64,3,1,'SAME','conv1')
            x=tf.layers.batch_normalization(x,training=self.is_train,name=scope+'_bn1')
            x=tf.nn.relu(x)
            return x

    def transform_layer(self,x,stride,scope):
        with tf.variable_scope(scope):
            x=Conv2D(x,depth,1,stride,'SAME','conv1')
            x=tf.layers.batch_normalization(x,training=self.is_train,name=scope+'bn1')
            x=tf.nn.relu(x)

            x=Conv2D(x,depth,3,1,'SAME','conv2')
            x=tf.layers.batch_normalization(x,training=self.is_train,name='bn2')
            x=tf.nn.relu(x)
            return x

    def transition_layer(self,x,out_dim,scope):
        with tf.variable_scope(scope):
            x=Conv2D(x,out_dim,1,1,'SAME','conv1')
            x=tf.layers.batch_normalization(x,training=self.is_train,name='bn1')
            # x=tf.nn.relu(x)
            return x

    def split_layer(self,x,stride,layer_name):
        with tf.variable_scope(layer_name):
            layers_split=list()
            for i in range(cardinality):
                splits=self.transform_layer(x,stride=stride,scope='splitN_'+str(i))
                layers_split.append(splits)

            return tf.concat(layers_split,axis=3)

    def residual_layer(self,input,out_dim,layer_num,res_block=3):

        for i in range(res_block):
            input_dim=int(np.shape(input)[-1])

            if input_dim*2==out_dim:
                increase_dim=True
                stride=2
                channel=input_dim//2

            else:
                increase_dim=False
                stride=1

            x=self.split_layer(input,stride=stride,layer_name='split_layer_'+layer_num+'_'+str(i))
            x=self.transition_layer(x,out_dim=out_dim,scope='trans_layer_'+layer_num+'_'+str(i))

            if increase_dim==True:
                pad_input=tf.layers.average_pooling2d(input,pool_size=[2,2],strides=2,padding='SAME')
                pad_input=tf.pad(pad_input,[[0,0],[0,0],[0,0],[channel,channel]])
            else:
                pad_input=input

            input=tf.nn.relu(tf.add(pad_input,x))

        return input

    def Build_ResNext(self,input):

        input=self.first_layer(input,scope='first_layer')

        x=self.residual_layer(input,out_dim=64,layer_num='1')
        x=self.residual_layer(x,out_dim=128,layer_num='2')
        x=self.residual_layer(x,out_dim=256,layer_num='3')

        h_w = int(np.shape(x)[1])
        x = tf.layers.average_pooling2d(x,pool_size=(h_w,h_w),strides=(h_w,h_w),padding='VALID')
        x=slim.flatten(x)
        x=tf.layers.dense(x,use_bias=False,units=10)

        return x
#
# def test_graph(logs_dir):
#     input_tensor=tf.constant(np.ones([128,32,32,3]),dtype=tf.float32)
#     result=ResNext(x=input_tensor,is_train=False).model
#     init=tf.initialize_all_variables()
#     sess=tf.Session()
#     sess.run(init)
#     sess.run(result)
#     summary_writer=tf.summary.FileWriter(logs_dir,sess.graph)
#     summary_writer.close()
#     sess.close()
# test_graph(logs_dir='graph')