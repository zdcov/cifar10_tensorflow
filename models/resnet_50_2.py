import tensorflow as tf
import tensorflow.contrib.slim as slim
def Conv2D(input,filter,kenel_size,stride,padding,name):
    return tf.layers.conv2d(inputs=input,filters=filter,kernel_size=kenel_size,strides=stride,padding=padding,use_bias=False,kernel_initializer=tf.contrib.layers.xavier_initializer(),name=name)

def bottleneck(input,out_channels,stage,block,is_train,stride=1):
    block_name='res'+str(stage)+block
    in_channels=input.get_shape().as_list()[-1]
    with tf.variable_scope(block_name):
        out=Conv2D(input,out_channels,[1,1],1,'VALID','conv_1')
        out=tf.layers.batch_normalization(out,axis=-1,training=is_train)
        out=tf.nn.relu(out)

        out=Conv2D(out,out_channels,[3,3],stride,'SAME','conv_2')
        out = tf.layers.batch_normalization(out, axis=-1, training=is_train)
        out = tf.nn.relu(out)

        out=Conv2D(out,out_channels*4,[1,1],1,'VALID','conv_3')
        out = tf.layers.batch_normalization(out, axis=-1, training=is_train)

        if stride != 1 or in_channels != 4*out_channels:
            input=Conv2D(input,out_channels*4,[1,1],stride,'VALID','conv_shortcut')
            input=tf.layers.batch_normalization(input,axis=-1,training=is_train)

        add_result=tf.nn.relu(tf.add(out,input))

    return add_result


def resnet50(input,n_claases):
    with tf.variable_scope('reference'):
        is_train = tf.placeholder(tf.bool)
        keep_pro=tf.placeholder(tf.float32)

        #stage1
        x=Conv2D(input,filter=64,kenel_size=[3,3],stride=1,padding='SAME',name='conv_1')
        x=tf.layers.batch_normalization(x,axis=-1,training=is_train)
        x=tf.nn.relu(x)

        #stage2
        x=bottleneck(x,64,2,'a',is_train=is_train,stride=1)
        x=bottleneck(x,64,2,'b',is_train=is_train,stride=1)
        x=bottleneck(x,64,2,'c',is_train=is_train,stride=1)



        #stage3
        x=bottleneck(x,128,3,'a',is_train=is_train,stride=2)
        x=bottleneck(x,128,3,'b',is_train=is_train,stride=1)
        x = bottleneck(x, 128, 3, 'c', is_train=is_train, stride=1)
        x = bottleneck(x, 128, 3, 'd', is_train=is_train, stride=1)



        #stage4
        x=bottleneck(x,256,4,'a',is_train=is_train,stride=2)
        x = bottleneck(x, 256, 4, 'b', is_train=is_train, stride=1)
        x = bottleneck(x, 256, 4, 'c', is_train=is_train, stride=1)
        x = bottleneck(x, 256, 4, 'd', is_train=is_train, stride=1)
        x = bottleneck(x, 256, 4, 'e', is_train=is_train, stride=1)
        x = bottleneck(x, 256, 4, 'f', is_train=is_train, stride=1)



        #stage5
        x = bottleneck(x, 512, 5, 'a', is_train=is_train, stride=2)
        x = bottleneck(x, 512, 5, 'b', is_train=is_train, stride=1)
        x = bottleneck(x, 512, 5, 'c', is_train=is_train, stride=1)



        #stage5
        x=tf.layers.average_pooling2d(x,pool_size=(4,4),strides=(4,4),padding='VALID')
        # x=tf.layers.flatten(x)
        x=slim.flatten(x)
        x = tf.layers.dense(x, units=n_claases,use_bias=False)

        return x,is_train







