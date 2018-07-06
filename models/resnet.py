import tensorflow as tf

def weight_variable(name,shape):
    with tf.variable_scope(name):
        w=tf.get_variable(name='w',shape=shape,initializer=tf.glorot_uniform_initializer())
        return w

def residual(input,out_channels,stage,block,same_shape=True,is_train=True):
    block_name='res'+str(stage)+block
    in_channels=input.get_shape()[-1]
    with tf.variable_scope(block_name):
        strides=1 if same_shape else 2
        W_conv1=weight_variable('w_conv1',[3,3,in_channels,out_channels])
        out=tf.nn.conv2d(input,W_conv1,strides=[1,strides,strides,1],padding='SAME')
        out = tf.layers.batch_normalization(out, axis=3, training=is_train)
        out=tf.nn.relu(out)

        W_conv2=weight_variable('w_conv2',[3,3,out_channels,out_channels])
        out=tf.nn.conv2d(out,W_conv2,strides=[1,1,1,1],padding='SAME')
        out = tf.layers.batch_normalization(out, axis=3, training=is_train)
        out=tf.nn.relu(out)

        if not same_shape:
            W_shortcut = weight_variable('w_shortcut', [1, 1, in_channels, out_channels])
            input=tf.nn.conv2d(input,W_shortcut,strides=[1,strides,strides,1],padding='VALID')

        add=tf.add(out,input)
        add_result = tf.nn.relu(add)
    return add_result

def resnet18(input,n_claases):
    in_channels=input.get_shape()[-1]
    with tf.variable_scope('reference'):
        is_train=tf.placeholder(tf.bool)
        #stage1
        W_conv1=weight_variable('w_conv1',[3,3,in_channels,32])
        x=tf.nn.conv2d(input,W_conv1,strides=[1,1,1,1],padding='SAME')
        x=tf.layers.batch_normalization(x,axis=3,training=is_train)
        x=tf.nn.relu(x)

        #stage2
        x=residual(x,32,stage=2,block='a',same_shape=True,is_train=is_train)
        x = residual(x, 32, stage=2, block='b', same_shape=True, is_train=is_train)
        x = residual(x, 32, stage=2, block='c', same_shape=True, is_train=is_train)

        #stage3
        x=residual(x,64,stage=3,block='a',same_shape=False,is_train=is_train)
        x=residual(x,64,stage=3,block='b',same_shape=True,is_train=is_train)
        x = residual(x, 64, stage=3, block='c', same_shape=True, is_train=is_train)

        #stage4
        x = residual(x, 128, stage=4, block='a', same_shape=False, is_train=is_train)
        x = residual(x, 128, stage=4, block='b', same_shape=True, is_train=is_train)
        x = residual(x, 128, stage=4, block='c', same_shape=True, is_train=is_train)

        #stage5
        h_w=x.get_shape()[1]
        x=tf.nn.avg_pool(x,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID')
        x=tf.layers.flatten(x)
        x=tf.layers.dense(x,units=n_claases)

        return x,is_train