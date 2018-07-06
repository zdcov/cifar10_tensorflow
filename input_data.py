import tensorflow as tf
import numpy as np
import os

def read_cifar10(data_dir,is_train,batch_size,shuffle):
    img_width=32
    img_height=32
    img_depth=3
    label_bytes=1
    img_bytes=img_depth*img_height*img_width
    with tf.name_scope('input'):
        if is_train:
            filenames=[os.path.join(data_dir,'data_batch_%d.bin'%ii) for ii in np.arange(1,6)]
        else:
            filenames=[os.path.join(data_dir,'test_batch.bin')]
        filename_queue=tf.train.string_input_producer(filenames)
        reader=tf.FixedLengthRecordReader(label_bytes+img_bytes)
        key,value=reader.read(filename_queue)
        record_bytes=tf.decode_raw(value,tf.uint8)
        label=tf.slice(record_bytes,[0],[label_bytes])
        label=tf.cast(label,tf.int32)

        img_raw=tf.slice(record_bytes,[label_bytes],[img_bytes])
        img_raw=tf.reshape(img_raw,[img_depth,img_height,img_width])
        img=tf.transpose(img_raw,(1,2,0))
        img=tf.cast(img,tf.float32)

        if is_train:
            img=tf.pad(img,[[4,4],[4,4],[0,0]])
            img=tf.random_crop(img,[32,32,3])
            img=tf.image.random_flip_left_right(img)
        # img=tf.image.random_brightness(img,max_delta=63)
        # img=tf.image.random_contrast(img,lower=0.2,upper=1.8)
        img=img/255.
        # img=tf.image.per_image_standardization(img)

        if shuffle:
            img_batch,label_batch=tf.train.shuffle_batch([img,label],batch_size=batch_size,num_threads=64,capacity=20000,min_after_dequeue=3000)
        else:
            img_batch,label_batch=tf.train.batch([img,label],batch_size=batch_size,num_threads=64,capacity=2000)
        n_classes=10
        label_batch=tf.one_hot(label_batch,depth=n_classes)
        label_batch=tf.cast(label_batch,dtype=tf.int32)
        label_batch=tf.reshape(label_batch,[batch_size,n_classes])

        return img_batch,label_batch


# import matplotlib.pyplot as plt
# data_dir='F:\DL\mxnet_cifar10\data\cifar-10-batches-py'
# batch_size=10
# img_batch,label_batch=read_cifar10(data_dir,is_train=True,batch_size=batch_size,shuffle=True)
#
# with tf.Session() as sess:
#     i=0
#     coord=tf.train.Coordinator()
#     threads=tf.train.start_queue_runners(coord=coord)
#     try:
#         while not coord.should_stop() and i<1:
#             img,label=sess.run([img_batch,label_batch])
#             print("!!!!")
#             for j in np.arange(batch_size):
#                 print(label[j])
#                 plt.imshow(img[j,:,:,:])
#                 plt.show()
#             i+=1
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)

