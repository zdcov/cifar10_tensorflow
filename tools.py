import tensorflow as tf
import resnet
import input_data
import os
import numpy as np
def num_correct_prediction(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Return:
      the number of correct predictions
  """
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  correct = tf.cast(correct, tf.int32)
  n_correct = tf.reduce_sum(correct)
  return n_correct

def loss(logits,labels):
    with tf.name_scope('loss'):
        cross_entropy=tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)
    cross_entropy_cost=tf.reduce_mean(cross_entropy)
    return cross_entropy_cost

def accuracy(logits,labels):
    with tf.name_scope('accuracy'):
        correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_op = tf.reduce_mean(correct_prediction)
    return accuracy_op

def train(lr,batch_size,max_step,n_classes):
    train_log_dir='F:\\DL\\mxnet_cifar10\\trian_logs'
    data_dir='F:\\DL\\mxnet_cifar10\\data\\cifar-10-batches-py'
    with tf.name_scope('input'):
        train_img_batch,train_label_batch=input_data.read_cifar10(data_dir,is_train=True,batch_size=batch_size,shuffle=True)
        val_img_batch, val_label_batch = input_data.read_cifar10(data_dir, is_train=False, batch_size=batch_size,
                                                                     shuffle=False)
    features = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
    labels = tf.placeholder(tf.int64, [batch_size, 10])
    logits,is_train = resnet.resnet18(features,n_classes)
    cross_entropy = loss(logits, labels)
    acc=accuracy(logits,labels)
    with tf.name_scope('adam_optimizer'):
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step=tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for step in range(max_step):
                if coord.should_stop():
                    break

                tra_img,tra_label=sess.run([train_img_batch,train_label_batch])
                _,tra_loss,tra_acc=sess.run([train_step,cross_entropy,acc],feed_dict={features:tra_img,labels:tra_label,is_train:True})

                if step%50==0 or (step+1)==max_step:
                    print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))

                if step%200==0 or (step+1)==max_step:
                    val_img,val_label=sess.run([val_img_batch,val_label_batch])
                    val_loss,val_acc=sess.run([cross_entropy,acc],feed_dict={features:val_img,labels:val_label,is_train:False})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))
                if step%2000==0 or (step+1)==max_step:
                    checkpoint_path=os.path.join(train_log_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path,global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


import math


def evaluate(batch_size):
    with tf.Graph().as_default():
        log_dir='F:\\DL\\mxnet_cifar10\\trian_logs'
        test_img_dir='F:\\DL\\mxnet_cifar10\\data\\cifar-10-batches-py'
        n_test=20000
        images,labels=input_data.read_cifar10(data_dir=test_img_dir,is_train=False,batch_size=batch_size,shuffle=False)
        logits,is_trian=resnet.resnet18(images,n_claases=10)
        correct=num_correct_prediction(logits,labels)
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print("No checkpoint file found")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                num_step=int(math.floor(n_test/batch_size))
                num_sample=num_step*batch_size
                step=0
                total_correct=0
                while step < num_step and not coord.should_stop():
                    batch_correct=sess.run(correct,feed_dict={is_trian:False})
                    total_correct+=np.sum(batch_correct)
                    step +=1
                print('Total testing samples: %d' % num_sample)
                print('Total correct predictions: %d' % total_correct)
                print('Average accuracy: %.2f%%' % (100 * total_correct / num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
            coord.join(threads)




train(0.001,64,20000,10)
evaluate(64)

