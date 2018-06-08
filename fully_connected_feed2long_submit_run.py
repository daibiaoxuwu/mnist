import os
import time
import math
import numpy as np

import tensorflow as tf
from mnistreader import reader

batch_size=50
input_length=784
log_dir='ckpt-deep27'

def test_acc(sess, correctcount,data_set,batch_size,imagein,labelin,keep_prob):
    oldpointer= data_set.pointer
    data_set.pointer=35000
    total=0
    for step in range(140):
#  print('pointer1:',data_set.pointer)
        inputs,answers=data_set.list_tags(batch_size,test=True)
        feed_dict= {
                    imagein:inputs,
                    labelin:answers,
                    keep_prob:0.5
                    }

        newcount=sess.run([correctcount], feed_dict=feed_dict)
        total += newcount[0]
        '''
        for i0 in range(batch_size):
                lgans=np.argmax(logi[i0])
                for i0 in range(batch_size):
                    lgans=np.argmax(logi[i0])
                    if(lgans!=answers[i0] and False):
                          for tt in range(input_length):
                              if(tt%28==0): print(' ');
                              if(inputs[i0][tt]!=0):
                                  print('1',end=' ');
                              else:
                                  print('0',end=' ');
#                      print('np',np.argmax(i),answers,answers[i0],'np')
                          print(lgans,answers[i0])
                          '''
                # Update the events file.
    precision = float(total) / 7000
    print('correct: %d  precision: %f' % (total, precision),end='')
    data_set.pointer=oldpointer

def weight(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(inv, weight):
  outv = tf.nn.conv2d(inv, weight, strides=[1, 1, 1, 1], padding='SAME')
  return outv

def main(_):
    data_sets=reader()
    with tf.Graph().as_default():
        imagein = tf.placeholder(tf.float32, shape=(batch_size, input_length))
        labelin = tf.placeholder(tf.int32, shape=(batch_size))
        with tf.name_scope('reshape'):
            re_image = tf.reshape(imagein, [-1, 28, 28, 1])

          # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            w1 = weight([5, 5, 1, 32])
            b1 = bias([32])
            h1 = tf.nn.relu(conv2d(re_image, w1) + b1)

          # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            hp1 = tf.nn.max_pool(h1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

          # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            w2 = weight([5, 5, 32, 64])
            b2 = bias([64])
            h2 = tf.nn.relu(conv2d(hp1, w2) + b2)

          # Second pooling layer.
        with tf.name_scope('pool2'):
            hp2 = tf.nn.max_pool(h2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

          # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
          # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            w3 = weight([7 * 7 * 64, 1024])
            b3 = bias([1024])

            hp2f = tf.reshape(hp2, [-1, 7 * 7 * 64])
            h3 = tf.nn.relu(tf.matmul(hp2f, w3) + b3)

          # Dropout - controls the complexity of the model, prevents co-adaptation of
          # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h3d = tf.nn.dropout(h3, keep_prob)

          # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            w4 = weight([1024, 10])
            b4 = bias([10])

            y_predict_mid = tf.matmul(h3d, w4) + b4


        y_predict=tf.nn.l2_normalize(y_predict_mid,[1])
        loss=tf.losses.sparse_softmax_cross_entropy(labels=labelin, logits=y_predict)
        correctcount=tf.reduce_sum(tf.cast(tf.nn.in_top_k(y_predict,labelin,1), tf.int32))

        optimizer = tf.train.GradientDescentOptimizer(0.001)
        trainop = optimizer.minimize(loss)


        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                model_file=tf.train.latest_checkpoint(log_dir)
                saver.restore(sess,model_file)
            except Exception as e:
                print(e)

            start_time = time.time()
            for step in range(1000000000):
              inputs,answers=data_sets.list_tags(batch_size,test=False)
              inputs2=[]
              for i  in range(len(inputs)):
                  inputs2.append(inputs[i]/255)
              feed_dict = {
                  imagein: inputs2,
                  labelin: answers,
                  keep_prob:0.5
              }
              _, loss_value, ypr, yprmid , cor = sess.run([trainop, loss, y_predict, y_predict_mid, correctcount], feed_dict=feed_dict)

              duration = time.time() - start_time

              if step % 10 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                print('ypr',ypr[0])
                print('yprmid',yprmid[0])

                '''
                for i0 in range(FLAGS.batch_size):
                    lgans=np.argmax(logi[i0])
                    if(lgans!=answers[i0] and False):
                          for tt in range(784):
                              if(tt%28==0): print(' ');
                              if(inputs[i0][tt]!=0):
                                  print('1',end=' ');
                              else:
                                  print('0',end=' ');
#                      print('np',np.argmax(i),answers,answers[i0],'np')
                          print(lgans,answers[i0])
                '''

              if step % 50 == 0:
                test_acc(sess, correctcount,data_sets,batch_size, imagein, labelin, keep_prob)
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('saved to',checkpoint_file)



if __name__ == '__main__':
    tf.app.run()
