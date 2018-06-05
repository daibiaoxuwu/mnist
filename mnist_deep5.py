# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import mnistreader
import mnistreaderout
import numpy as np

import tensorflow as tf

FLAGS = None
import os
import sys
log_dir='ckpt-deep25/'
os.environ["CUDA_VISIBLE_DEVICES"]=""#环境变量：使用第一块gpu


batch_size=50
def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv2)

  # Dropout layer
  keep_prob = tf.placeholder(tf.float32)
  with tf.name_scope('dropout'):
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)

  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool1_drop, W_conv3) + b_conv3)

  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([5, 5, 64, 32])
    b_conv4 = bias_variable([32])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
  with tf.name_scope('dropout2'):
    h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)
  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv4_drop)
  with tf.name_scope('dropout3'):
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    h_flat = tf.reshape(h_pool2_drop, [-1, 7 * 7 * 32])

    W_fc1 = weight_variable([7 * 7 * 32, 128])
    b_fc1 = bias_variable([128])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout4'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([128, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def main(_):
  # Import data
  data_sets=mnistreader.reader(patchlength=0,\
            maxlength=300,\
            embedding_size=100,\
            num_verbs=10,\
            allinclude=False,\
            shorten=False,\
            shorten_front=False,\
            testflag=False,\
            passnum=0,\
            dpflag=False)

  

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        model_file=tf.train.latest_checkpoint(log_dir)
        print('loading from:',model_file)
        saver.restore(sess,model_file)
    except:
        print('checkpoint not found. will be saving to:',model_file)
    for i in range(400000):
#    if False:
      inputs,answers=data_sets.list_tags(batch_size,test=False)
#      crossloss,_=sess.run([cross_entropy,train_step],feed_dict={x: inputs, y_: answers, keep_prob: 0.5})
      if i % 10 == 0:
        train_accuracy,train_loss = sess.run([accuracy,cross_entropy],feed_dict={
            x: inputs, y_: answers, keep_prob: 0.5})
        print('step %d, training accuracy %g loss %g' % (i, train_accuracy,train_loss))
      if i%100==99:
        testpointer=data_sets.pointer
        data_sets.pointer=int(data_sets.readlength*5/6)
        acc=0
        cnt=0
        while data_sets.pointer!=data_sets.readlength:  
            cnt+=1
            inputs,answers=data_sets.list_tags(batch_size,test=True)
            train_accuracy = accuracy.eval(feed_dict={
                x: inputs, y_: answers, keep_prob: 1.0})
            acc+=train_accuracy
        acc=acc/cnt
        data_sets.pointer=testpointer
        print('step %d, cnt:%d, test accuracy %g' % (i, cnt, acc))
        print('saved to',saver.save(sess,log_dir+'model.ckpt',global_step=i))
      train_step.run(feed_dict={x: inputs, y_: answers, keep_prob: 0.7})
#    print('test accuracy %g' % accuracy.eval(feed_dict={
#      x: inputs, y_: answers, keep_prob: 1.0}))
    with open('submission6.csv','w') as f:
        f.write('ImageId,Label\n')
        data_sets=mnistreaderout.reader()
        for step in range(560):
#          print(step,data_sets.pointer)
          inputs,answers=data_sets.list_tags(50,test=True)
#          inputs2=[]
#          for i  in range(len(inputs)):
#              inputs2.append(inputs[i]/255)
          feed_dict = { x: inputs, y_: answers, keep_prob:1.0}
          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.
          anst=sess.run([y_conv], feed_dict=feed_dict)[0]
          for i in range(len(anst)):
              f.write(str(data_sets.pointer-batch_size+i+1)+','+str(np.argmax(anst[i]))+'\n')
          if(data_sets.pointer>100000000 ):
              print('anst:',np.argmax(anst[0]),' gen:',data_sets.pointer,' step:',step)
              for i2 in range(784):
                  if(inputs[0][i2]!=0):
                      print('1',end=' ');
                  else:
                      print(' ',end=' ');
                  if(i2%28==0): print(' ');
              input()
    '''         
      print(inputs.shape,answers)
      input()
      for k1 in range(50):
          for k2 in range(784):
              if(inputs[k1][k2]>0.5):print('1',end=' ')
              else:print(' ',end=' ')
              if(k2%28==27):print('')
          print(answers[k1])
      if i % 10 == 0:
            print('loss:',crossloss,end=' ')
            do_evalfake(sess,
                    accuracy,data_sets,batch_size,
                    x,
                    y_,
                    y_conv,keep_prob)
            sys.stdout.flush()
            if i%100==0:
                print('saved to',saver.save(sess,log_dir+'model.ckpt',global_step=i))
      '''
      

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
