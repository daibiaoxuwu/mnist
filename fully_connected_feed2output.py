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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from mnistreader import reader
from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None


os.environ["CUDA_VISIBLE_DEVICES"]=""#环境变量：使用第一块gpu
def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess, eval_correct,data_set,batch_size,images_placeholder,labels_placeholder):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.readlength // FLAGS.batch_size 
  oldpointer= data_set.pointer
  data_set.pointer=data_set.readlength
  print(data_set.pointer)
  #steps_per_epoch = data_set.readlength // FLAGS.batch_size 

  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    inputs,answers=data_set.list_tags(batch_size,test=False)
    feed_dict= {
                images_placeholder:inputs,
                labels_placeholder:answers
                }

    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('fakeeval Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  data_set.pointer=oldpointer




def do_evalfake(sess, eval_correct,data_set,batch_size,images_placeholder,labels_placeholder):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.readlength // FLAGS.batch_size // 6
  oldpointer= data_set.pointer
  data_set.pointer=data_set.readlength *5 //6
  
  #steps_per_epoch = data_set.readlength // FLAGS.batch_size 

  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
  #  print('pointer1:',data_set.pointer)
    inputs,answers=data_set.list_tags(batch_size,test=True)
    feed_dict= {
                images_placeholder:inputs,
                labels_placeholder:answers
                }

    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision),end='')
  data_set.pointer=oldpointer
  #print('pointer2:',data_set.pointer)



def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets=reader(patchlength=0,\
            maxlength=300,\
            embedding_size=100,\
            num_verbs=10,\
            allinclude=False,\
            shorten=False,\
            shorten_front=False,\
            testflag=False,\
            passnum=0,\
            dpflag=False)

  
  

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)
    if True:
        model_file=tf.train.latest_checkpoint(FLAGS.log_dir)
        saver.restore(sess,model_file)

    # Start the training loop.
    start_time = time.time()
    ans=[]
    with open('submission.csv','w') as f:
        f.write('ImageId,Label\n')
        count=0
        for step in range(42000):

          # Fill a feed dictionary with the actual set of images and labels
          # for this particular training step.

        
          inputs,answers=data_sets.list_tags(FLAGS.batch_size,test=False)


          inputs2=[]
          for i  in range(len(inputs)):
              inputs2.append(inputs[i]/255)
          '''
          print(len(inputs2),len(inputs2[0]),inputs2[0])
          input()
          '''
          feed_dict = {
              images_placeholder: inputs2,
              labels_placeholder: answers
          }
          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.
          anst=sess.run([logits], feed_dict=feed_dict)[0]
          for i in anst:

              count+=1
              f.write(str(count)+','+str(np.argmax(i))+'\n')
              ans.append(np.argmax(i))

              if(np.argmax(i)!=answers[0]):
                  for tt in range(784):
                      if(tt%28==0): print(' ');
                      if(inputs[0][tt]!=0):
                          print('1',end=' ');
                      else:
                          print('0',end=' ');
                  print('np',np.argmax(i),answers,answers[0],'np')
                  input()
              print(step,data_sets.pointer,data_sets.readlength,i,np.argmax(i),answers,answers[0],'np')
   
    


def main(_):
#  if tf.gfile.Exists(FLAGS.log_dir):
#    tf.gfile.DeleteRecursively(FLAGS.log_dir)
#  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000000000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='logs/fully_connected_feed2s',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
