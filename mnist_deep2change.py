import argparse
import sys
import tempfile

import mnistreader_old2
import mnistreaderout
import numpy as np

import tensorflow as tf

FLAGS = None
import os
import sys
log_dir='ckpt-deep28/'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"#环境变量：使用第一块gpu


batch_size=50
def deepnn(x):
  with tf.name_scope('reshape'):
    layer0 = tf.reshape(x, [-1, 28, 28, 1])

  with tf.name_scope('conv1'):
    weight = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    bias = tf.Variable(tf.truncated_normal([32], stddev=0.1))
    conv = tf.nn.conv2d(layer0, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
    layer1 = tf.nn.relu(conv)

  with tf.name_scope('pool1'):
    layer2 = tf.nn.max_pool(layer1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  with tf.name_scope('conv2'):
    weight = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    bias = tf.Variable(tf.truncated_normal([64], stddev=0.1))
    conv = tf.nn.conv2d(layer2, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
    layer3 = tf.nn.relu(conv)

  with tf.name_scope('pool2'):
    layer4 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  with tf.name_scope('conv3'):
    weight = tf.Variable(tf.truncated_normal([3, 3, 64, 10], stddev=0.1))
    bias = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    conv = tf.nn.conv2d(layer4, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
    layer5 = tf.nn.relu(conv)

  with tf.name_scope('pool3'):
    layer6 = tf.nn.max_pool(layer5, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')

  with tf.name_scope('flatten'):
    layer7 = tf.reshape(layer6, [-1, 10])

  return layer7 


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


def do_evalfake(sess, eval_correct,data_set,batch_size,images_placeholder,labels_placeholder,logits):
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
  steps_per_epoch = data_set.readlength // batch_size // 6
  oldpointer= data_set.pointer
  data_set.pointer=data_set.readlength *5 //6
  
  #steps_per_epoch = data_set.readlength // batch_size 

  num_examples = steps_per_epoch
  print(steps_per_epoch)
  for step in range(steps_per_epoch):
  #  print('pointer1:',data_set.pointer)
    inputs,answers=data_set.list_tags(batch_size,test=True)
    feed_dict= {
                images_placeholder:inputs,
                labels_placeholder:answers,
                }

    newcount,logi=sess.run([eval_correct,logits], feed_dict=feed_dict)
    true_count += newcount
    for i0 in range(batch_size):
            lgans=np.argmax(logi[i0])
            if(lgans!=answers[i0]):
                  for tt in range(784):
                      if(tt%28==0): print(' ');
                      if(inputs[i0][tt]!=0):
                          print('1',end=' ');
                      else:
                          print(' ',end=' ');
#                      print('np',np.argmax(i),answers,answers[i0],'np')
                  print(lgans,answers[i0])
                  input()
            else:
                print('correct')
            # Update the events file.
  precision = float(true_count) / steps_per_epoch
  print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  data_set.pointer=oldpointer
  #print('pointer2:',data_set.pointer)

if __name__ == '__main__':
  # Import data
  data_sets=mnistreader_old2.reader()
  

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # Build the graph for the deep net
  y_conv = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy( labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  '''
    y_conv_norm=tf.nn.l2_normalize(y_conv,[1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_, logits=y_conv_norm)
    cross_entropy_fake = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_, logits=y_conv)
    cross_entropy_fake2 = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy_mean = tf.reduce_mean(cross_entropy)
    '''

  with tf.name_scope('adam_optimizer'):
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)
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
        print(model_file)
        saver.restore(sess,model_file)
    except Exception as e:
        print(e)
    maxacc=0
    minloss=1000
    for i in range(400000):
#    if False:
      inputs,answers=data_sets.list_tags(batch_size)
      if data_sets.pointer<35000 and i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: inputs, y_: answers })
        print('step %d, point %d, training accuracy %g' % (i, data_sets.pointer, train_accuracy))
      elif data_sets.pointer>=35000 and i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: inputs, y_: answers})
        print('step %d, point %d, test accuracy %g' % (i, data_sets.pointer, train_accuracy))
      if i % 1000 == 0:
          print('saved to',saver.save(sess,log_dir+'model.ckpt',global_step=i))

      if data_sets.pointer<35000:  
          train_step.run(feed_dict={x: inputs, y_: answers })

      '''
      inputs,answers=data_sets.list_tags(batch_size,test=False)
      train_step.run(feed_dict={x: inputs, y_: answers })
#      crossloss,_=sess.run([cross_entropy,train_step],feed_dict={x: inputs, y_: answers })
      if i % 10 == 0:
        train_accuracy, lossop = sess.run([accuracy,cross_entropy_mean],feed_dict={x: inputs, y_: answers })
        print('step %d, training accuracy %g loss: %g' % (i, train_accuracy, lossop))
        train_accuracy, lossop, yc, ycn, ce, cor, cfk, cfk2 = sess.run([accuracy,cross_entropy_mean, y_conv, y_conv_norm,cross_entropy, correct_prediction,cross_entropy_fake, cross_entropy_fake2],feed_dict={
            x: inputs, y_: answers })
        print('step %d, training accuracy %g loss: %g' % (i, train_accuracy, lossop))
        print('yconv:',yc[0],len(yc))
        print('yconvnorm:',ycn[0],len(ycn))
        print('yans:',answers[0])
        print('cross:',ce[0])
        print('correct:',cor[0])
        print('convfake:',cfk[0])
        print('convfake2:',cfk2)
        '''
            
#    print('test accuracy %g' % accuracy.eval(feed_dict={
#      x: inputs, y_: answers }))
    with open('submission6.csv','w') as f:
        f.write('ImageId,Label\n')
        data_sets=mnistreaderout.reader()
        for step in range(560):
#          print(step,data_sets.pointer)
          inputs,answers=data_sets.list_tags(batch_size)
#          inputs2=[]
#          for i  in range(len(inputs)):
#              inputs2.append(inputs[i]/255)
          feed_dict = { x: inputs, y_: answers }
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
                    y_conv)
            sys.stdout.flush()
            if i%100==0:
                print('saved to',saver.save(sess,log_dir+'model.ckpt',global_step=i))
      '''
      

