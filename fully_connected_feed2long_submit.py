import os
import time
import math
import numpy as np

import tensorflow as tf
from mnistreader import reader

batch_size=50
input_length=784
log_dir='fcckpt'

def test_acc(sess, eval_correct,data_set,batch_size,images_placeholder,labels_placeholder,logits,keep_prob):
    oldpointer= data_set.pointer
    data_set.pointer=35000
    total=0
    for step in range(140):
#  print('pointer1:',data_set.pointer)
        inputs,answers=data_set.list_tags(batch_size,test=True)
        feed_dict= {
                    images_placeholder:inputs,
                    labels_placeholder:answers,
                    keep_prob:0.5
                    }

        newcount,logi=sess.run([eval_correct,logits], feed_dict=feed_dict)
        total += newcount
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


def main(_):
    data_sets=reader()
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, input_length))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

        with tf.name_scope('conv1'):
            weights = tf.Variable(tf.truncated_normal([input_length, 128], stddev=1.0 / math.sqrt(float(input_length))), name='weights')
            biases = tf.Variable(tf.zeros([128]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights) + biases)
# Hidden 2
        with tf.name_scope('conv2'):
            weights = tf.Variable( tf.truncated_normal([128, 32], stddev=1.0 / math.sqrt(float(128))), name='weights')
            biases = tf.Variable(tf.zeros([32]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(hidden2, keep_prob)
# Linear
        with tf.name_scope('softmax'):
            weights = tf.Variable( tf.truncated_normal([32, 10], stddev=1.0 / math.sqrt(float(32))), name='weights')
            biases = tf.Variable(tf.zeros([10]), name='biases')
            logits = tf.matmul(h_fc1_drop, weights) + biases

        loss=tf.losses.sparse_softmax_cross_entropy(labels=labels_placeholder, logits=logits)
        eval_correct=tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits,labels_placeholder,1), tf.int32))

        tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(loss)


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
                  images_placeholder: inputs2,
                  labels_placeholder: answers,
                  keep_prob:0.5
              }
              _, loss_value,logi = sess.run([train_op, loss,logits], feed_dict=feed_dict)

              duration = time.time() - start_time

              if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
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

              if (step + 1) % 500 == 0:
                test_acc(sess, eval_correct,data_sets,batch_size, images_placeholder, labels_placeholder, logits,keep_prob)
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('saved to',checkpoint_file)



if __name__ == '__main__':
    tf.app.run()
