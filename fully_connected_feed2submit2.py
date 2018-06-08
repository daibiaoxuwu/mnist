import os
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf

batch_size=50
input_length=784
log_dir='fcckpt'
pointer=0
readin=pd.read_csv('train.csv').values #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。

images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, input_length))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

with tf.name_scope('cv1'):
    weights = tf.Variable(tf.truncated_normal([input_length, 128], stddev=1.0 / math.sqrt(float(input_length))), name='weights')
    biases = tf.Variable(tf.zeros([128]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights) + biases)
# Hidden 2
with tf.name_scope('cv2'):
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
      tempread=readin[pointer:pointer+50]
      inputs=tempread[:,1:]
      answers=tempread[:,0]
      pointer+=50
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

      if pointer >= 35000:
        total=0
        for step in range(140):
            tempread=readin[pointer:pointer+50]
            inputs=tempread[:,1:]
            answers=tempread[:,0]
            pointer+=50
            newcount=sess.run([eval_correct,logits], feed_dict= { images_placeholder:inputs, labels_placeholder:answers, keep_prob:0.5 } )
            total += newcount
        precision = float(total) / 7000
        print('correct: %d  precision: %f' % (total, precision),end='')
        pointer=0

        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        print('saved to',checkpoint_file)



