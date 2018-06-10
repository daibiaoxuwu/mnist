import pandas as pd
import numpy as np
import tensorflow as tf
log_dir='ckpt/'

batch_size=50
def cnn(input_x, input_y):
    with tf.name_scope('reshape'):
        layer0 = tf.reshape(input_x, [-1, 28, 28, 1])
  
    with tf.name_scope('conv_1'):
        weight = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        bias = tf.Variable(tf.truncated_normal([32], stddev=0.1))
        conv = tf.nn.conv2d(layer0, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
        layer1 = tf.nn.relu(conv)
  
    with tf.name_scope('pool_1'):
        layer2 = tf.nn.max_pool(layer1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
    with tf.name_scope('conv_2'):
        weight = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        bias = tf.Variable(tf.truncated_normal([64], stddev=0.1))
        conv = tf.nn.conv2d(layer2, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
        layer3 = tf.nn.relu(conv)
  
    with tf.name_scope('pool_2'):
        layer4 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
    with tf.name_scope('conv_3'):
        weight = tf.Variable(tf.truncated_normal([3, 3, 64, 10], stddev=0.1))
        bias = tf.Variable(tf.truncated_normal([10], stddev=0.1))
        conv = tf.nn.conv2d(layer4, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
        layer5 = tf.nn.relu(conv)
  
    with tf.name_scope('pool_3'):
        layer6 = tf.nn.max_pool(layer5, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')
  
    with tf.name_scope('flatten'):
        layer7 = tf.reshape(layer6, [-1, 10])
  
#without normalize, losses cross
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy( labels=input_y, logits=layer7))
    '''
#normalize, nn cross
      layer_out_norm=tf.nn.l2_normalize(layer_out,[1])
      loss = tf.nn.sparse_softmax_loss_with_logits(
          labels=input_y, logits=layer_out_norm)

#without normalize, nn cross
      loss_fake = tf.nn.sparse_softmax_loss_with_logits(
          labels=input_y, logits=layer_out)

#without logits, losses cross
      loss_fake2 = tf.losses.sparse_softmax_loss(
          labels=input_y, logits=layer_out)
    loss_mean = tf.reduce_mean(loss)
      '''

#optimizer
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

#accuracy
    isequal = tf.equal(tf.argmax(layer7, 1), input_y)
    accuracy = tf.reduce_mean(tf.cast(isequal, tf.int32))
    return layer7, loss, optimizer, accuracy



if __name__ == '__main__':
#input train data
    data=pd.read_csv('train.csv').values 

#input placeholder
#inputs shape[batch_size,784]
    input_x = tf.placeholder(tf.float32, [None, 784])
#answers shape[batch_size]
    input_y = tf.placeholder(tf.int64, [None])
  
#cnn prediction: shape: [batch_size, 10]
    layer_out, loss, optimizer, accuracy = cnn(input_x, input_y)


#saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

#load checkpoint
        '''
        model_file=tf.train.latest_checkpoint(log_dir)
        print('loaded:' , model_file)
        saver.restore(sess,model_file)
        '''
        maxacc=0
        minloss=1000
        for i in range(20):

            for j in range(0 , 35000, batch_size):
#generate data
                databatch=data[j : j + batch_size]
                inputs=databatch[:,1:]
                answers=databatch[:,0]
#train
                optimizer.run(feed_dict={input_x: inputs, input_y: answers })

                if j % 7000 == 0:
                    acc = accuracy.eval(feed_dict={ input_x: inputs, input_y: answers })
                    print('epoch %d, pos %d, trainacc %g' % (i, j, acc))

            totacc=0
            for j in range(35000, 42000, batch_size):
                databatch=data[j : j + batch_size]
                totacc += accuracy.eval(feed_dict={ input_x: databatch[:,1:], input_y: databatch[:,0]})
            print('epoch %d, testacc %g' % (i, (totacc * batch_size) // 7000))
            print('saved to',saver.save(sess,log_dir+'model.ckpt',global_step=i))
            '''
            optimizer.run(feed_dict={input_x: inputs, input_y: answers })
      #      crossloss,_=sess.run([loss,optimizer],feed_dict={input_x: inputs, input_y: answers })
            if i % 10 == 0:
                train_accuracy, lossop = sess.run([accuracy,loss_mean],feed_dict={input_x: inputs, input_y: answers })
            print('step %d, training accuracy %g loss: %g' % (i, train_accuracy, lossop))
            train_accuracy, lossop, yc, ycn, ce, cor, cfk, cfk2 = sess.run([accuracy,loss_mean, layer_out, layer_out_norm,loss, correct_prediction,loss_fake, loss_fake2],feed_dict={
                input_x: inputs, input_y: answers })
            print('step %d, training accuracy %g loss: %g' % (i, train_accuracy, lossop))
            print('yconv:',yc[0],len(yc))
            print('yconvnorm:',ycn[0],len(ycn))
            print('yans:',answers[0])
            print('cross:',ce[0])
            print('correct:',cor[0])
            print('convfake:',cfk[0])
            print('convfake2:',cfk2)
            '''
                
        with open('submission6.csv','w') as f:
            f.write('ImageId,Label\n')
            data=pd.read_csv('train.csv').values 

            for step in range(0, 28000, batch_size):
#generate data
                databatch=data[j : j + batch_size]
                feed_dict = { input_x: databatch[:,1:], input_y: databatch[:,0]}
                anst=sess.run([layer_out], feed_dict=feed_dict)[0]
                for i in range(len(anst)):
                    f.write(str(datapointer-batch_size+i+1)+','+str(np.argmax(anst[i]))+'\n')
