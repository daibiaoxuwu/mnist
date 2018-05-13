#encoding:utf-8
#rnnmodel3.py 针对两个动词
#learning rate decay
#patchlength 0 readfrom resp
#add:saving session
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

PIXELNUM=784

class rnnmodel(object):
    def __init__(self,\
                vocab_single=6,\
                maxlength=200,\
                embedding_size=100,\
                initial_training_rate=0.001,\
                batch_size=1,\
                num_verbs=2):

#针对多个动词:
        vocab_size=pow(vocab_single,num_verbs)
#learning_rate 可依次递减.然而global_step好像只能在run里每train一次+1,而不能写在这个init函数里?不太清楚.
#每500步变为原来的0.8倍,指数递减.
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(initial_training_rate, global_step=self.global_step, decay_steps=500,decay_rate=0.8)

#输入信息.
#x:输入的数据.batch_size是每次输入的数据的多少.大致是越多越精确,不过越多数据用得越快,需要多循环用几次.而且越多吃的显存越多,太大会报错.
#也有可能设置得太大导致无法学习的情况.
#maxlength为输入的序列最大长度,也就是一个句子最多有几个词.
#embedding_size是表示每个词的词向量的维度数.
        self.x = tf.placeholder("float", [batch_size, PIXELNUM ])

#输入信息对应的答案.
#vocab_size是输出数据的维度数.
#这是一个单分类问题,输出为预测的6种时态各自的可能性.
        self.y = tf.placeholder("float", [batch_size, 10])


        layer1size=128
        layer2size=32
# RNN output node weights and biases
#这两个是需要被训练的变量.除此之外,lstm状态也是需要保存的变量.
        weights1 = tf.Variable(tf.random_normal([PIXELNUM, layer1size])) 
        biases1 =  tf.Variable(tf.random_normal([layer1size])) 
        layer1 = tf.nn.relu(tf.matmul(self.x,  weights1) +  biases1)

        weights2 = tf.Variable(tf.random_normal([layer1size, layer2size])) 
        biases2 =  tf.Variable(tf.random_normal([layer2size])) 
        layer2 = tf.nn.relu(tf.matmul( layer1, weights2) +  biases2)

        weights3 = tf.Variable(tf.random_normal([layer2size, 10])) 
        biases3 =  tf.Variable(tf.random_normal([10])) 
        self.pred= tf.matmul(layer2, weights3) + biases3


# Loss函数:
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
# 优化器:
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
# 预测正确率:(这个与训练无关,纯粹是监控使用)
        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

if __name__ == '__main__':
	model = rnnmodel()
