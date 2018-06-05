import pandas as pd
import numpy as np

#encoding:utf-8
#注意!没做nounflag.需要的去reader.py取

import numpy as np
import word2vec
import re
import time
import os
import pickle
import random
import requests
import json
from queue import Queue
#bug:shorten和shorten_front不一样的话,每一遍都得重新计算而不是直接从队列里拿出来!


class reader(object):
    def __init__(self,\
                patchlength=3,\
                maxlength=700,\
                embedding_size=100,\
                num_verbs=2,\
                allinclude=False,\
                shorten=False,\
                shorten_front=False,\
                testflag=False,\
                passnum=0,\
                dpflag=False):   #几句前文是否shorten #是否输出不带tag,只有单词的句子 

#patchlength:每次输入前文额外的句子的数量.
#maxlength:每句话的最大长度.(包括前文额外句子).超过该长度的句子会被丢弃.
#embedding_size:词向量维度数.



        self.testflag=testflag

        self.resp=pd.read_csv('train.csv').values #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
        '''
        for i in range(5):
            print(self.resp[i])
            input()
'''
        self.readlength=len(self.resp)
        print('readlength',self.readlength)
        self.pointer=random.randint(0,self.readlength-1)
#            self.pointer=101118
        self.pointer=0
        print('pointer',self.pointer)


    def list_tags(self,batch_size,test=False):
        print('pointer',self.pointer)
        self.pointer+=batch_size
#        print(self.pointer)
        if test==False:
            if self.pointer>=self.readlength*5/6:
                self.pointer=batch_size+random.randint(0,batch_size)
                print('epoch')
        else:
            if self.pointer>=self.readlength:
                self.pointer=self.readlength
 #               print('epoch')
        temp=self.resp[self.pointer-batch_size:self.pointer]

    
        answer=temp[:,0]
        temp=temp[:,1:]
        if False:
            for i in temp:
                add=(random.random()-0.5)/5+1
                for j in range(784):
                    i[j]*=add
                    if(i[j]>255):i[j]=255
                for j in range(30):
                    rd=random.randint(0,783)
                    if(i[rd]!=0):
                        i[rd]=0
                    else:
                        i[rd]=random.randint(64,255)
            #    print(i)
            #    input()

        stopper=2283
        if(self.pointer>stopper):
              k1=stopper-self.pointer+batch_size
              for k2 in range(784):
                  if(temp[k1][k2]>254):print('8',end=' ')
                  elif(temp[k1][k2]>220):print('+',end=' ')
                  elif(temp[k1][k2]>100):print('.',end=' ')
                  else:print(' ',end=' ')
                  if(k2%28==27):print('')
              print(answer[k1],self.pointer-batch_size+k1)
              input()

                
        return temp,answer




if __name__ == '__main__':
    model = reader()
    while True:
        t,p=model.list_tags(50)
