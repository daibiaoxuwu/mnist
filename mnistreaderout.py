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
    def printtag(self,number):
#        return [k for k, v in self.verbtags.items() if v == number][0]
        return self.verbtags[number]

    def work(self,a,posdict):
        url = 'http://166.111.139.15:9000'
        params = {'properties' : r"{'annotators': 'tokenize,ssplit,pos,depparse', 'outputFormat': 'json'}"}
        while True:
            try:
                #print('0')
                resp = requests.post(url,a,params=params).text
                content=json.loads(resp)
                #print('1')
                break
            except Exception as e:
                if e!=KeyboardInterrupt:
                    print('error...')
                    return -2
                else:
                    raise KeyboardInterrupt
        for sentence in content['sentences']:
            for i in sentence['enhancedPlusPlusDependencies']:
                    #print('out',i)
#                for j in i:
                    if i['dep']=='nsubjpass':
                        #print(i)
                        #input()
#                        print(posdict)
 #                           print(':\n',a,posdict[i['governor']]+1)
                        if i['governor'] in posdict:
                            return posdict[i['governor']]+1
                        else:
                            return -i['governor']
        return 0

    def isverb(self,verb):
        if verb not in self.ldict: return False
        for i in self.verbtags:
            if (self.ldict[verb]+'('+i) not in self.cldict:
                if verb in self.verbset:
                    print('verb in set and ldict but not in verbtags',verb)
                    return True
                else:
                    return False
        if verb not in self.verbset:
            print('verb not in set but in ldict and verbtags',verb)
            return False
        return True
        

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



        self.url = 'http://166.111.139.15:9000'
        self.shorten=shorten
        self.shorten_front=shorten_front   #几句前文是否shorten #是否输出不带tag,只有单词的句子 
        self.patchlength=patchlength
        self.maxlength=maxlength
        self.embedding_size=embedding_size
        self.num_verbs=num_verbs
        self.allinclude=allinclude
        self.passnum=passnum
        self.dpflag=dpflag
        print('pas',passnum)
        self.verbtags=['VB','VBZ','VBP','VBD','VBN','VBG'] #所有动词的tag
#        self.model=word2vec.load('tense/combine100.bin')   #加载词向量模型
        print('loaded model')
        self.oldqueue=Queue()
        self.testflag=testflag

        if testflag==False:
            self.resp=pd.read_csv('test.csv').values #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
            '''
            for i in range(5):
                print(self.resp[i])
                input()
'''
            self.readlength=len(self.resp)
            print('readlength',self.readlength)
            self.pointer=0
#            self.pointer=101118
            print('pointer',self.pointer)

            '''
        else:
            self.testfile=open('input.txt')
            for _ in range(self.patchlength):
                if shorten_front==True:
                    #self.oldqueue.put(input('0:type sentence:'))
                    self.oldqueue.put(self.testfile.readline())
                else:
                    #self.oldqueue.put(self.parse(input('0:type sentence:')))
                    self.oldqueue.put(self.parse(self.testfile.readline()))
#加载文字

#加载原型词典(把动词变为它的原型)
        with open('tense/ldict2', 'rb') as f:
            self.ldict = pickle.load(f)
        with open('tense/tagdict', 'rb') as f:
            self.tagdict = pickle.load(f)
        with open('tense/cldict', 'rb') as f:
            self.cldict = pickle.load(f)
        with open('tense/verbset', 'rb') as f:
            self.verbset = pickle.load(f)
    def lemma(self,verb):
        if verb in self.ldict:
            return self.ldict[verb]
        else:
            params = {'properties' : r"{'annotators': 'lemma', 'outputFormat': 'json'}"}
            resp = requests.post(self.url, verb, params=params).text
            content=json.loads(resp)
            word=content['sentences'][0]['tokens'][0]['lemma']
            self.ldict[verb]=word
            print('errorverb: ',verb,word)
            return word
    def readmodel(self,word):
        if word in self.model:
            return self.model[word].tolist()
        else:
            return [0]*self.embedding_size
        '''     

    def list_tags(self,batch_size,test=True):
        self.pointer+=batch_size
        if test==False:
            if self.pointer>=self.readlength*5/6:
                self.pointer=batch_size+random.randint(0,batch_size)
                print('epoch')
        else:
            if self.pointer>self.readlength:
                self.pointer=batch_size+random.randint(0,batch_size)
                print('epoch')
        temp=self.resp[self.pointer-batch_size:self.pointer]

    
        answer=[0]*batch_size
        return temp,answer




if __name__ == '__main__':
    model = reader()
    for i in range(20):
        t,p=model.list_tags(2)
