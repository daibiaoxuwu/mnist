import pandas as pd
import numpy as np
import random


class reader(object):
    def __init__(self):
        self.resp=pd.read_csv('train.csv').values #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
        self.readlength=len(self.resp)
        print('readlength',self.readlength)
        self.pointer=0

    def list_tags(self,batch_size,test=False):
        self.pointer+=batch_size
        if test:
            if self.pointer>=self.readlength:
                self.pointer=batch_size+random.randint(0,batch_size)
                print('epoch')
        else:
            if self.pointer>=self.readlength*5/6:
                self.pointer=batch_size+random.randint(0,batch_size)
                print('epoch')

        temp=self.resp[self.pointer-batch_size:self.pointer]
        answer=temp[:,0]
        temp=temp[:,1:]
        return temp,answer


