import pandas as pd
import numpy as np
import random


class reader(object):
    def __init__():
        self.resp=pd.read_csv('test.csv').values #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
        self.readlength=len(self.resp)
        print('readlength',self.readlength)
        self.pointer=0

    def list_tags(self,batch_size):
        self.pointer+=batch_size
        if self.pointer>self.readlength:
            raise Exception
        temp=self.resp[self.pointer-batch_size:self.pointer]
        answer=[0]*batch_size
        return temp,answer
