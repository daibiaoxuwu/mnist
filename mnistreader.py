#encoding:utf-8
import pandas as pd
import numpy as np
import time
import os
import random

class reader(object):
    def __init__(self)

        self.testflag=testflag

        readin=pd.read_csv('train.csv').values #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
        '''
        for i in range(5):
            print(self.resp[i])
            input()
'''
        lst = [12817,  # 0's
       60, 191, 2284, 2316, 5275, 7389, 19633, 19979, 24891, 29296, 32565, 38191, 38544, 40339, 41739,  # 1's
       4677, 7527, 9162, 13471, 16598, 20891, 27364,  # 2's
       240, 11593, 11896, 17966, 25708, 28560, 33198, 34477, 36018, 41492,  # 3's
       1383, 6781, 22478, 23604, 26171, 26182, 26411, 18593, 34862, 36051, 36241, 36830, 37544,  # 4's
       456, 2867, 2872, 5695, 6697, 9195, 18319, 19364, 27034, 29253, 35620,  # 5's
       7610, 12388, 12560, 14659, 15219, 18283, 24122, 31649, 40214, 40358, 40653,  # 6's
       6295, 7396, 15284, 19880, 20089, 21423, 25233, 26366, 26932, 27422, 31741,  # 7's
       8566, 10920, 23489, 25069, 28003, 28851, 30352, 30362, 35396, 36984, 39990, 40675, 40868, 41229,  # 8's
       631, 4226, 9943, 14914, 15065, 17300, 18316, 19399, 20003, 20018, 23135, 23732, 29524, 33641, 40881, 41354  # 9's
       ]
        self.resp = np.delete(readin, lst, 0)
        self.readlength=len(self.resp)
        print('readlength',self.readlength)
        self.pointer=random.randint(0,self.readlength-1)
#            self.pointer=101118
        self.pointer=0
        print('pointer',self.pointer)


    def list_tags(self,batch_size,test=False):
#        print('pointer',self.pointer)
        self.pointer+=batch_size
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
        if test==False:
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
        return temp,answer



if __name__ == '__main__':
    model = reader()
    while True:
        t,p=model.list_tags(50)
