import pandas as pd
import numpy as np
df=pd.read_csv('train.csv',header=None,sep=' ') #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
print(df.values[1])
