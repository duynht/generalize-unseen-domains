import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
from ConfigParser import *
import os
import cPickle
import scipy.io
import sys
import glob
from numpy.linalg import norm
from scipy import misc

import utils

from cross_vali_input_data import csv_import, DataSet
from sklearn.utils import shuffle
# for i in ['bathroom','bathroom2','bedrooms','bedrooms2','corridor1','corridor2_1','corridor2_2','kitchen','kitchen2','lab2']:
#     shuffle(x_dic[str(i)],y_dic[str(i)],random_state = 0)
#     x_path = 'falldefi'+str(i)+'_images.pkl'
#     y_path = 'falldefi'+str(i)+'_labels.pkl'
#     pickle_out = open(x_path,"wb")
#     cPickle.dump(np.array(x_dic[str(i)]),pickle_out)
#     pickle_out.close()
#     pickle_out = open(y_path,"wb")
#     cPickle.dump(np.array(y_dic[str(i)]),pickle_out)
#     pickle_out.close()

dic = {'bathroom':0,'bathroom2':1,'bedrooms':2,'bedrooms2':3,'corridor1':4,'corridor2_1':5,'corridor2_2':6,'kitchen':7,'kitchen2':8,'lab2':9}

for i in ['bathroom','bathroom2','bedrooms','bedrooms2','corridor1','corridor2_1','corridor2_2','kitchen','kitchen2','lab2']:
    x_path = 'falldefi'+str(i)+'_images.pkl'
    y_path = 'falldefi'+str(i)+'_labels.pkl'
    pickle_in = open(x_path,"rb")
    xx = cPickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open(y_path,"rb")
    yy = cPickle.load(pickle_in)
    pickle_in.close()
    xx,yy = shuffle(xx,yy,random_state=0)
    pickle_out = open('./falldefi_shuffled/'+x_path,'wb')
    cPickle.dump(xx,pickle_out)
    pickle_out.close()
    pickle_out = open('./falldefi_shuffled/'+y_path,'wb')
    cPickle.dump(yy,pickle_out)
    pickle_out.close()
    print(str(i),yy)
    print(str(i)+' ',xx.shape,yy.shape)
