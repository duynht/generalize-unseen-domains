"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
import numpy as np,numpy
import csv
import glob
import pandas as pd

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                        labels.shape))
        self._num_examples = images.shape[0]
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def csv_import():
    x_dic = {}
    y_dic = {}
    print("csv file importing...")
    for i in ['bathroom','bathroom2','bedrooms','bedrooms2','corridor1','corridor2_1','corridor2_2','kitchen','kitchen2','lab2']:
        x_dic[str(i)] = np.empty([0,500,90],float)
        y_dic[str(i)] = np.empty([0,1],int)
        
   # for i in ["bed", "fall", "pickup", "run", "standup", "walk"]:
#        xx = np.array([[ float(elm) for elm in v] for v in csv.reader(open("./input_files/xx_1000_60_" + str(i) + ".csv","r"))])
#        yy = np.array([[ float(elm) for elm in v] for v in csv.reader(open("./input_files/yy_1000_60_" + str(i) + ".csv","r"))])

#        xx = xx[::2,:]
#        yy = yy[::2,:]

        SKIPROW = 2 #Skip every 2 rows -> overlap 800ms to 600ms  (To avoid memory error)
        # num_lines = sum(1 for l in open("../Wifi_Activity_Recognition/input_files/xx_1000_60_" + str(i) + ".csv"))
        skip_idx =[]# [x for x in range(1, num_lines) if x % SKIPROW !=0]
        for label in ['fall','nonfall']:
            xx = np.array(pd.read_csv("../Wifi_Activity_Recognition//input_files/"+str(i)+"/xx_1000_60_"+str(label)+".csv", header=None, skiprows = skip_idx))
            yy = np.array(pd.read_csv("../Wifi_Activity_Recognition//input_files/"+str(i)+"/yy_1000_60_"+str(label)+".csv", header=None, skiprows = skip_idx))

            # eliminate the NoActivity Data
          #  rows, cols = np.where(yy>0)
          #  xx = np.delete(xx, rows[ np.where(cols==0)],0)
          #  yy = np.delete(yy, rows[ np.where(cols==0)],0)

            xx = xx.reshape(len(xx),1000,90)

            # 1000 Hz to 500 Hz (To avoid memory error)
            xx = xx[:,::2,:90]

            x_dic[str(i)] = np.concatenate((x_dic[str(i)],xx),axis = 0)
            y_dic[str(i)] = np.concatenate((y_dic[str(i)],yy),axis = 0)

        print(str(i), "finished...", "xx=", xx.shape, "yy=",  yy.shape)
    return x_dic,y_dic
   # return x_dic['bathroom'],x_dic['bathroom2'],x_dic['bedrooms'],x_dic['bedrooms2'],x_dic['corridor1'],x_dic['corridor2_1'],x_dic['corridor2_2'],x_dic['kitchen'],x_dic['kitchen2'],x_dic['lab2'], \
    #        y_dic['bathroom'],y_dic['bathroom2'],y_dic['bedrooms'],y_dic['bedrooms2'],y_dic['corridor1'],y_dic['corridor2_1'],y_dic['corridor2_2'],y_dic['kitchen'],y_dic['kitchen2'],y_dic['lab2']
