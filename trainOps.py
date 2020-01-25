import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
from configparser import *
import os
import pickle
import scipy.io
import sys
import glob
from numpy.linalg import norm
from scipy import misc

import utils
from sklearn.model_selection import train_test_split
from cross_vali_input_data import csv_import, DataSet
from sklearn.utils import shuffle
import gc
from model import Model
class TrainOps(object):

    def __init__(self, model, exp_dir):

	self.model = model
	self.exp_dir = exp_dir

	self.config = tf.ConfigProto()
	self.config.gpu_options.allow_growth=True

	self.data_dir = './data/'
		    	
    def load_exp_config(self):

	config = ConfigParser()
	config.read(self.exp_dir + '/exp_configuration')

	self.source_dataset = config.get('EXPERIMENT_SETTINGS', 'source_dataset')
	self.target_dataset = config.get('EXPERIMENT_SETTINGS', 'target_dataset')
	self.no_images = config.getint('EXPERIMENT_SETTINGS', 'no_images')

	self.log_dir = os.path.join(self.exp_dir,'logs')
	self.model_save_path = os.path.join(self.exp_dir,'model')

	if not os.path.exists(self.log_dir):
	    os.makedirs(self.log_dir)

	if not os.path.exists(self.model_save_path):
	    os.makedirs(self.model_save_path)

	self.train_iters = config.getint('MAIN_SETTINGS', 'train_iters')
	self.k = config.getint('MAIN_SETTINGS', 'k')	
	self.batch_size = config.getint('MAIN_SETTINGS', 'batch_size')
	self.model.batch_size = self.batch_size
	self.model.gamma = config.getfloat('MAIN_SETTINGS', 'gamma')
	self.model.learning_rate_min = config.getfloat('MAIN_SETTINGS', 'learning_rate_min')
	self.model.learning_rate_max = config.getfloat('MAIN_SETTINGS', 'learning_rate_max')
	self.T_adv = config.getint('MAIN_SETTINGS', 'T_adv')
	self.T_min = config.getint('MAIN_SETTINGS', 'T_min')
	
    def load_svhn(self, split='train'):

	print ('Loading SVHN dataset.')

	image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'

	image_dir = os.path.join(self.data_dir, 'svhn', image_file)
	svhn = scipy.io.loadmat(image_dir)
	images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 255.
	labels = svhn['y'].reshape(-1)
	labels[np.where(labels==10)] = 0
	return images, labels

    def load_mnist(self, split='train'):

	print ('Loading MNIST dataset.')
	image_file = 'train.pkl' if split=='train' else 'test.pkl'
	image_dir = os.path.join(self.data_dir, 'mnist', image_file)
	with open(image_dir, 'rb') as f:
	    mnist = pickle.load(f)
	images = mnist['X'] 
	labels = mnist['y']

	images = images / 255.

	images = np.stack((images,images,images), axis=3) # grayscale to rgb

	return np.squeeze(images[:self.no_images]), labels[:self.no_images]

    def load_test_data(self, target):

	if target=='svhn':
	    self.target_test_images, self.target_test_labels = self.load_svhn(split='test')
	elif target=='mnist':
	    self.target_test_images, self.target_test_labels = self.load_mnist(split='test')

	return self.target_test_images,self.target_test_labels

    def train(self): 


	print('Loading data.')
        
       # x_dic, y_dic = csv_import()

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
        x = []
        y = []
        for i in ['bathroom','bathroom2','bedrooms','bedrooms2','corridor1','corridor2_1','corridor2_2','kitchen','kitchen2']:
            x_path = './falldefi_shuffled/falldefi'+str(i)+'_images.pkl'
            y_path = './falldefi_shuffled/falldefi'+str(i)+'_labels.pkl'
            pickle_in = open(x_path,"rb")
            xx = pickle.load(pickle_in)
            xx = xx[int(len(xx)/3):]
            x.append(xx)
            pickle_in.close()
            pickle_in = open(y_path,"rb")
            yy = pickle.load(pickle_in)
            yy = yy[int(len(yy)/3):]
            pickle_in.close()
            y.append(yy)
            gc.collect()
            #print(str(i),yy)
            print((str(i)+' finish',xx.shape,yy.shape))
        print((np.array(x).shape, np.array(y).shape))
        
        
        #target_test
      #  target_test_images = np.array(x[dic['lab2']])
      #  target_test_labels = np.array(y[dic['lab2']])
      #  
      #  pickle_out = open('target_images.pkl','wb')
      #  cPickle.dump(target_test_images,pickle_out)
      #  pickle_out.close()
      #  pickle_out = open('target_labels.pkl','wb')
      #  cPickle.dump(target_test_labels,pickle_out)
      #  pickle_out.close()
        source_images = pickle.load(open('target_images.pkl','rb'))
        source_labels = pickle.load(open('target_labels.pkl','rb'))

        #source
        target_test_images = np.r_[x[dic['bathroom']],x[dic['bathroom2']],x[dic['bedrooms']],x[dic['bedrooms2']],x[dic['corridor1']],x[dic['corridor2_1']],x[dic['corridor2_2']], x[dic['kitchen']],x[dic['kitchen2']]]

        target_test_labels = np.r_[y[dic['bathroom']],y[dic['bathroom2']],y[dic['bedrooms']],y[dic['bedrooms2']],y[dic['corridor1']],y[dic['corridor2_1']],y[dic['corridor2_2']], y[dic['kitchen']],y[dic['kitchen2']]]
        
        target_test_images, target_test_labels = shuffle(target_test_images, target_test_labels, random_state=0)

      #  pickle_out = open('source_images.pkl','wb')
      #  cPickle.dump(source_images,pickle_out)
      #  pickle_out.close()
      #  pickle_out = open('source_labels.pkl','wb')
      #  cPickle.dump(source_labels,pickle_out)
      #  pickle_out.close()

        source_train_images, source_test_images, source_train_labels, source_test_labels = train_test_split(source_images, source_labels, test_size = 0.33, random_state=0)

        #source test
     #   source_test_images = np.r_[x[dic['bathroom']][:int(len(x[dic['bathroom']])/10)*4],x[dic['bathroom2']][:int(len(x[dic['bathroom2']])/10)*4],x[dic['bedrooms']][:int(len(x[dic['bedrooms']])/10)*4],x[dic['bedrooms2']][:int(len(x[dic['bedrooms2']])/10)*4],x[dic['corridor1']][:int(len(x[dic['corridor1']])/10)*4],x[dic['corridor2_1']][:int(len(x[dic['corridor2_1']])/10)*4],x[dic['corridor2_2']][:int(len(x[dic['corridor2_2']])/10)*4],x[dic['kitchen']][:int(len(x[dic['kitchen']])/10)*4],x[dic['kitchen2']][:int(len(x[dic['kitchen2']])/10)*4]]

     #   source_test_labels = np.r_[y[dic['bathroom']][:int(len(y[dic['bathroom']])/10)*4],y[dic['bathroom2']][:int(len(y[dic['bathroom2']])/10)*4],y[dic['bedrooms']][:int(len(x[dic['bedrooms']])/10)*4],y[dic['bedrooms2']][:int(len(y[dic['bedrooms2']])/10)*4],y[dic['corridor1']][:int(len(y[dic['corridor1']])/10)*4],y[dic['corridor2_1']][:int(len(y[dic['corridor2_1']])/10)*4],y[dic['corridor2_2']][:int(len(y[dic['corridor2_2']])/10)*4],y[dic['kitchen']][:int(len(y[dic['kitchen']])/10)*4],y[dic['kitchen2']][:int(len(y[dic['kitchen2']])/10)*4]
     #   
     #   #source train
     #   source_train_images = np.r_[x[dic['bathroom']][int(len(x[dic['bathroom']])/10)*4:],x[dic['bathroom2']][int(len(x[dic['bathroom2']])/10)*4:],x[dic['bedrooms']][int(len(x[dic['bedrooms']])/10)*4:],x[dic['bedrooms2']][int(len(x[dic['bedrooms2']])/10)*4:],x[dic['corridor1']][int(len(x[dic['corridor1']])/10)*4:],x[dic['corridor2_1']][int(len(x[dic['corridor2_1']])/10)*4:],x[dic['corridor2_2']][int(len(x[dic['corridor2_2']])/10)*4:],x[dic['kitchen']][int(len(x[dic['kitchen']])/10)*4:],x[dic['kitchen2']][int(len(x[dic['kitchen2']])/10)*4:]

     #   source_train_labels = np.r_[y[dic['bathroom']][int(len(y[dic['bathroom']])/10)*4:],y[dic['bathroom2']][int(len(y[dic['bathroom2']])/10)*4:],y[dic['bedrooms']][int(len(y[dic['bedrooms']])/10)*4:],y[dic['bedrooms2']][int(len(y[dic['bedrooms2']])/10)*4:],y[dic['lab2']][int(len(y[dic['lab2']])/10)*4:]]
     #   
     #   #target test
     #   target_test_images = np.r_[x[dic['corridor1']][int(len(x[dic['corridor1']])/2):],x[dic['corridor2_1']][int(len(x[dic['corridor2_1']])/2):],x[dic['corridor2_2']][int(len(x[dic['corridor2_2']])/2):],x[dic['kitchen']][int(len(x[dic['kitchen']])/2):],x[dic['kitchen2']][int(len(x[dic['kitchen2']])/2):]]

     #   target_test_labels = np.r_[y[dic['corridor1']][int(len(y[dic['corridor1']])/2):],y[dic['corridor2_1']][int(len(y[dic['corridor2_1']])/2):],y[dic['corridor2_2']][int(len(y[dic['corridor2_2']])/2):],y[dic['kitchen']][int(len(y[dic['kitchen']])/2):],y[dic['kitchen2']][int(len(y[dic['kitchen2']])/2):]]
       
        
        #expand dims
        source_train_images = np.expand_dims(source_train_images,axis=-1)
        source_test_images = np.expand_dims(source_test_images,axis=-1)
        target_test_images = np.expand_dims(target_test_images,axis=-1)
        
        #squeeze dims
        source_train_labels = np.squeeze(source_train_labels)
        source_test_labels = np.squeeze(source_test_labels)
        target_test_labels = np.squeeze(target_test_labels)


        print((source_train_images.shape,source_train_labels.shape))
        print((source_test_images.shape,source_test_labels.shape))
        print((target_test_images.shape,target_test_labels.shape))

      #  x_bed, x_fall, x_pickup, x_run, x_standup, x_walk, \
      #  y_bed, y_fall, y_pickup, y_run, y_standup, y_walk = csv_import()
      #  #data shuffle
      #  x_bed, y_bed = shuffle(x_bed, y_bed, random_state = 0)
      #  x_fall, y_fall = shuffle(x_fall, y_fall, random_state = 0)
      #  x_pickup, y_pickup = shuffle(x_pickup, y_pickup, random_state = 0)
      #  x_run, y_run = shuffle(x_run, y_run, random_state = 0)
      #  x_standup, y_standup = shuffle(x_standup, y_standup, random_state = 0)
      #  x_walk, y_walk = shuffle(x_walk, y_walk, random_state = 0)

        #data seperation
      #  source_train_images = np.r_[x_bed[int(len(x_bed) / 10):], x_fall[int(len(x_fall) / 10):],x_pickup[int(len(x_pickup) / 10):], x_run[int(len(x_run) / 10):], x_standup[int(len(x_standup) / 10):], x_walk[int(len(x_walk) / 10):]]
      #  source_train_labels = np.r_[y_bed[int(len(y_bed) / 10):], y_fall[int(len(y_fall) / 10):],y_pickup[int(len(y_pickup) / 10):], y_run[int(len(y_run) / 10):], y_standup[int(len(y_standup) / 10):], y_walk[int(len(y_walk) / 10):]]
      #  
      #  pickle_out = open("ermon_wifi_train_last9over10_x.pkl","wb")
      #  cPickle.dump(source_train_images,pickle_out)
      #  pickle_out.close()
      #  pickle_out = open("ermon_wifi_train_last9over10_y.pkl","wb")
      #  cPickle.dump(source_train_labels,pickle_out)
      #  pickle_out.close()
        
        # load pickled ermon data
      #  pickle_in = open("ermon_wifi_train_last9over10_x.pkl","rb")
      #  source_train_images = cPickle.load(pickle_in)
      #  pickle_in.close()
      #  pickle_in = open("ermon_wifi_train_last9over10_y.pkl","rb")
      #  source_train_labels = cPickle.load(pickle_in)
      #  pickle_in.close()
      #  #discard the no activity class
      #  source_train_labels = source_train_labels[:,1:]
      #  source_train_labels = np.argmax(source_train_labels,axis=1)
      #  source_train_images = np.expand_dims(source_train_images, axis = -1)
        
        #source_test_images = np.r_[x_bed[:int(len(x_bed) / 10)], x_fall[:int(len(x_fall) / 10)],x_pickup[:int(len(x_pickup) / 10)], x_run[:int(len(x_run) / 10)], x_standup[:int(len(x_standup) / 10)], x_walk[:int(len(x_walk) / 10)]]
        #source_test_labels = np.r_[y_bed[:int(len(y_bed) / 10)], y_fall[:int(len(y_fall) / 10)],y_pickup[:int(len(y_pickup) / 10)], y_run[:int(len(y_run) / 10)], y_standup[:int(len(y_standup) / 10)], y_walk[:int(len(y_walk) / 10)]]

      #  pickle_out = open("ermon_wifi_test_first1over10_x.pkl","wb")
      #  cPickle.dump(source_test_images,pickle_out)
      #  pickle_out.close()
      #  pickle_out = open("ermon_wifi_test_first1over10_y.pkl","wb")
      #  cPickle.dump(source_test_labels,pickle_out)
      #  pickle_out.close()
        
        #load the pickled ermon data
      #  pickle_in = open("ermon_wifi_test_first1over10_x.pkl","rb")
      #  source_test_images = cPickle.load(pickle_in)
      #  pickle_in.close()
      #  pickle_in = open("ermon_wifi_test_first1over10_y.pkl","rb")
      #  source_test_labels = cPickle.load(pickle_in)
      #  pickle_in.close()
      #  #discard the no activity class
      #  source_test_labels = source_test_labels[:,1:]
      #  source_test_labels = np.argmax(source_test_labels,axis=1)
      #  source_test_images = np.expand_dims(source_test_images, axis = -1) 

      #  target_test_images = source_test_images[:int(len(source_test_images)/2)]
      #  target_test_labels = source_test_labels[:int(len(source_test_labels)/2)]
      #  source_test_images = source_test_images[int(len(source_test_images)/2):]
      #  source_test_labels = source_test_labels[int(len(source_test_labels)/2):]
	
        #source_train_images, source_train_labels = self.load_mnist(split='train')
	#source_test_images, source_test_labels = self.load_mnist(split='test')
	#target_test_images, target_test_labels = self.load_test_data(target=self.target_dataset)
	
        
        print('Loaded')

        gamma_ensemble = [0.000001]	
        for gamma_id,gamma in enumerate(gamma_ensemble):
            if gamma in [1,0.001,0.0001,0.00001]:
                continue
            self.model = Model()
            self.load_exp_config()
            self.model_save_path = os.path.join(self.exp_dir,'model_'+str(gamma_id))
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            self.model.gamma = gamma          
            # build a graph
            print(('Building model'+str(gamma_id)))
            self.model.mode='train_encoder'
            self.model.build_model()
            print('Built')
            print(('Training model #'+str(gamma_id)))
            with tf.Session(config=self.config) as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()

                summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

                counter_k = 0
                                
                print('Training')
                self.no_images = source_train_images.shape[0]
                print(('self.no_images = ', self.no_images))
                for t in range(self.train_iters):
                    if ((t+1) % self.T_min == 0) and (counter_k < self.k): #if T_min iterations are passed
                        print('Generating adversarial images [iter %d]'%(counter_k))
                        for start, end in zip(list(range(0, self.no_images, self.batch_size)), list(range(self.batch_size, self.no_images, self.batch_size))): 
                            feed_dict = {self.model.z: source_train_images[start:end], self.model.labels: source_train_labels[start:end]} 

                            #assigning the current batch of images to the variable to learn z_hat
                            sess.run(self.model.z_hat_assign_op, feed_dict) 
                            for n in range(self.T_adv): #running T_adv gradient ascent steps
                                sess.run(self.model.max_train_op, feed_dict)
                                
                            #tmp variable with the learned images
                            learnt_imgs_tmp = sess.run(self.model.z_hat, feed_dict)

                            #stacking the learned images and corresponding labels to the original dataset
                            source_train_images = np.vstack((source_train_images, learnt_imgs_tmp))
                            source_train_labels = np.hstack((source_train_labels, source_train_labels[start:end]))
                        
                        #shuffling the dataset
                        rnd_indices = list(range(len(source_train_images)))
                        npr.shuffle(rnd_indices)
                        source_train_images = source_train_images[rnd_indices]
                        source_train_labels = source_train_labels[rnd_indices]

                        counter_k+=1
                        
                    i = t % int(source_train_images.shape[0] / self.batch_size)

                    #current batch of images and labels
                    batch_z = source_train_images[i*self.batch_size:(i+1)*self.batch_size]
                    batch_labels = source_train_labels[i*self.batch_size:(i+1)*self.batch_size]

                    feed_dict = {self.model.z: batch_z, self.model.labels: batch_labels} 

                    #running a step of gradient descent
                    sess.run([self.model.min_train_op, self.model.min_loss], feed_dict) 

                    #evaluating the model
                    if t % 250 == 0:

                        summary, min_l, max_l, acc = sess.run([self.model.summary_op, self.model.min_loss, self.model.max_loss, self.model.accuracy], feed_dict)

                        train_rand_idxs = np.random.permutation(source_train_images.shape[0])[:100]
                        test_rand_idxs = np.random.permutation(target_test_images.shape[0])[:100]

                        train_acc, train_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
                                               feed_dict={self.model.z: source_train_images[train_rand_idxs], 
                                                          self.model.labels: source_train_labels[train_rand_idxs]})
                        test_acc, test_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
                                               feed_dict={self.model.z: target_test_images[test_rand_idxs], 
                                                          self.model.labels: target_test_labels[test_rand_idxs]})
                                                                                                          
                        summary_writer.add_summary(summary, t)
                        print(('Step: [%d/%d] train_min_loss: [%.4f] train_acc: [%.4f] test_min_loss: [%.4f] test_acc: [%.4f]'%(t+1, self.train_iters, train_min_loss, train_acc, test_min_loss, test_acc)))

                print('Saving')
                saver.save(sess, os.path.join(self.model_save_path, 'encoder'))
                tf.global_variables_initializer().run()

    def test(self, target):

	#test_images, test_labels = self.load_test_data(target=self.target_dataset)

      #  pickle_in = open("ermon_wifi_test_first1over10_x.pkl","rb")
      #  test_images = cPickle.load(pickle_in)
      #  pickle_in.close()
      #  pickle_in = open("ermon_wifi_test_first1over10_y.pkl","rb")
      #  test_labels = cPickle.load(pickle_in)
      #  pickle_in.close()
      #  #discard the no activity class
      #  test_labels = test_labels[:,1:]
      #  test_labels = np.argmax(test_labels,axis=1)
      #  test_images = np.expand_dims(test_images, axis = -1) 
        
      #  test_images = cPickle.load(open('target_images.pkl','rb'))
      #  test_labels = cPickle.load(open('target_labels.pkl','rb'))

        test_images = np.empty([0,500,90],float)
        test_labels = np.empty([0,1],int)
        for i in ['bathroom','bathroom2','bedrooms','bedrooms2','corridor1','corridor2_1','corridor2_2','kitchen','kitchen2']:
            x_path = './falldefi_shuffled/falldefi'+str(i)+'_images.pkl'
            y_path = './falldefi_shuffled/falldefi'+str(i)+'_labels.pkl'
            pickle_in = open(x_path,"rb")
            xx = pickle.load(pickle_in)
            xx = xx[int(len(xx)/3):]
            test_images = np.concatenate((test_images,xx),axis=0)
            pickle_in.close()
            pickle_in = open(y_path,"rb")
            yy = pickle.load(pickle_in)
            yy = yy[int(len(yy)/3):]
            pickle_in.close()
            test_labels = np.concatenate((test_labels,yy),axis=0)
            #gc.collect()
            #print(str(i),yy)
            print((str(i)+' finish'))
        print((test_images.shape,test_labels.shape))
        
        #expand dims
        test_images = np.expand_dims(test_images,axis=-1)
        
        #squeeze dims
        test_labels = np.squeeze(test_labels)
        
        print(('Test shape = ',test_images.shape))

        # build a graph
	print('Building model')
	self.model.mode='train_encoder'
	self.model.build_model()
	print('Built')

      #  gamma_ensemble = [1,0.1,0.01,0.001,0.0001,0.00001,0.000001]	
      #  for gamma_id,gamma in enumerate(gamma_ensemble):
      #      
      #      self.model_save_path = os.path.join(self.exp_dir,'model'+str(gamma_id))
      #      if not os.path.exists(self.model_save_path):
      #          os.makedirs(self.model_save_path)
      #      self.model.gamma = gamma          
      #      # build a graph
      #      print ('Building model'+str(gamma_id))
      #      self.model.mode='train_encoder'
      #      self.model.build_model()
      #      print 'Built'
      #      print ('Training model #'+str(gamma_id))
	with tf.Session() as sess:

	    tf.global_variables_initializer().run()

	    print ('Loading pre-trained model.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
            gamma_ensemble = [1,0.1,0.01,0.001,0.0001,0.00001,0.000001]	
            for gamma_id,gamma in enumerate(gamma_ensemble):
                print(('Model #'+str(gamma_id))) 
                self.model_save_path = os.path.join(self.exp_dir,'model_'+str(gamma_id))
                restorer.restore(sess, os.path.join(self.model_save_path,'encoder'))

                N = 100 #set accordingly to GPU memory
                target_accuracy = 0
                target_loss = 0

                print('Calculating accuracy')

                for test_images_batch, test_labels_batch in zip(np.array_split(test_images, N), np.array_split(test_labels, N)):
                    feed_dict = {self.model.z: test_images_batch, self.model.labels: test_labels_batch} 
                    target_accuracy_tmp, target_loss_tmp = sess.run([self.model.accuracy, self.model.min_loss], feed_dict) 
                    target_accuracy += target_accuracy_tmp/float(N)
                    target_loss += target_loss_tmp/float(N)

                print(('Target accuracy: [%.4f] target loss: [%.4f]'%(target_accuracy, target_loss)))
	
if __name__=='__main__':

    print('...')


