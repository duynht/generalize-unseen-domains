from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np,numpy
import csv
import glob
import os
import pickle
import pandas as pd

window_size = 1000
threshold = 60
slide_size = 200 #less than window_size!!!

def data_import(path1, label):

        xx = np.empty([0,window_size,60],float)
        yy = np.empty([0,1],int)

        ###Input data###
        input_mat_files = glob.glob(path1)
        #data import from csv
        print(label)
        path1 = path1[:len(path1)-5]
        print(path1)
        if (label == 'fall'):
            input_mat_files = filter(lambda x: (x.startswith(path1+'fa') \
                                                or x.startswith(path1+'lb') \
                                                or x.startswith(path1+'lc') \
                                                or x.startswith(path1+'tr') \
                                                or x.startswith(path1+'sl') \
                                                or x.startswith(path1+'fw')), input_mat_files)
        else:
            input_mat_files = filter(lambda x: not ((x.startswith(path1+'fa') \
                                                or x.startswith(path1+'lb') \
                                                or x.startswith(path1+'lc') \
                                                or x.startswith(path1+'tr') \
                                                or x.startswith(path1+'sl') \
                                                or x.startswith(path1+'fw'))), input_mat_files)
        
        input_mat_files = sorted(input_mat_files)
        print(input_mat_files)

        for f in input_mat_files:
                print("input_file_name=",f)
                data = loadmat(f)
                data = {k:v for k,v in data.items() if k[0] != '_'}
                data = [data[k] for k in data]
                data = np.array(data)
                data = np.squeeze(data, axis = 0)
                tmp1 = data
                x2 =np.empty([0,window_size,60],float)

                #data import by slide window
                k = 0
                while k <= (len(tmp1) + 1 - 2 * window_size):
                        x = np.dstack(np.array(tmp1[k:k+window_size, 0:60]).T)
                        x2 = np.concatenate((x2, x),axis=0)
                        print(x2)
                        k += slide_size

                xx = np.concatenate((xx,x2),axis=0)
        xx = xx.reshape(len(xx),-1)
    
        # labeling
        if (label == 'fall'):
            yy = np.ones((len(xx),1))
        else:
            yy = np.zeros((len(xx),1))
        print(xx.shape,yy.shape)
        return xx, yy

def mat2csv():
    if not os.path.exists("input_files/"):
            os.makedirs("input_files/")

    for k, folder in enumerate(['bathroom', 'bathroom2', 'bedrooms', 'bedrooms2', 'corridor1', 'corridor2_1','corridor2_2','kitchen','kitchen2','lab2']):
    # for k, folder in enumerate(['lab2']):
        if not os.path.exists("input_files/"+str(folder)+'/'):
            os.makedirs("input_files/"+str(folder)+'/')

        for i, label in enumerate (['fall','nonfall']):
            filepath = '../FallDeFi/interp/'+str(folder)+'/*.mat'
            
            outputfilename1 = 'input_files/'+str(folder)+'/xx_'+str(window_size)+'_'+str(threshold)+'_'+str(label)+".csv"
            outputfilename2 = 'input_files/'+str(folder)+'/yy_'+str(window_size)+'_'+str(threshold)+'_'+str(label)+".csv"       

            x, y = data_import(filepath,str(label))
            with open(outputfilename1, "w") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerows(x)
            with open(outputfilename2, "w") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerows(y)
            print(label + "finish!")

def csv_import():
    x_dic = {}
    y_dic = {}
    print("csv file importing...")
    for i in ['bathroom','bathroom2','bedrooms','bedrooms2','corridor1','corridor2_1','corridor2_2','kitchen','kitchen2','lab2']:
        x_dic[str(i)] = np.empty([0,500,60],float)
        y_dic[str(i)] = np.empty([0,1],int)

        SKIPROW = 2 #Skip every 2 rows -> overlap 800ms to 600ms  (To avoid memory error)
        # num_lines = sum(1 for l in open("../Wifi_Activity_Recognition/input_files/xx_1000_60_" + str(i) + ".csv"))
        skip_idx =[]# [x for x in range(1, num_lines) if x % SKIPROW !=0]
        for label in ['fall','nonfall']:
            xx = np.array(pd.read_csv("input_files/"+str(i)+"/xx_1000_60_"+str(label)+".csv", header=None, skiprows = skip_idx))
            yy = np.array(pd.read_csv("input_files/"+str(i)+"/yy_1000_60_"+str(label)+".csv", header=None, skiprows = skip_idx))

            xx = xx.reshape(len(xx),1000,60)

            # 1000 Hz to 500 Hz (To avoid memory error)
            xx = xx[:,::2,:60]

            x_dic[str(i)] = np.concatenate((x_dic[str(i)],xx),axis = 0)
            y_dic[str(i)] = np.concatenate((y_dic[str(i)],yy),axis = 0)

        print(str(i), "finished...", "xx=", xx.shape, "yy=",  yy.shape)
    return x_dic,y_dic

def csv_shuffle():
    x_dic, y_dic = csv_import()
    for i in ['bathroom','bathroom2','bedrooms','bedrooms2','corridor1','corridor2_1','corridor2_2','kitchen','kitchen2','lab2']:
           shuffle(x_dic[str(i)],y_dic[str(i)],random_state = 0)
           x_path = 'falldefi_'+str(i)+'_images.pkl'
           y_path = 'falldefi_'+str(i)+'_labels.pkl'
           pickle_out = open(x_path,"wb")
           cPickle.dump(np.array(x_dic[str(i)]),pickle_out)
           pickle_out.close()
           pickle_out = open(y_path,"wb")
           cPickle.dump(np.array(y_dic[str(i)]),pickle_out)
           pickle_out.close()

def data_split():
    dic = {'bathroom':0,'bathroom2':1,'bedrooms':2,'bedrooms2':3,'corridor1':4,'corridor2_1':5,'corridor2_2':6,'kitchen':7,'kitchen2':8,'lab2':9}
    x = []
    y = []
    for i in ['bathroom','bathroom2','bedrooms','bedrooms2','corridor1','corridor2_1','corridor2_2','kitchen','kitchen2']:
        x_path = './falldefi_shuffled/falldefi_'+str(i)+'_images.pkl'
        y_path = './falldefi_shuffled/falldefi_'+str(i)+'_labels.pkl'
        pickle_in = open(x_path,"rb")
        xx = pickle.load(pickle_in)
        # xx = xx[int(len(xx)/3):]
        x.append(xx)
        pickle_in.close()
        pickle_in = open(y_path,"rb")
        yy = pickle.load(pickle_in)
        # yy = yy[int(len(yy)/3):]
        pickle_in.close()
        y.append(yy)
        gc.collect()
        #print(str(i),yy)
        print((str(i)+' finish',xx.shape,yy.shape))
    print((np.array(x).shape, np.array(y).shape))
    
    
    # target_test all target is used for test
    target_images = np.array(x[dic['lab2']])
    target_labels = np.array(y[dic['lab2']])

    target_dic = {'X': np.expand_dims(target_images, axis=-1), 'y': np.squeeze(target_labels)}
    pickle.dump(target_dic, open('fdftarget_.pkl'))

    # source_images = pickle.load(open('target_images.pkl','rb'))
    # source_labels = pickle.load(open('target_labels.pkl','rb'))
    
    target_train_images, target_test_images, target_train_labels, target_test_labels = train_test_split(target_images, target_labels, test_size = 0.33, random_state=0)

    #source
    source_images = np.r_[x[dic['bathroom']],x[dic['bathroom2']],x[dic['bedrooms']],x[dic['bedrooms2']],x[dic['corridor1']],x[dic['corridor2_1']],x[dic['corridor2_2']], x[dic['kitchen']],x[dic['kitchen2']]]

    source_labels = np.r_[y[dic['bathroom']],y[dic['bathroom2']],y[dic['bedrooms']],y[dic['bedrooms2']],y[dic['corridor1']],y[dic['corridor2_1']],y[dic['corridor2_2']], y[dic['kitchen']],y[dic['kitchen2']]]

    # source_dic = {'X': np.expand_dims(sourcet_images, axis=-1), 'y': np.squeeze(source_labels)}
    # pickle.dump(target_dic, open('fdfsource.pkl'))

    source_train_images, source_test_images, source_train_labels, source_test_labels = train_test_split(source_images, source_labels, test_size = 0.33, random_state=0)     
    
    #expand dims
    source_train_images = np.expand_dims(source_train_images,axis=-1)
    source_test_images = np.expand_dims(source_test_images,axis=-1)
    target_train_images = np.expand_dims(target_train_images,axis=-1)
    target_test_images = np.expand_dims(target_test_images,axis=-1)
    
    #squeeze dims
    source_train_labels = np.squeeze(source_train_labels)
    source_test_labels = np.squeeze(source_test_labels)
    target_train_labels = np.squeeze(target_train_labels)
    target_test_labels = np.squeeze(target_test_labels)


    print((source_train_images.shape,source_train_labels.shape))
    print((source_test_images.shape,source_test_labels.shape))
    print((target_train_images.shape,target_train_labels.shape))
    print((target_test_images.shape,target_test_labels.shape))

    source_train_dic = {'X': source_train_images, 'y': source_train_labels}
    source_test_dic = {'X': source_test_images, 'y': source_test_labels}
    test_train_dic = {'X': target_train_images, 'y': target_train_labels}
    test_test_dic = {'X': target_test_images, 'y': target_test_labels}

    # source_df = pd.DataFrame(source_dic)
    # test_df = pd.DataFrame(test_dic)

    pickle.dump(source_train_dic, open('fdfsource_train.pkl','wb'))
    pickle.dump(source_test_dic, open('fdfsource_test.pkl','wb'))
    pickle.dump(target_train_dic, open('fdftarget_train.pkl','wb'))
    pickle.dump(target_test_dic, open('fdftarget_test.pkl','wb'))

def main():
    print('mat2csv...')
    mat2csv()
    print('csv_shuffle...')
    csv_shuffle()
    print('data_split...')
    data_split()
    print('done!')

if __name__ == "__main__":
    main()