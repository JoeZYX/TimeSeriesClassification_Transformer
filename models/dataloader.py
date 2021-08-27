from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# ============================== UCR TS DATALOADER ======================================
UCR_2018_with_NAN = ['AllGestureWiimoteX', 'AllGestureWiimoteY',  'AllGestureWiimoteZ',
                      'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend',
                      'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1',
                      'GesturePebbleZ2', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ']

class UCR_TSC_DATA_UNIVARIATE(Dataset):

    def __init__(self, args, flag="train"):

        self.root_path    = args.root_path
        self.data         = args.data_name
        self.difference   = args.difference
        self.augmentation = args.augmentation
        self.normalizer   = args.normalizer
        self.flag         = flag
        # Check if the dataloadt suitble
        if self.data in UCR_2018_with_NAN:
            print("it needs another dataloader")
            raise NotImplementedError


        self.read_data()

    def read_data(self):
        print("load the data ", self.root_path, " " , self.data)
        train_x, train_y, test_x, test_y = self.load_the_data(root_path     = self.root_path, 
                                                              data_name     = self.data, 
                                                              normalizer    = self.normalizer,
                                                              difference    = self.difference)

        train_y, test_y = self.transform_labels(train_y, test_y)

        if self.flag == "train":
            self.data_x = train_x.copy()
            self.data_y = train_y.copy()
            print("the shape of train is : ", train_x.shape)
        else:
            self.data_x  = test_x.copy()
            self.data_y  = test_y.copy()
            print("the shape of test is : ",  test_x.shape)

        self.nb_classes = len(np.unique(np.concatenate((train_y, test_y), axis=0)))
        print("The number of classes is : ", self.nb_classes)
        self.input_length = self.data_x.shape[1]
        self.channel_in = self.data_x.shape[2]

    def load_the_data(self, root_path, data_name, normalizer, difference):
        # TODO CHECK!!
        df_train = pd.read_csv(os.path.join(root_path , "{}/{}_TRAIN.txt".format(data_name,data_name)),  header=None,sep='\s+' )
        df_test  = pd.read_csv(os.path.join(root_path , "{}/{}_TEST.txt".format(data_name,data_name)),   header=None,sep='\s+' )
        train_x = df_train.iloc[:,1:].values
        train_y = df_train.iloc[:,0].values
        test_x  = df_test.iloc[:,1:].values
        test_y  = df_test.iloc[:,0].values
        # 如果normalizer 参考informer
        train_x = np.expand_dims(train_x,2)
        test_x  = np.expand_dims(test_x,2)
        return train_x, train_y, test_x, test_y

    def transform_labels(self, y_train, y_test):
        """
        Transform label to min equal zero and continuous
        For example if we have [1,3,4] --->  [0,1,2]
        """
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        
        return new_y_train, new_y_test

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index]
        return sample_x,sample_y

    def __len__(self):
        return len(self.data_x)

def plot_the_data_set(train_x, train_y, test_x, test_y):

        
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    axs[0].hist(train_y,bins=len(set(train_y)))
    axs[0].set_title('Train')
    axs[1].hist(test_y,bins=len(set(test_y)))
    axs[1].set_title('Test')
    plt.show()

    y_train_test = np.concatenate((train_y, test_y), axis=0)

    classes = set(y_train_test)
    number_of_class = len(classes)
    fig, axs = plt.subplots(nrows=number_of_class, ncols=1, figsize=(15,5*number_of_class))
    for index,i in enumerate(classes):
        for item in test_x[test_y==i]:
            axs[index].plot(item,color="b",label = "test")
        for item in train_x[train_y == i]:
            axs[index].plot(item,color="r",label = "train")   
