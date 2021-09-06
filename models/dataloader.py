from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from random import sample

class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        
    def fit(self, df):
        if self.norm_type == "standardization":
            self.mean = df.mean(0)
            self.std = df.std(0)
        elif self.norm_type == "minmax":
            self.max_val = df.max()
            self.min_val = df.min() 
        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":

            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":

            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

# ============================== UCI HAR  ======================================
class UCI_HAR_DATA(Dataset):

    def __init__(self, args, flag="train"):

        self.root_path    = args.root_path
        self.data         = "UCI HAR"
        self.difference   = args.difference
        self.augmentation = args.augmentation
        self.datanorm_type= args.datanorm_type
        self.flag         = flag
        self.read_data()

    def read_data(self):
        print("load the data ", self.root_path, " " , self.data)
        train_x, train_y, test_x, test_y = self.load_the_data(root_path     = self.root_path, 
                                                              norm_type     = self.datanorm_type,
                                                              difference    = self.difference)



        if self.flag == "train":
            print("Train data number : ", len(train_x)/train_x.loc[0].shape[0])
            self.data_x = train_x.copy()
            self.data_y = train_y.copy()

        else:
            print("Test data number : ", len(test_x)/test_x.loc[0].shape[0])
            self.data_x  = test_x.copy()
            self.data_y  = test_y.copy()


        self.nb_classes = len(np.unique(np.concatenate((train_y, test_y), axis=0)))
        print("The number of classes is : ", self.nb_classes)
        self.input_length = self.data_x.loc[0].shape[0]
        self.channel_in = self.data_x.loc[0].shape[1]
        if self.flag == "train":
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)

    def load_the_data(self, root_path, norm_type, difference):
        train_df = pd.read_csv(root_path+"UCI_HAR_Train.csv", index_col=0)
        test_df  = pd.read_csv(root_path+"UCI_HAR_Test.csv",  index_col=0)
        train_label = pd.read_csv(root_path+"y_train.txt",header=None)
        test_label = pd.read_csv(root_path+"y_test.txt",header=None)

        if difference:
            columns = ["diff_"+i for i in train_df.columns]
            grouped_train_df = train_df.groupby(by=train_df.index)
            diff_train_df = grouped_train_df.diff()
            diff_train_df.columns = columns
            diff_train_df.fillna(method ="backfill",inplace=True)

            grouped_test_df = test_df.groupby(by=test_df.index)
            diff_test_df = grouped_test_df.diff()
            diff_test_df.columns = columns
            diff_test_df.fillna(method ="backfill",inplace=True)

            train_df = pd.concat([train_df,diff_train_df], axis=1)
            test_df  = pd.concat([test_df, diff_test_df],  axis=1)


        if norm_type is not None:
            self.normalizer = Normalizer(norm_type)
            self.normalizer.fit(train_df)
            train_df = self.normalizer.normalize(train_df)
            test_df  = self.normalizer.normalize(test_df)

        return train_df, train_label.iloc[:,0].values-1, test_df, test_label.iloc[:,0].values-1
    
    def __getitem__(self, index):
        sample_x = self.data_x.loc[index].values
        sample_y = self.data_y[index]
        return sample_x, sample_y

    def __len__(self):
        return len(self.data_x.index.unique())

def plot_the_uci_har_data_set(train_x, train_y, test_x, test_y):

        
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    axs[0].hist(train_y,bins=len(set(train_y)))
    axs[0].set_title('Train')
    axs[1].hist(test_y,bins=len(set(test_y)))
    axs[1].set_title('Test')
    plt.show()
    label_dict = {1: "WALKING",
                  2: "WALKING_UPSTAIRS",
                  3: "WALKING_DOWNSTAIRS",
                  4: "SITTING",
                  5: "STANDING",
                  6: "LAYING"}


    y_train_test = np.concatenate((train_y, test_y), axis=0)
    classes = set(y_train_test)
    number_of_class = len(classes)
    fig, axs = plt.subplots(nrows=number_of_class, ncols=2, figsize=(15,5*number_of_class))
    for index,i in enumerate(classes):
        label_list = list(np.argwhere(train_y==i).reshape(-1))
        indexsample = sample(label_list,3)
        axs[index,0].plot(train_x.loc[indexsample[0]].reset_index(drop=True))
        axs[index,1].plot(train_x.loc[indexsample[1]].reset_index(drop=True))
        axs[index,0].title.set_text("Class :"+ label_dict[i+1])











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
        self.datanorm_type= args.datanorm_type
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
                                                              norm_type     = self.datanorm_type,
                                                              difference    = self.difference)

        train_y, test_y = self.transform_labels(train_y, test_y)

        if self.flag == "train":
            self.data_x = train_x.copy()
            self.data_y = train_y.copy()
            print("the number of train is : ", len(train_x)/train_x.loc[0].shape[0])
        else:
            self.data_x  = test_x.copy()
            self.data_y  = test_y.copy()
            print("the number of test is : ",  len(test_x)/test_x.loc[0].shape[0])

        self.nb_classes = len(np.unique(np.concatenate((train_y, test_y), axis=0)))
        print("The number of classes is : ", self.nb_classes)
        self.input_length = self.data_x.loc[0].shape[0]
        self.channel_in = self.data_x.loc[0].shape[1]
        if self.flag == "train":
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)

    def load_the_data(self, root_path, data_name, norm_type, difference):
        # TODO CHECK!!
        df_train = pd.read_csv(os.path.join(root_path , "{}/{}_TRAIN.txt".format(data_name,data_name)),  header=None,sep='\s+' )
        df_test  = pd.read_csv(os.path.join(root_path , "{}/{}_TEST.txt".format(data_name,data_name)),   header=None,sep='\s+' )
        train_x = df_train.iloc[:,1:]
        train_y = df_train.iloc[:,0].values
        test_x  = df_test.iloc[:,1:]
        test_y  = df_test.iloc[:,0].values
        train_df =pd.concat([pd.DataFrame(train_x.loc[i].values).reset_index(drop=True).set_index(pd.Series(train_x.shape[1]*[i])) for i in range(train_x.shape[0])], axis=0)
        test_df  =pd.concat([pd.DataFrame(test_x.loc[i].values).reset_index(drop=True).set_index(pd.Series(test_x.shape[1]*[i])) for i in range(test_x.shape[0])], axis=0)
        train_df.columns = ["feature"]
        test_df.columns = ["feature"]

        if difference:
            columns = ["diff_"+i for i in train_df.columns]
            grouped_train_df = train_df.groupby(by=train_df.index)
            diff_train_df = grouped_train_df.diff()
            diff_train_df.columns = columns
            diff_train_df.fillna(method ="backfill",inplace=True)

            grouped_test_df = test_df.groupby(by=test_df.index)
            diff_test_df = grouped_test_df.diff()
            diff_test_df.columns = columns
            diff_test_df.fillna(method ="backfill",inplace=True)

            train_df = pd.concat([train_df,diff_train_df], axis=1)
            test_df  = pd.concat([test_df, diff_test_df],  axis=1)

        if norm_type is not None:
            self.normalizer = Normalizer(norm_type)
            #train_row, train_col = train_x.shape
            #test_row,  test_col  = test_x.shape
            #self.normalizer.fit(pd.DataFrame(train_x.reshape(-1)))
            #train_x = self.normalizer.normalize(pd.DataFrame(train_x.reshape(-1))).values.reshape(train_row, train_col)
            #test_x  = self.normalizer.normalize(pd.DataFrame(test_x.reshape(-1))).values.reshape(test_row, test_col)
            self.normalizer.fit(train_df)
            self.normalizer.fit(train_df)
            train_df = self.normalizer.normalize(train_df)
            test_df  = self.normalizer.normalize(test_df)

        # 如果normalizer 参考informer
        #train_x = np.expand_dims(train_x,2)
        #test_x  = np.expand_dims(test_x,2)
        return train_df, train_y, test_df, test_y

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
        sample_x = self.data_x.loc[index].values
        sample_y = self.data_y[index]
        return sample_x,sample_y

    def __len__(self):
        return len(self.data_x)

def plot_the_ucr_uni_data_set(train_x, train_y, test_x, test_y):

    length = train_x.loc[0].shape[0]
    train_x = train_x.values.reshape(-1,length)
    test_x  = test_x.values.reshape(-1,length)
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

data_loader_dict = {"uci_har" : UCI_HAR_DATA,
                    "ucr_uni" : UCR_TSC_DATA_UNIVARIATE}

plot_dict = {"uci_har" : plot_the_uci_har_data_set,
             "ucr_uni" : plot_the_ucr_uni_data_set}