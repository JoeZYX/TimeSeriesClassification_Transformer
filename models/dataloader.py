from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from random import sample
from sktime.utils import load_data
import glob
import re
import csv
from scipy.stats import stats
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
        elif self.norm_type == "per_sample_std":
            self.max_val = None
            self.min_val = None
        elif self.norm_type == "per_sample_minmax":
            self.max_val = None
            self.min_val = None
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
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')
        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

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
        return len(self.data_x.index.unique())

def plot_the_ucr_uni_data_set(train_x, train_y, test_x, test_y):

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
        for item in np.where(test_y == i)[0]:
            axs[index].plot(test_x.loc[item].reset_index(drop=True),color="b",label = "test")
        for item in np.where(train_y == i)[0]:
            axs[index].plot(train_x.loc[item].reset_index(drop=True),color="r",label = "train")  

# ======================================= UCR MULTI TIME SERIES ====================================
def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y

class UCR_TSC_DATA_MULTIVARIATE(Dataset):
    def __init__(self, args, flag="train"):

        self.root_path    = args.root_path
        self.data_name    = args.data_name
        
        self.difference   = args.difference
        self.augmentation = args.augmentation
        self.datanorm_type= args.datanorm_type
        self.flag         = flag


        self.read_data()
    def read_data(self):
        print("load the data ", self.root_path, " " , self.data_name)
        train_x, train_y, test_x, test_y = self.load_the_data(root_path     = self.root_path, 
                                                              data_name     = self.data_name, 
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
    
    def check_if_all_sensor_same_length(self, data_frame):
        lengths = data_frame.applymap(lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            return True
        else:
            return False
    def check_if_all_data_same_length(self, data_frame):
        lengths = data_frame.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            return True
        else:
            return False
        
    def load_the_data(self, root_path, data_name, norm_type, difference):
        # get the right data_file 
        data_paths = glob.glob(os.path.join(os.path.join(root_path,data_name), '*'))
        train_path = list(filter(lambda x: re.search("TRAIN", x), data_paths))
        test_path = list(filter(lambda x: re.search("TEST", x), data_paths))
        assert len(train_path)==1
        assert len(test_path)==1
        train_path = train_path[0]
        test_path = test_path[0]

        # load the data
        train_df, train_labels = load_data.load_from_tsfile_to_dataframe(train_path, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
        train_labels = pd.Series(train_labels, dtype="category")
        train_labels = pd.DataFrame(train_labels.cat.codes, dtype=np.int8)

        test_df, test_labels = load_data.load_from_tsfile_to_dataframe(test_path, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
        test_labels = pd.Series(test_labels, dtype="category")
        test_labels = pd.DataFrame(test_labels.cat.codes, dtype=np.int8)  
        
        test_labels = test_labels.iloc[:,0].values
        train_labels = train_labels.iloc[:,0].values

        # check the data valid
        flag_train_0 = self.check_if_all_sensor_same_length(train_df.copy())
        flag_test_0  = self.check_if_all_sensor_same_length(test_df.copy())
        flag_train_1 = self.check_if_all_data_same_length(train_df.copy())
        flag_test_1  = self.check_if_all_data_same_length(test_df.copy())
        if flag_train_0 or flag_test_0 or flag_train_1 or flag_test_1:
            raise Exception("change a dataset")

        # 
        lengths_train = train_df.applymap(lambda x: len(x)).values
        train_df = pd.concat((pd.DataFrame({col: train_df.loc[row, col] for col in train_df.columns})\
                              .reset_index(drop=True).set_index(pd.Series(lengths_train[row, 0]*[row])) \
                              for row in range(train_df.shape[0])), axis=0)
        lengths_test = test_df.applymap(lambda x: len(x)).values
        test_df  = pd.concat((pd.DataFrame({col: test_df.loc[row, col] for col in test_df.columns})\
                              .reset_index(drop=True).set_index(pd.Series(lengths_test[row, 0]*[row])) \
                              for row in range(test_df.shape[0])), axis=0)

        # Replace NaN values
        traingrp = train_df.groupby(by=train_df.index)
        train_df = traingrp.transform(interpolate_missing)

        testgrp = test_df.groupby(by=test_df.index)
        test_df = testgrp.transform(interpolate_missing) 
        
        
        
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
        
        return train_df, train_labels, test_df, test_labels

    def __getitem__(self, index):
        sample_x = self.data_x.loc[index].values
        sample_y = self.data_y[index]
        return sample_x, sample_y

    def __len__(self):
        return len(self.data_x.index.unique())

def plot_the_ucr_multi_data_set(train_x, train_y, test_x, test_y, col_plot=0):

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
        for item in np.where(test_y == i)[0]:
            axs[index].plot(test_x.loc[item].iloc[:,col_plot].reset_index(drop=True),color="b",label = "test")
        for item in np.where(train_y == i)[0]:
            axs[index].plot(train_x.loc[item].iloc[:,col_plot].reset_index(drop=True),color="r",label = "train")





# ================================= PAMAP2 HAR DATASET ============================================
class PAMAP2_HAR_DATA(Dataset):

    def __init__(self, args, flag="train"):

        self.root_path    = args.root_path
        self.data_name    = "PAMAP2 HAR"
        self.difference   = args.difference
        self.augmentation = args.augmentation
        self.datanorm_type= args.datanorm_type
        self.flag         = flag
        self.used_cols    = [1,  # Label
                             4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # IMU Hand
                             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # IMU Chest
                             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49  # IMU ankle
                            ]
        self.train_keys   = ['subject101', 'subject102', 'subject103', 
                             'subject104', 'subject105', 
                             'subject107', 'subject108', 'subject109']
        self.test_keys    = ['subject106']
        
        col_names=['activity_id']

        IMU_locations = ['hand', 'chest', 'ankle']
        IMU_data      = ['acc_16_01', 'acc_16_02', 'acc_16_03',
                         'acc_06_01', 'acc_06_02', 'acc_06_03',
                         'gyr_01', 'gyr_02', 'gyr_03',
                         'mag_01', 'mag_02', 'mag_03']

        self.col_names = col_names + [item for sublist in [[dat+'_'+loc for dat in IMU_data] for loc in IMU_locations] for item in sublist]

        self.read_data()
        
    def read_data(self):
        print("load the data ", self.root_path, " " , self.data_name)
        train_x, train_y, test_x, test_y = self.load_the_data(root_path     = self.root_path, 
                                                              data_name     = self.data_name, 
                                                              norm_type     = self.datanorm_type,
                                                              difference    = self.difference)
        

        train_y, test_y = self.transform_labels(train_y, test_y)


        if self.flag == "train":

            self.data_x = train_x.reset_index(drop=True).copy()
            self.data_y = train_y.reset_index(drop=True).copy()
            self.get_the_sliding_index(train_x.copy(),train_y.copy())
            print("The number of training data is : ", len(self.sliding_index.keys()))
        else:

            self.data_x  = test_x.reset_index(drop=True).copy()
            self.data_y  = test_y.reset_index(drop=True).copy()
            self.get_the_sliding_index(test_x.copy(), test_y.copy())
            print("The number of test data is : ", len(self.sliding_index.keys()))

        self.nb_classes = len(np.unique(np.concatenate((train_y, test_y), axis=0)))
        print("The number of classes is : ", self.nb_classes)
        self.input_length = self.data_x.iloc[self.sliding_index[0][0]:self.sliding_index[0][1],:].shape[0]
        self.channel_in = self.data_x.iloc[self.sliding_index[0][0]:self.sliding_index[0][1],:].shape[1]
        if self.flag == "train":
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)
        
    def load_the_data(self, root_path, data_name, norm_type, difference):
        file_list = os.listdir(root_path)
        
        df_dict = {}
        for file in file_list:
            sub_data = pd.read_table(os.path.join(root_path,file), header=None, sep='\s+')
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names
            drop_index = list(sub_data.index[(sub_data['activity_id'].isin([0,9,10,11,18,19,20]))])
            sub_data = sub_data.drop(drop_index)
            #sub_data.fillna(method ="backfill",inplace=True)  拿到了97，1的成绩
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            sub_data['sub_id'] =int(file[9])
            df_dict[file.split(".")[0]] = sub_data   

        train_df = pd.DataFrame()
        for key in self.train_keys:
            train_df = pd.concat([train_df,df_dict[key]])

        test_df = pd.DataFrame()
        for key in self.test_keys:
            test_df = pd.concat([test_df,df_dict[key]])
        
        train_df = train_df.set_index('sub_id')
        train_x = train_df.iloc[:,1:]
        train_y = train_df.iloc[:,0]

        test_df = test_df.set_index('sub_id')
        test_x = test_df.iloc[:,1:]
        test_y = test_df.iloc[:,0]     
        

        if difference:
            columns = ["diff_"+i for i in train_x.columns]

            grouped_train_x = train_x.groupby(by=train_x.index)
            diff_train_x = grouped_train_x.diff()
            diff_train_x.columns = columns
            diff_train_x.fillna(method ="backfill",inplace=True)

            grouped_test_x = test_x.groupby(by=test_x.index)
            diff_test_x = grouped_test_x.diff()
            diff_test_x.columns = columns
            diff_test_x.fillna(method ="backfill",inplace=True)

            train_x = pd.concat([train_x,diff_train_x], axis=1)
            test_x  = pd.concat([test_x, diff_test_x],  axis=1)


        if norm_type:
            normalizer = Normalizer(norm_type)
            normalizer.fit(train_x)
            train_x = normalizer.normalize(train_x)
            test_x  = normalizer.normalize(test_x)
            
        return train_x, train_y, test_x, test_y
    
    def get_the_sliding_index(self, data_x, data_y):
        data_x = data_x.reset_index()
        data_y = data_y.reset_index()
        data_x["activity_id"] = data_y["activity_id"]
        data_x['act_block'] = ((data_x['activity_id'].shift(1) != data_x['activity_id']) | (data_x['sub_id'].shift(1) != data_x['sub_id'])).astype(int).cumsum()

        freq         = 100
        windowsize   = int(5.12 * freq)
        displacement = 1*freq
        drop_long    = 7.5
        train_window = {}
        id_          = 0

        drop_index = []
        numblocks = data_x['act_block'].max()
        for block in range(1, numblocks+1):
            drop_index += list(data_x[data_x['act_block']==block].head(int(drop_long * freq)).index)
            drop_index += list(data_x[data_x['act_block']==block].tail(int(drop_long * freq)).index)
        dropped_data_x = data_x.drop(drop_index)

        for index in dropped_data_x.act_block.unique():
            temp_df = dropped_data_x[dropped_data_x["act_block"]==index]

            start = temp_df.index[0]
            end   = start+windowsize

            while end < temp_df.index[-1]:
                train_window[id_]=[start, end]
                id_ = id_ + 1
                start = start + displacement
                end   = start + windowsize
        self.sliding_index = train_window

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
        
        new_y_train = pd.Series(new_y_train)
        new_y_train.index = y_train.index
        new_y_train.name = "activity_id"
        
        new_y_test =  pd.Series(new_y_test)
        new_y_test.index = y_test.index
        new_y_test.name = "activity_id"

        return new_y_train, new_y_test    
        
    def __getitem__(self, index):
        start, end = self.sliding_index[index]
        sample_x = self.data_x.iloc[start:end,:].values
        sample_y = stats.mode(self.data_y.iloc[start:end]).mode[0]
        return sample_x, sample_y

    def __len__(self):
        return len(self.sliding_index.keys())




# ========================================       Opportunity HAR UCI                =============================
class Opportunity_UCI_Data(Dataset):

    def __init__(self, args, flag="train"):

        self.root_path    = args.root_path
        self.data_name    = "Opportunity"
        self.difference   = args.difference
        self.augmentation = args.augmentation
        self.datanorm_type= args.datanorm_type
        self.flag         = flag

        self.used_cols = [1,  2,   3, # Accelerometer RKN^ 
                          4,  5,   6, # Accelerometer HIP
                          7,  8,   9, # Accelerometer LUA^ 
                          10, 11,  12, # Accelerometer RUA_
                          13, 14,  15, # Accelerometer LH
                          16, 17,  18, # Accelerometer BACK
                          19, 20,  21, # Accelerometer RKN_ 
                          22, 23,  24, # Accelerometer RWR
                          25, 26,  27, # Accelerometer RUA^
                          28, 29,  30, # Accelerometer LUA_ 
                          31, 32,  33, # Accelerometer LWR
                          34, 35,  36, # Accelerometer RH
                          37, 38,  39, 40, 41, 42, 43, 44, 45, # InertialMeasurementUnit BACK
                          50, 51,  52, 53, 54, 55, 56, 57, 58, # InertialMeasurementUnit RUA
                          63, 64,  65, 66, 67, 68, 69, 70, 71, # InertialMeasurementUnit RLA 
                          76, 77,  78, 79, 80, 81, 82, 83, 84, # InertialMeasurementUnit LUA
                          89, 90,  91, 92, 93, 94, 95, 96, 97,  # InertialMeasurementUnit LLA
                          102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, # InertialMeasurementUnit L-SHOE
                          118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, # InertialMeasurementUnit R-SHOE
                          249  # Label
                         ]
        self.train_keys   = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 
                             'S1-ADL5.dat', 'S1-Drill.dat', # subject 1
                             
                             'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-Drill.dat', # subject 2
                             
                             'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-Drill.dat', # subject 3
                             
                             'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat',
                             'S4-ADL5.dat', 'S4-Drill.dat'] # subject 4

        self.test_keys    = ['S2-ADL4.dat', 'S2-ADL5.dat','S3-ADL4.dat', 'S3-ADL5.dat']
        
        
        col_names         = ["dim_{}".format(i) for i in range(len(self.used_cols)-1)]
        self.col_names    =  col_names + ["activity_id"]
        
        self.label_map = [(0,      'Other'),
                          (406516, 'Open Door 1'),
                          (406517, 'Open Door 2'),
                          (404516, 'Close Door 1'),
                          (404517, 'Close Door 2'),
                          (406520, 'Open Fridge'),
                          (404520, 'Close Fridge'),
                          (406505, 'Open Dishwasher'),
                          (404505, 'Close Dishwasher'),
                          (406519, 'Open Drawer 1'),
                          (404519, 'Close Drawer 1'),
                          (406511, 'Open Drawer 2'),
                          (404511, 'Close Drawer 2'),
                          (406508, 'Open Drawer 3'),
                          (404508, 'Close Drawer 3'),
                          (408512, 'Clean Table'),
                          (407521, 'Drink from Cup'),
                          (405506, 'Toggle Switch')]
        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        
        self.read_data()


    def read_data(self):
        print("load the data ", self.root_path, " " , self.data_name)
        train_x, train_y, test_x, test_y = self.load_the_data(root_path     = self.root_path, 
                                                              data_name     = self.data_name, 
                                                              norm_type     = self.datanorm_type,
                                                              difference    = self.difference)


        # 这里index都是sub——id
        if self.flag == "train":

            self.data_x = train_x.reset_index(drop=True).copy()
            self.data_y = train_y.reset_index(drop=True).copy()
            self.get_the_sliding_index(train_x.copy(),train_y.copy())
            print("The number of training data is : ", len(self.sliding_index.keys()))
        else:

            self.data_x  = test_x.reset_index(drop=True).copy()
            self.data_y  = test_y.reset_index(drop=True).copy()
            self.get_the_sliding_index(test_x.copy(), test_y.copy())
            print("The number of test data is : ", len(self.sliding_index.keys()))

        self.nb_classes = len(np.unique(np.concatenate((train_y, test_y), axis=0)))
        print("The number of classes is : ", self.nb_classes)
        self.input_length = self.data_x.iloc[self.sliding_index[0][0]:self.sliding_index[0][1],:].shape[0]
        self.channel_in = self.data_x.iloc[self.sliding_index[0][0]:self.sliding_index[0][1],:].shape[1]
        if self.flag == "train":
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)
    
    def load_the_data(self, root_path, data_name, norm_type, difference):
        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if file[-3:]=="dat"]
        df_dict = {}

        for index,file in enumerate(file_list):
            sub_data = pd.read_table(os.path.join(root_path,file), header=None, sep='\s+')
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names
            # 检查过了 label那一列 有18个, 且没有缺失
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            sub_data["activity_id"] = sub_data["activity_id"].map(self.labelToId)
            sub_data['sub_id'] =int(index)
            df_dict[file] = sub_data
            
        train_df = pd.DataFrame()
        for key in self.train_keys:
            train_df = pd.concat([train_df,df_dict[key]])

        test_df = pd.DataFrame()
        for key in self.test_keys:
            test_df = pd.concat([test_df,df_dict[key]])
        
        train_df = train_df.set_index('sub_id')
        train_x = train_df.iloc[:,:-1]
        train_y = train_df.iloc[:,-1]

        test_df = test_df.set_index('sub_id')
        test_x = test_df.iloc[:,:-1]
        test_y = test_df.iloc[:,-1]  

        if difference:
            columns = ["diff_"+i for i in train_x.columns]

            grouped_train_x = train_x.groupby(by=train_x.index)
            diff_train_x = grouped_train_x.diff()
            diff_train_x.columns = columns
            diff_train_x.fillna(method ="backfill",inplace=True)

            grouped_test_x = test_x.groupby(by=test_x.index)
            diff_test_x = grouped_test_x.diff()
            diff_test_x.columns = columns
            diff_test_x.fillna(method ="backfill",inplace=True)

            train_x = pd.concat([train_x,diff_train_x], axis=1)
            test_x  = pd.concat([test_x, diff_test_x],  axis=1)


        if norm_type:
            normalizer = Normalizer(norm_type)
            normalizer.fit(train_x)
            train_x = normalizer.normalize(train_x)
            test_x  = normalizer.normalize(test_x)
            
        return train_x, train_y, test_x, test_y

    def get_the_sliding_index(self, data_x, data_y):
        data_x = data_x.reset_index()
        data_y = data_y.reset_index()
        data_x["activity_id"] = data_y["activity_id"]
        data_x['act_block'] = ((data_x['activity_id'].shift(1) != data_x['activity_id']) | (data_x['sub_id'].shift(1) != data_x['sub_id'])).astype(int).cumsum()

        # TODO 设置windowsize 以及 sliding step
        freq         = 30
        windowsize   = int(2 * freq)
        displacement = int(0.5*freq)
        drop_long    = 1
        train_window = {}
        id_          = 0

        drop_index = []
        numblocks = data_x['act_block'].max()
        for block in range(1, numblocks+1):
            drop_index += list(data_x[data_x['act_block']==block].head(int(drop_long * freq)).index)
            drop_index += list(data_x[data_x['act_block']==block].tail(int(drop_long * freq)).index)
        dropped_data_x = data_x.drop(drop_index)

        for index in dropped_data_x.act_block.unique():
            temp_df = dropped_data_x[dropped_data_x["act_block"]==index]

            start = temp_df.index[0]
            end   = start+windowsize

            while end < temp_df.index[-1]:
                train_window[id_]=[start, end]
                id_ = id_ + 1
                start = start + displacement
                end   = start + windowsize
        self.sliding_index = train_window

    def __getitem__(self, index):
        start, end = self.sliding_index[index]
        sample_x = self.data_x.iloc[start:end,:].values
        sample_y = stats.mode(self.data_y.iloc[start:end]).mode[0]
        return sample_x, sample_y

    def __len__(self):
        return len(self.sliding_index.keys())
# ================================================

data_loader_dict = {"uci_har"   : UCI_HAR_DATA,
                    "ucr_uni"   : UCR_TSC_DATA_UNIVARIATE,
                    "ucr_multi" : UCR_TSC_DATA_MULTIVARIATE,
                    "pamap2"    : PAMAP2_HAR_DATA,
                    "opport"    : Opportunity_UCI_Data}

plot_dict = {"uci_har"    : plot_the_uci_har_data_set,
             "ucr_uni"    : plot_the_ucr_uni_data_set,
             "ucr_multi"  : plot_the_ucr_multi_data_set}