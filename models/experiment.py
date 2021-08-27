import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
import time
from models.dataloader import UCR_TSC_DATA_UNIVARIATE
from models.model import TSCtransformer
from sklearn.metrics import accuracy_score


Data_Loader_Dict = {"ucr_univariante" : UCR_TSC_DATA_UNIVARIATE}

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
  
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("new best score!!!!")
            self.best_score = score
            self.save_checkpoint(val_loss, model,path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss


class adjust_learning_rate_class:
    def __init__(self, args, verbose):
        self.patience = args.learning_rate_patience
        self.factor   = args.learning_rate_factor
        self.learning_rate = args.learning_rate
        self.args = args
        self.verbose = verbose
        self.val_loss_min = np.Inf
        self.counter = 0
        self.best_score = None
    def __call__(self, optimizer, val_loss):
        # val_loss 是正值，越小越好
        # 但是这里加了负值，score愈大越好
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.counter += 1
        elif score <= self.best_score :
            self.counter += 1
            if self.verbose:
                print(f'Learning rate adjusting counter: {self.counter} out of {self.patience}')
        else:
            if self.verbose:
                print("new best score!!!!")
            self.best_score = score
            self.counter = 0
            
        if self.counter == self.patience:
            self.learning_rate = self.learning_rate * self.factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                if self.verbose:
                    print('Updating learning rate to {}'.format(self.learning_rate))
            self.counter = 0

class Exp(object):
    def __init__(self, args):
        self.args = args
        # set the device
        self.device = self.acquire_device()
        self.model  = self.build_model().to(self.device)
        self.optimizer_dict = {"Adam":optim.Adam}
        self.criterion_dict = {"MSE":nn.MSELoss,"CrossEntropy":nn.CrossEntropyLoss}




    def acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def build_model(self):

        model  = TSCtransformer(self.args)
        print("Build the model!")
        return model.double()

    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError
        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError
        criterion = self.criterion_dict[self.args.criterion]()
        return criterion

    def _get_data(self, flag="train"):
        if flag == 'train':
            shuffle_flag = True
        else:
            shuffle_flag = False

        data_set = Data_Loader_Dict[self.args.dataloader]( args = self.args, flag = flag )
        data_loader = DataLoader(data_set, 
                                 batch_size   =  self.args.batch_size,
                                 shuffle      =  shuffle_flag,
                                 num_workers  =  0,
                                 drop_last    =  False)

        return data_set, data_loader

    def train(self, save_path):
        train_data, train_loader = self._get_data(flag = 'train')
        test_data, test_loader   = self._get_data(flag = 'test')

        path = './logs/'+save_path
        if not os.path.exists(path):
            os.makedirs(path)



        train_steps = len(train_loader)

        early_stopping        = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
        learning_rate_adapter = adjust_learning_rate_class(self.args,True)

        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):

            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):

                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.long().to(self.device)

                # model prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x)[0]
                else:
                    outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss , vali_acc = self.validation(test_loader, criterion)
            _ , train_acc = self.validation(train_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Accuracy {3:.7f} Vali Loss: {4:.7f} Vali Accuracy: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, train_acc, vali_loss, vali_acc))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #adjust_learning_rate(model_optim, epoch+1, self.args)
            learning_rate_adapter(model_optim,vali_loss)
        
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))


    def validation(self, data_loader, criterion):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        for i, (batch_x,batch_y) in enumerate(data_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.long().to(self.device)
            
            # prediction
            if self.args.output_attention:
                outputs = self.model(batch_x)[0]
            else:
                outputs = self.model(batch_x)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true) 

            total_loss.append(loss)
            preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
            trues.extend(list(batch_y.detach().cpu().numpy()))            
        total_loss = np.average(total_loss)
        acc = accuracy_score(preds,trues)
        self.model.train()
        return total_loss,  acc


