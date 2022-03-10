# -*- coding: utf-8 -*-
"""
@author: user
"""

import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np

from .utils.torchsummary import summary as _summary
from .utils.WrappedDataLoader import WrappedDataLoader
from .utils.History import History
### множество входов
class Layer_01(nn.Module):
    def __init__(self, *input_shapes , **kwargs):
        super(Layer_01, self).__init__(**kwargs )
        self.input_shapes = input_shapes
        
        self._criterion = None
        self._optimizer = None
        
    def reset_parameters(self):
        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            return (-lim, lim)
        
        for module in self._modules.values():
            if hasattr(module, 'weight') and (module.weight is not None):
                module.weight.data.uniform_(*hidden_init(module))
            if hasattr(module, 'bias') and (module.bias is not None):
                module.bias.data.fill_(0)

    def _get_regularizer(self):
        raise Exception("Need to override method _get_regularizer()!");
        
    def summary(self):
        _summary(self, input_size = self.input_shapes, device = self.device)

    def weights_is_nan(self):
        is_nan = False
        for module in self._modules.values():
            if hasattr(module, 'weight'):
                if ((isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach())) or
                     (isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach()))):
                    is_nan = True
                    break
            if hasattr(module, 'bias'):
                if (isinstance(module.bias, torch.Tensor) and torch.isnan(torch.sum(module.bias.data).detach())):
                    is_nan = True
                    break
            
        return is_nan

    def save_state(self, file_path):
        torch.save(self.state_dict(), file_path)
        
    def load_state(self, file_path):
        try:
            print()
            print('Loading preset weights... ', end='')

            self.load_state_dict(torch.load(file_path))
            self.eval()
            is_nan = False
            for module in self._modules.values():
                if hasattr(module, 'weight'):
                    if ((isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach())) or
                         (isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach()))):
                        is_nan = True
                        break
                if hasattr(module, 'bias'):
                    if (isinstance(module.bias, torch.Tensor) and torch.isnan(torch.sum(module.bias.data).detach())):
                        is_nan = True
                        break
                    
            if (is_nan):
                raise Exception("[Error]: Parameters of layers is NAN!")
                
            print("Ok.")
        except Exception as e:
            print("Fail! ", end='')
            print(str(e))
            print("[Action]: Reseting to random values!")
            self.reset_parameters()

    def compile(self, criterion, optimizer, **kwargs):
        if criterion == 'mse-mean':
            self._criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mse-sum':
            self._criterion = nn.MSELoss(reduction='sum')
        else:
            raise Exception("Unknown loss-function!")
            
        if (optimizer == 'sgd'):
            lr = 0.001
            momentum = 0.2
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            if ('momentum' in kwargs.keys()):
                momentum = kwargs['momentum']
            self._optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum);
        elif (optimizer == 'adam'):
            lr = 0.001
            betas = (0.9, 0.999)
            eps = 1e-8
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            if ('betas' in kwargs.keys()):
                momentum = kwargs['betas']
            if ('eps' in kwargs.keys()):
                eps = kwargs['eps']

            self._optimizer = optim.Adam(self.parameters(), lr=lr, betas=betas, eps=eps);
        else:
            raise Exception("Unknown optimizer!")
            
    def fit(self, x, y, batch_size, epochs = 1, validation_data = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)

        _x = []
        for xi in x:
            if isinstance(xi, (np.ndarray, list, tuple)):
                xi = torch.FloatTensor(xi).to(self.device)
            _x.append(xi)
        x = tuple(_x)
        
        if isinstance(y, (np.ndarray, list, tuple)):
            y = torch.FloatTensor(y).to(self.device)

        train_ds = TensorDataset(*x, y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        train_dl = WrappedDataLoader(train_dl)
        if validation_data is not None:
            x_test = validation_data[0]
            y_test = validation_data[1]
            
            _x_test = []
            for xi in x_test:
                if isinstance(xi, (np.ndarray, list, tuple)):
                    xi = torch.FloatTensor(xi).to(self.device)
                _x_test.append(xi)
            x_test = tuple(_x_test)
            
            y_test = x_test if isinstance(y_test, torch.Tensor) else \
                        torch.FloatTensor(y_test).to(self.device)

            valid_ds = TensorDataset(*x_test, y_test)
            valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)
            valid_dl = WrappedDataLoader(valid_dl)
        else:
            valid_ds = None
            valid_dl = None

        def loss_batch(loss_func, xb, yb, opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
                #logsoftmax = nn.LogSoftmax()
                #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
                #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
                #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
                #logsoftmax = nn.LogSoftmax()
                #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
                #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
                #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

            pred = self(*xb)
            if isinstance(pred, tuple):
                pred0 = pred[0]
                del pred
            else:
                pred0 = pred
            
#            if (len(yb.shape) > 1):
#                _, yb = torch.max(yb, dim = 1)
            loss = loss_func(pred0, yb)
            #loss = cross_entropy(pred, yb)
            
            del pred0
            
            #_, predicted = torch.max(pred.data, dim = 1)
            #_, ind_target = torch.max(yb, dim = 1)
            #correct = (predicted == ind_target).sum().item()
            #acc = correct / len(yb) #.size(0)
            
            _regularizer = self._get_regularizer()
            
            reg_loss = 0
            for param in self.parameters():
                reg_loss += _regularizer(param)
                
            loss += reg_loss
                
            if opt is not None:
                with torch.no_grad():
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            print("*", end='')
        
            loss_item = loss.item()
            
            del loss
            del reg_loss
            
            return loss_item, len(yb)#, acc

        history = History()
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            
            print("[", end='')
            losses, nums = zip(
                *(loss_batch(self._criterion, (xb0, xb1, xb2), yb, self._optimizer) for xb0, xb1, xb2, yb in train_dl)
            )
            print("]", end='')
            
            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            #acc = np.sum(np.multiply(accs, nums)) / sum_nums
            
            if validation_data is not None:
                if (len(valid_dl) == 0):
                    raise Exception("Test data are empty!")
                    
                self.eval()
                
                print("[", end='')
                with torch.no_grad():
                    losses, nums = zip(
                        *(loss_batch(self._criterion, (xb0, xb1, xb2), yb) for xb0, xb1, xb2, yb in valid_dl)
                    )
                print("]", end='')
                
                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                #val_acc = np.sum(np.multiply(accs, nums)) / sum_nums
                
                #print(', Test-accuracy: {:.6f}'.format(val_acc), end='')
                
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})
                
                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')
                
            print("")
            
        del x
        del y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
######################################################################################3
    #def fit_loader(self, x, y, batch_size, epochs = 1, validation_data = None):
    def fit_dataloader(self, loader, batch_size, epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


        def loss_batch_01(loss_func, xb, yb, opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
                #logsoftmax = nn.LogSoftmax()
                #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
                #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
                #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
                #logsoftmax = nn.LogSoftmax()
                #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
                #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
                #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
             

            pred = self(*xb)
             
            if isinstance(pred, tuple):
                pred0 = pred[0]
                del pred
            else:
                pred0 = pred
            
#            if (len(yb.shape) > 1):
#                _, yb = torch.max(yb, dim = 1)
             

            loss = loss_func(pred0, yb)
            #loss = cross_entropy(pred, yb)
             
            del pred0
               
            #_, predicted = torch.max(pred.data, dim = 1)
            #_, ind_target = torch.max(yb, dim = 1)
            #correct = (predicted == ind_target).sum().item()
            #acc = correct / len(yb) #.size(0)
             
            _regularizer = self._get_regularizer()
            
            reg_loss = 0
            for param in self.parameters():
                reg_loss += _regularizer(param)
                
            loss += reg_loss
             
            if opt is not None:
                with torch.no_grad():
                      
                    opt.zero_grad()
                     
                    loss.backward()
                     
                    opt.step()
                     
            
            print("*", end='')
        
            loss_item = loss.item()
            
            del loss
            del reg_loss
             
            return loss_item, len(yb)#, acc
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№

        history = History()
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            ########################################3
            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            for s in loader:
            #s = next(iter(loader))

                train_ds = TensorDataset( torch.FloatTensor(s['images_ref'].numpy()).to(self.device)   ,\
                                         torch.FloatTensor(s['sketch_cur'].numpy()).to(self.device)   ,\
                                         torch.FloatTensor(s['sketch_ref'].numpy() ).to(self.device) ,\
                                         torch.FloatTensor(s['images_cur'].numpy() ).to(self.device))
                 
                model_SA_predict,_=loader.model(train_ds.tensors[0].to(loader.device),train_ds.tensors[1].to(loader.device),\
                                                train_ds.tensors[2].to(loader.device))
                 
                 
                 
                losses_, nums_   =   loss_batch_01(self._criterion, \
                                                   (train_ds.tensors[0],train_ds.tensors[1],\
                                                    torch.FloatTensor(model_SA_predict.detach().numpy() ).to(self.device)),\
                                                    train_ds.tensors[3], \
                                                   self._optimizer)
                 
                losses.append(losses_)
                nums.append(nums_ )
            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            ######################################
             
            ### test mode ###
            if validation_loader is not None:
                self.eval()
                
                 
                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))

                    val_ds = TensorDataset( torch.FloatTensor(s['images_ref'].numpy()).to(self.device)   ,\
                                             torch.FloatTensor(s['sketch_cur'].numpy()).to(self.device)   ,\
                                             torch.FloatTensor(s['sketch_ref'].numpy() ).to(self.device) ,\
                                             torch.FloatTensor(s['images_cur'].numpy() ).to(self.device))
                    
                    model_SA_predict,_=loader.model(val_ds.tensors[0].to(loader.device),val_ds.tensors[1].to(loader.device),\
                                                    val_ds.tensors[2].to(loader.device))
                    losses_, nums_   =   loss_batch_01( self._criterion, \
                                                       (val_ds.tensors[0],val_ds.tensors[1],\
                                                        torch.FloatTensor(model_SA_predict.detach().numpy()).to(self.device)),\
                                                       val_ds.tensors[3], self._optimizer)     
                    losses.append(losses_)
                    nums.append(nums_ )
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                    #acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})
                
                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')
                
            print("")
            
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
#################################################################
    def _call_simple_layer(self, name_layer, x):
        y = self._modules[name_layer](x)
        if self.device.type == 'cuda' and not y.is_contiguous():
            y = y.contiguous()
        return y
    
    def _contiguous(self, x):
        if self.device.type == 'cuda' and not x.is_contiguous():
            x = x.contiguous()
        return x
#####################################################################3