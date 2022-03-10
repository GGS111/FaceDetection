#!pip3 install torch==1.5.0 !pip3 install torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html 
import torch

from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from .layers.Lambda import Lambda
from .utils.torchsummary import summary as _summary
from .utils.History import History
from .utils.Regularizer import Regularizer
 
from torch.utils.data import TensorDataset
import numpy as np
from enum import Enum
###########################################
txt_00_f='./txt_00.JPG'
#############################################################################
    
############################################################################################
class Layer_06(torch.nn.Module):
    def __init__(self, *input_shapes , **kwargs):
        super(Layer_06, self).__init__(**kwargs )
        self.input_shapes = input_shapes
        self.eps_=10**(-20)
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
        
    def load_state(self, file_path, map_location = None):
        try:
            print()
            print('Loading preset weights... ', end='')

            self.load_state_dict(torch.load(file_path, map_location))
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
            
    def cross_entropy_00(self, pred, soft_targets):
        return -torch.log(self.eps_+torch.mean(torch.sum(soft_targets * pred, -1)))
    
    def MSE_00(self, pred, soft_targets):
        return   torch.mean(torch.mean((soft_targets - pred)**2, -1)) 

    def compile(self, criterion, optimizer,   **kwargs):
        
        if criterion == 'mse-mean':
            self._criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mse-sum':
            self._criterion = nn.MSELoss(reduction='sum')
        elif criterion == '000':
            self._criterion = self.MSE_00 
        elif criterion == '001':
            self._criterion = self.cross_entropy_00
        elif criterion == 'torch_cross':
            self._criterion = nn.CrossEntropyLoss()


        else:
            raise Exception("Unknown loss-function!")
            
        if (optimizer == 'sgd'):
             
            momentum = 0.2
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            if ('momentum' in kwargs.keys()):
                momentum = kwargs['momentum']
            self._optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum);
        elif (optimizer == 'adam'):
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            self._optimizer   = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False) 
             
        else:
            raise Exception("Unknown optimizer!")
            
 
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

class fully_connect_modul_300(Layer_06):
    def __init__(self, Size, device = None, L1 = 0., L2 = 0., show=0):
        super(fully_connect_modul_300, self).__init__()
        self.show = show
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
             
        self.L1=L1
        self.L2=L2
        self.Size = Size
        self.regularizer = Regularizer(L1, L2)
        self.LeakyRelu = nn.LeakyReLU(0.05) 
        self.DropOut = nn.Dropout(0.1)
        self.BNorm = nn.BatchNorm1d(256)
        self.Lin1 = nn.Linear(self.Size[0], self.Size[1], bias = True)
        self.Lin2 = nn.Linear(self.Size[1], self.Size[2], bias = True)
        self.Lin3 = nn.Linear(self.Size[2], self.Size[3], bias = True)
        self.Lin4 = nn.Linear(self.Size[3], self.Size[4], bias = True)
       
        #########################
        self.to(self.device)
        self.reset_parameters()
 
    def forward(self, vect_00):
        
        if self.show:
            print('vect_00',vect_00.shape)
            
        vect_01=self.Lin1(vect_00)        
        vect_01=self.DropOut(vect_01)
        vect_01=self.LeakyRelu(vect_01)
        if self.show:
            print('vect_01',vect_01.shape) 
         
        vect_02=self.Lin2(vect_01)
        vect_02=self.DropOut(vect_02)
        vect_02=self.LeakyRelu(vect_02)
        if self.show:
            print('vect_02',vect_02.shape)   
            
        vect_03=self.Lin3(vect_02) 
        vect_03=self.LeakyRelu(vect_03)
        if self.show:
            print('vect_03',vect_03.shape)
            
        vect_04=self.Lin4(vect_03)  
        if self.show: 
            print('vect_04',vect_04.shape)            
            
        return vect_04
    
class fully_connect_modul_301(Layer_06):
    def __init__(self, Size, device = None, L1 = 0., L2 = 0., show=0):
        super(fully_connect_modul_301, self).__init__()
        self.show = show
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
             
        self.L1=L1
        self.L2=L2
        self.Size = Size
        self.regularizer = Regularizer(L1, L2)
        self.LeakyRelu = nn.LeakyReLU(0.05) 
        self.DropOut = nn.Dropout(0.1)
        self.BNorm = nn.BatchNorm1d(256)
        self.Lin5 = nn.Linear(self.Size[0], self.Size[1], bias = True)
       
        #########################
        self.to(self.device)
        self.reset_parameters()
 
    def forward(self, vect_00):
        
        if self.show:
            print('vect_00',vect_00.shape)
            
        vect_01=self.Lin5(vect_00)        
        vect_01=self.LeakyRelu(vect_01)
        if self.show:
            print('vect_01',vect_01.shape)             
            
        return vect_01
    
class TL_Faces_00(Layer_06):
    def __init__(self, imageSize, L1 = 0., L2 = 0., device = None, margin_=1.0, show=0 ):
        super(TL_Faces_00, self).__init__((imageSize))    
         
        self.imageSize = imageSize
        self.regularizer = Regularizer(L1, L2)
        self.show=show
        self.L1=L1
        self.L2=L2 
        self.criterion_tml = nn.TripletMarginLoss(margin=margin_, p=2)
        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #######################
        self.fully_connect_modul_TL1 = fully_connect_modul_300([225, 200, 170, 140, 128], device, L1, L2, show)
        #self.fully_connect_modul_TL1 = fully_connect_modul_300([497, 350, 256, 180, 128], device, L1, L2, show)
        #self.fully_connect_modul_TL2 = fully_connect_modul_301([128, 291], device, L1, L2, show)
        ####################### 
        
    def forward(self, scatch):
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
        
           
        x_input = scatch
        
        _t_input = []
        _x_input = []
        for x in (x_input,x_input):
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        scatch = self._contiguous(_x_input[0])
      
         
         
        ##############
        
            
        
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))

        vect_00 = self.fully_connect_modul_TL1(scatch)  
        #vect_01 = self.fully_connect_modul_TL2(vect_00) 
        ################################# 
         
        ######################################
        x = vect_00
        x = self._contiguous(x)

        ###################    

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return x 
     
    def _get_regularizer(self):
        return self.regularizer
###################################################################################################
    def loss_batch_01(self,dsrmn_model, xb, yb,   opt=None):
        pred = self(xb)  

        if isinstance(pred, tuple):
            pred0 = pred[0]
            del pred
        else:
            pred0 = pred
            
        loss=0
        
        loss_mse = self._criterion(pred0, yb)
         
        
        loss +=1.1*loss_mse 


        _, predicted = torch.max(pred0, dim = -1)
        _, ind_target = torch.max(yb, dim = -1)
        correct = (predicted == ind_target).sum().item()
        acc = correct / len(yb)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if (opt is not None)  :
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss

        return loss_item, 1, acc

    
################################################################
    def fit_dataloader_CLASS(self, dscrm_model, loader, epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


         
            
            
        history = History()
        self.count=0
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            ########################################3
             
            
            ### train mode ###
            print("[", end='')
            losses = []
            nums = []
            accs = []
            for s in loader:
                
 
                
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['label'].numpy()).to(self.device)   
                                        )
                
                 
                
                 
                
                images_Anchor=train_ds.tensors[0] 
                 
                class_=train_ds.tensors[1]
            
                #print('images_Anchor', images_Anchor)
                 

                
                
                

                losses_, nums_, acc_   =   self.loss_batch_01(dscrm_model, \
                                                   images_Anchor,\
                                                    class_,  self._optimizer)                                                                                                       


                losses.append(losses_)
                nums.append(nums_ )
                accs.append(acc_)
                 
                
            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            acc = np.sum(np.multiply(accs, nums)) / sum_nums
            ######################################
             
            ### test mode ###
            if validation_loader is not None:
                 


                self.eval()
                
                 
                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))
                
                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['label'].numpy()).to(self.device)  
                                        )
                

                    images_Anchor=val_ds.tensors[0] 

                    class_=val_ds.tensors[1]
                     
                
                 
                
                     
                
                

  
                      
                    
                                                                                                                         

                    losses_, nums_   =  \
                    self.loss_batch_01( dscrm_model,\
                           (images_Anchor ,images_Anchor ),\
                           class_, self._optimizer)      

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
                print(' - Acc: {:.6f}'.format(acc), end='')
                
            print("")
            
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
    ##################################################
    def loss_batch_02(self,dsrmn_model,  x,   opt=None):

        pred_ancor_0  = self.forward(x[0]) 
         
        pred_positive_0  = self.forward(x[1])
         
        pred_neg_0 = self.forward(x[2])
        loss_0 = self.criterion_tml(pred_ancor_0,pred_positive_0,pred_neg_0)
        loss_1 = self._criterion(pred_ancor_0,pred_positive_0)
            
            
            
        if 0:     
            print(loss_0,loss_1) 
            print('----------') 
             
         
        loss =1.0*loss_0+1.3*loss_1 
        

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if (opt is not None) :
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss
        
        return loss_item, 1#, acc

    
################################################################
    def fit_dataloader_TL(self, dscrm_model,loader,   epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


         
            
            
        history = History()
        self.count=0
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
                
 
                
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['Positive'].numpy()).to(self.device) ,
                                        torch.FloatTensor(s['Negative'].numpy()).to(self.device) 
                                        
                                        )
                
                 
                
                 
                
                images_Anchor=train_ds.tensors[0] 
                 
                images_Positive=train_ds.tensors[1]
                images_Negative=train_ds.tensors[2]

                
                
                

                losses_, nums_   =   self.loss_batch_02(dscrm_model, \
                                                   (images_Anchor ,images_Positive,images_Negative ),\
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
                
                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['Positive'].numpy()).to(self.device) ,
                                        torch.FloatTensor(s['Negative'].numpy()).to(self.device) 
                                        
                                        )
                

                    images_Anchor=val_ds.tensors[0] 

                    images_Positive=val_ds.tensors[1]
                    images_Negative=val_ds.tensors[2]
                
                 
                
                     
                
                

  
                      
                    
                                                                                                                         

                    losses_, nums_   =  \
                    self.loss_batch_00( dscrm_model,\
                           (images_Anchor ,images_Positive,images_Negative ),\
                            self._optimizer)      

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
######################################  
