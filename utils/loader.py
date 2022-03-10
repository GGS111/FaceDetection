from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib as plt

class DatasetForTLFacesFromCSV(Dataset):
    def __init__(self, csv_file, flag):
        
        df = np.loadtxt(csv_file, delimiter = ',', dtype = np.float32, skiprows = 1)
        self.flag = flag
        
        count = 0
        self.full = []
        while count < int(df[-1][-1]):
            list_fold = []
            for val in df:
                if val[-1] == count:
                    list_fold.append(val[1:-1])
            self.full.append(list_fold)
            count += 1
        
        
            
    def __len__(self):
        
        return len(self.full)
    
    def plot_im_2(self, img, img2):

        plt.figure(figsize=(25,10))

        plt.plot(img.ravel() ,'k')
        plt.plot(img2.ravel() ,'r')

        plt.show()  

    
    def __getitem__(self, idx):
        
        random.shuffle(self.full)
        
        anchor_path = np.array(self.full[0])
        negative_path = np.array(self.full[1])
        np.random.shuffle(anchor_path)
        np.random.shuffle(negative_path)
        anchor = anchor_path[0]
        positive = anchor_path[1]
        negative = negative_path[0]
        
        return {'Anchor':anchor,
                'Positive':positive,
                'Negative':negative 
                }
