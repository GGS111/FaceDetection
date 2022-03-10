# -*- coding: utf-8 -*-
"""
@author: user
"""

import gc
import psutil
import torch
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

def mem_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

def ai_2(img):
    if isinstance(img, torch.Tensor):
        if hasattr(img, 'device') and img.device is not None and img.device.type == 'cuda':
            img = img.cpu().detach().numpy()
        else:
            img = img.detach().numpy()
            
    mmin = np.min(img)
    mmax = np.max(img)
    plt.imshow(255-255*(img-mmin)/(mmax-mmin), cmap = 'Greys')
    plt.show()
    return
