# -*- coding: utf-8 -*-
"""
@author: user
"""

import onnxruntime as ort
import numpy as np

class Model_onnx:
    def __init__(self, onnx_filename):
        self.ort_sess = ort.InferenceSession(onnx_filename)
        
    def __call__(self, x):
        if not isinstance(x, (np.ndarray)):
            x = np.array(x).astype(np.float32)
        else:
            x = x.astype(np.float32)
            
        input_feed = {'input_data': x}
        outputs = self.ort_sess.run(None, input_feed)
        
        if not isinstance(outputs, (np.ndarray)):
            outputs = np.array(outputs)
            
        if len(outputs.shape) > 2 and outputs.shape[0] == 1:
            outputs = np.squeeze(outputs, axis=0)
            
        return outputs

