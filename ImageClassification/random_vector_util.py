import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from scipy.spatial.distance import cdist
from collections import Counter
import tensorflow as tf
 
class OutputVector():
    def __init__(self, seed):
        np.random.seed(seed)
        
    def label_to_class_vector(self, labels):
        class_vectors = [self.classes[int(label)] for label in labels]
        return np.array(class_vectors, np.float32)
        
    def logit_to_label(self, logits, metric='euclidean'):
        pred = cdist(logits, self.classes, metric=metric)
        pred = np.argmin(pred, axis=1)
        return pred    

class RandomIndex(OutputVector):
    def __init__(self, num_classes, rnd_index_dim=200, rnd_index_one_dim=120, drange=1.0, seed=1):
        super().__init__(seed)
        root = np.zeros(rnd_index_dim, dtype=np.float32) - drange
        root[0 : rnd_index_one_dim] = drange
        self.classes = np.empty((num_classes, rnd_index_dim), dtype=np.float32)
        for i in range(num_classes):
            self.classes[i] = np.random.permutation(root)  

class RandomVector(OutputVector):        
    def __init__(self, num_classes, vector_dim, drange=1.0, seed=1):
        super().__init__(seed)
        root_vector = (np.random.random(vector_dim).astype(np.float32) - 0.5) * 2 * drange
        #root_vector = np.random.random(vector_dim).astype(np.float32)

        self.classes = np.empty((num_classes, vector_dim), dtype=np.float32)
        for i in range(num_classes):
            self.classes[i] = np.random.permutation(root_vector)  
           
class ValidatationCallback(Callback):   
    def __init__(self, x_val, y_val, sdr, metric='euclidean'):
        super().__init__()
        self.sdr = sdr
        self.x_val = x_val
        self.y_val = y_val
        self.metric = metric
        self.best_score = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logits = self.model.predict(self.x_val)
        pred = self.sdr.logit_to_label(logits, metric=self.metric)
        pred = np.mean(pred == self.y_val)
        print("val_accuracy:", pred, flush=True)
        if pred > self.best_score:
            self.best_score = pred
            print("\nBest val_accuracy:", pred, "\n", flush=True)