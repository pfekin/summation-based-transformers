import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10, cifar100

def data_load(name_str): 
    if name_str.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = eval('cifar' + str(100) + '.load_data()')
        x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
        y_train, y_test = y_train.flatten(), y_test.flatten()
        return x_train, y_train, x_test, y_test, 100, (32, 32, 3) # 100 classes, 32*32*3 numpy array
        
        
    elif name_str.lower() == 'lfw':
        lfw_people = fetch_lfw_people(min_faces_per_person=14, slice_ = (slice(61,189) ,slice(61,189)),
                          resize=0.5, color = True) 
        x,y = lfw_people.images, lfw_people.target 

        num_classes = lfw_people.target_names.shape[0]
        y = np.asarray(y, dtype=np.int32)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) # split train, test
        # number of classes (person id) varies and is invariably proportional to min_faces_per_person 
        return x_train, y_train, x_test, y_test, num_classes, (64, 64, 3) # 64*64*3 numpy array with resize = 0.5
    
    elif name_str.lower() == 'eurosat':    
        train_ds = tfds.load('eurosat/rgb', split='train[:80%]')
        x_train, y_train = [], []
        for el in train_ds:  # only take first element of dataset
            x_train.append(el['image'].numpy())
            y_train.append(el['label'].numpy())

        test_ds = tfds.load('eurosat/rgb',  split='train[80%:]')
        x_test, y_test = [], []
        for el in test_ds:  # only take first element of dataset
            x_test.append(el['image'].numpy())
            y_test.append(el['label'].numpy())
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), 10, (64, 64, 3) # 64*64*3 numpy array 
        
    if name_str.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = eval('cifar' + str(10) + '.load_data()')
        x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
        y_train, y_test = y_train.flatten(), y_test.flatten()
        return x_train, y_train, x_test, y_test, 10, (32, 32, 3) # 100 classes, 32*32*3 numpy array
            
    elif name_str.lower() == 'oxford_flowers102':    
        train_ds = tfds.load('oxford_flowers102', split=tfds.Split.TRAIN)
        x_train, y_train = [], []
        for el in train_ds:  # only take first element of dataset
            image = tf.image.resize(el['image'], size=(64, 64))
            image = tf.divide(image, 255.0)
            x_train.append(image.numpy())
            y_train.append(el['label'].numpy())

        test_ds = tfds.load('oxford_flowers102', split=tfds.Split.VALIDATION)
        x_test, y_test = [], []
        for el in test_ds:  # only take first element of dataset
            image = tf.image.resize(el['image'], size=(64, 64))
            image = tf.divide(image, 255.0)
            x_test.append(image.numpy())
            y_test.append(el['label'].numpy())
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), 102, (64, 64, 3) # 64*64*3 numpy array 
    
    else: 
        raise NameError('name_str does not match a dataset name')
