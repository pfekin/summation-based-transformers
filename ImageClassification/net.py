from dataset_util import data_load
from model_util import cnn, resnet, cct, lr_schedule
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, AdamW, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random_vector_util import RandomIndex, RandomVector, ValidatationCallback
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow import random
import numpy as np
import tensorflow as tf

random.set_seed(1)
np.random.seed(1)

dataset_name = 'cifar100' # cifar100, cifar10, lfw, eurosat, oxford_flowers102
model_type = 'cnn' # cnn, resnet, cct
is_sdr = True # True is Random vector, False defaults to softmax classifier for benchmarking purposes

rnd_index_dim = 200
rnd_index_one_dim = 120
dense_size = 4096
batch_size = 128
epochs = 200
learning_rate = 0.0005
weight_decay = 0.0001
drange = 1.0
#rnd_vector_loss, metric = 'cosine_similarity', 'cosine' 
rnd_vector_loss, metric = 'mean_squared_error', 'euclidean' 

x_train, y_train, x_test, y_test, num_classes, x_shape = data_load(dataset_name)
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)    

callbacks = []

if model_type == 'cnn':
    base_model = cnn(input_shape=x_shape, dense_size=dense_size)
    
    optimizer = Adam(learning_rate=learning_rate)
    drange = 10.0
    
elif model_type == 'resnet':
    base_model = resnet(input_shape=x_shape, depth=2*9+2, dense_size=dense_size)
        
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    callbacks = [lr_reducer, lr_scheduler]
    optimizer = Adam(learning_rate=lr_schedule(0))
    
else:
    base_model = cct(input_shape=x_shape, num_heads=2, projection_dim=256)
    
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    
if is_sdr:
    #random_vectors = RandomVector(num_classes, rnd_index_dim)
    random_vectors = RandomIndex(num_classes, rnd_index_dim, rnd_index_one_dim, drange=drange, seed=1)
    y_train_indexes = random_vectors.label_to_class_vector(y_train) 
    
    x = base_model.output
    predictions = Dense(rnd_index_dim, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    val_callback = ValidatationCallback(x_test, y_test, random_vectors, metric=metric)
    callbacks.append(val_callback)
    
    model.compile(optimizer=optimizer, loss=rnd_vector_loss, metrics=['accuracy']) 
    model.fit(datagen.flow(x_train, y_train_indexes, batch_size=batch_size, shuffle=True), epochs=epochs, 
          callbacks=callbacks)
else: 
    x = base_model.output
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])    
    
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True), epochs=epochs, 
              validation_data=(x_test, y_test), callbacks=callbacks)



