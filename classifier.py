import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence, text
from datasets import load_dataset
from tensorflow.keras.datasets import reuters
import tensorflow_datasets as tfds
from sklearn.datasets import fetch_20newsgroups
import math
import numpy as np

# Configuration
MAX_LEN = 256
BATCH_SIZE = 64
MAX_FEATURES = 10000
EMBED_DIM = 256
NUM_HEADS = 4
EPOCHS = 20

def load_ag_news_hf(num_words=10000):
    dataset = load_dataset("fancyzhx/ag_news") #  download_mode='force_redownload'

    train_data = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_data = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]
    
    tokenizer = text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_data)
    
    x_train = tokenizer.texts_to_sequences(train_data)
    x_test = tokenizer.texts_to_sequences(test_data)
    
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    
    return (x_train, np.array(train_labels)), (x_test, np.array(test_labels))

def load_20newsgroups(num_words=10000):
    newsgroups_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    newsgroups_test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    
    tokenizer = text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(newsgroups_train.data)
    
    x_train = tokenizer.texts_to_sequences(newsgroups_train.data)
    x_test = tokenizer.texts_to_sequences(newsgroups_test.data)
    
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    
    x_train = np.array(x_train, dtype=np.int32)
    x_test = np.array(x_test, dtype=np.int32)
    y_train = np.array(newsgroups_train.target, dtype=np.int32)
    y_test = np.array(newsgroups_test.target, dtype=np.int32)
    
    return (x_train, y_train), (x_test, y_test)

def load_reuters(num_words=10000):
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)
    
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    return (x_train, np.array(y_train, dtype=np.int32)), (x_test, np.array(y_test, dtype=np.int32))

def load_imbd(num_words=10000):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    
    return (x_train, y_train), (x_test, y_test)

# Select dataset by uncommenting
#(x_train, y_train), (x_test, y_test) = load_ag_news_hf(MAX_FEATURES)
(x_train, y_train), (x_test, y_test) = load_20newsgroups(MAX_FEATURES)
#(x_train, y_train), (x_test, y_test) = load_reuters(MAX_FEATURES)
#(x_train, y_train), (x_test, y_test) = load_imbd(MAX_FEATURES)

# Determine number of classes 
num_classes = len(np.unique(y_train))
num_classes = 1 if num_classes == 2 else num_classes
loss_fn = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
activation = "softmax" if num_classes > 2 else "sigmoid"

print(num_classes, loss_fn, activation)

class PositionalEmbedding(layers.Layer):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_embedding = layers.Embedding(
            input_dim=max_seq_len, 
            output_dim=d_model,
            embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        ) 

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1) 
        positional_encodings = self.pos_embedding(positions) 
        return positional_encodings  

# Aggregation
class Superposition(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def build(self, input_shape):
        self.pos_encoder = PositionalEmbedding(input_shape[1], self.d_model)
        self.proj = layers.Dense(self.d_model, activation="relu", use_bias=False)
        
    def call(self, x, training=None):
        x = x * (self.pos_encoder(x) + 1.0) # positional encoding bias
        x = self.proj(x)
      
        x = tf.reduce_sum(x, axis=1)
            
        return x 
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
# Multi-head attention
class DotProductAttention(layers.Layer):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.pos_encoder = PositionalEmbedding(input_shape[1], self.d_model)

        self.mha = layers.MultiHeadAttention(self.num_heads, self.d_model)
        super().build(input_shape)
        
    def call(self, x):
        x = x + self.pos_encoder(x)
        return self.mha(x, x)
    
    def compute_output_shape(self, input_shape):
        return input_shape

def build_model(attention_type="superposition", **kwargs):
    inputs = layers.Input(shape=(MAX_LEN,))
    x = layers.Embedding(MAX_FEATURES, EMBED_DIM)(inputs)

    if attention_type == "superposition":
        x = Superposition(EMBED_DIM)(x)
    else:
        x = DotProductAttention(EMBED_DIM)(x)
        x = layers.GlobalAvgPool1D()(x)
        
    outputs = layers.Dense(num_classes, activation=activation)(x)
    return Model(inputs=inputs, outputs=outputs)

# Training Loop
attention_types = {
    "superposition": {},
    "dot_product": {"num_heads": NUM_HEADS}
}

results = {}
for name, params in attention_types.items():
    print(f"\nTraining {name} attention")
    model = build_model(attention_type=name, **params)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4, clipvalue=1.0),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    results[name] = {
        "val_accuracy": max(history.history["val_accuracy"]),
        "params": model.count_params()
    }

print("\nFinal Results")
for name, res in results.items():
    print(f"{name:<12} | Val Acc: {res['val_accuracy']:.4f} | Params: {res['params']}")