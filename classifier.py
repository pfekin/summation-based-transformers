import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence, text
from datasets import load_dataset
import math
import numpy as np

# Configuration
MAX_LEN = 256
BATCH_SIZE = 64
MAX_FEATURES = 10000
EMBED_DIM = 256
NUM_HEADS = 4
EPOCHS = 5

"""
# Load IMDB data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
"""

def load_ag_news_hf():
    dataset = load_dataset("ag_news")
    train_data = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_data = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]
    
    # Tokenize and pad
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(train_data)
    x_train = tokenizer.texts_to_sequences(train_data)
    x_test = tokenizer.texts_to_sequences(test_data)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    
    return (x_train, np.array(train_labels)), (x_test, np.array(test_labels))

# Load AG News data
(x_train, y_train), (x_test, y_test) = load_ag_news_hf()

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

# Position Encoding
class AdaptivePositionEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]  # (max_len, 1)
        div_term = tf.exp(
            tf.range(0, self.d_model, 2, dtype=tf.float32) *
            (-math.log(10000.0) / self.d_model
        ))  # (d_model//2,)

        # Compute sinusoidal embeddings
        pe = tf.zeros((self.max_len, self.d_model))  # (max_len, d_model)
        sin_vals = tf.sin(position * div_term)  # (max_len, d_model//2)
        cos_vals = tf.cos(position * div_term)  # (max_len, d_model//2)

        # Interleave sin and cos values
        pe = tf.reshape(
            tf.stack([sin_vals, cos_vals], axis=2),
            [self.max_len, self.d_model]
        )
        self.pe = tf.Variable(pe[tf.newaxis, :, :], trainable=True)

    def call(self, x):
        return self.pe[:, :tf.shape(x)[1], :] 

# Aggregation
class Aggregation(layers.Layer):
    def __init__(self, d_model, projection=True, noise=0.0):
        super().__init__()
        self.d_model = d_model
        self.projection = projection
        self.noise = noise      
    
    def build(self, input_shape):
        self.pos_encoder = AdaptivePositionEncoding(input_shape[1], self.d_model)
        if self.projection:
            self.proj = layers.Dense(self.d_model, activation='linear', use_bias=False)
        self.add_noise = layers.GaussianNoise(self.noise, seed=None)
        
    def call(self, x):
        x_high = x * self.pos_encoder(x)
        if self.projection:
            x_high = self.proj(x_high)  # (batch, seq_len, d_model)
      
        x_sum = tf.reduce_sum(x_high, axis=1, keepdims=True)
            
        if self.trainable and self.noise > 0.0:
            x_sum = self.add_noise(x_sum)
        return x_sum + x  # (batch, seq_len, d_model)
        
# Multi-head attention
class DotProductAttention(layers.Layer):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.pos_encoder = AdaptivePositionEncoding(input_shape[1], self.d_model)
        self.mha = layers.MultiHeadAttention(self.num_heads, self.d_model)
        super().build(input_shape)
        
    def call(self, x):
        x = x + self.pos_encoder(x)
        return self.mha(x, x)
    
    def compute_output_shape(self, input_shape):
        return input_shape

# --- Model Builder ---
def build_model(attention_type="aggregate", **kwargs):
    inputs = layers.Input(shape=(MAX_LEN,))
    x = layers.Embedding(MAX_FEATURES, EMBED_DIM)(inputs)
 
    if attention_type == "aggregate":
        x = Aggregation(EMBED_DIM)(x)
    else:
        x = DotProductAttention(EMBED_DIM)(x)
        
    x = layers.GlobalAvgPool1D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs)

# --- Training Loop ---
attention_types = {
    "aggregate": {"projection": False, "noise": 0.1},
    "dot_product": {"num_heads": NUM_HEADS}
}

results = {}
for name, params in attention_types.items():
    print(f"\n=== Training {name} attention ===")
    model = build_model(attention_type=name, **params)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    results[name] = {
        "val_accuracy": max(history.history['val_accuracy']),
        "params": model.count_params()
    }

# --- Results Summary ---
print("\n=== Final Results ===")
for name, res in results.items():
    print(f"{name:<12} | Val Acc: {res['val_accuracy']:.4f} | Params: {res['params']}")