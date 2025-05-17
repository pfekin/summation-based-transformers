import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import tensorflow_datasets as tfds
from datasets import load_dataset
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# Configuration
SEQ_LENGTH = 256
BATCH_SIZE = 64
VOCAB_SIZE = 10000
EMBED_DIM = 256
EPOCHS = 30

# Load and preparing IMDB
def prepare_imdb(split, tokenizer):
    ds = tfds.load('imdb_reviews', split=split)  # Using IMDB reviews dataset
    tokenizer.adapt(ds.map(lambda x: x['text']))

    def process(example):
        tokens = tokenizer(example['text'])
        return tokens[:-1], tokens[1:]

    return ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)

def prepare_wikitext(split, tokenizer):
    # Load split from Hugging Face
    #raw_dataset = load_dataset("wikitext", "wikitext-103-v1", split=split) #+100M words, won't run on Colab
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)# 2M words

    # Extract just the text lines and remove empty lines
    texts = [x['text'] for x in raw_dataset if x['text'].strip() != '']

    # Wrap texts in a tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices(texts)

    # Adapt tokenizer to the data (fit vocabulary)
    tokenizer.adapt(ds)
             
    # Function to tokenize and create input-target pairs
    def process(text):
        tokens = tokenizer(text)
        return tokens[:-1], tokens[1:]
    
    def filter_empty(x, y):
        return tf.shape(x)[0] > 0
        
    return ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)\
             .filter(filter_empty)    
             
tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=SEQ_LENGTH+1,
        standardize='lower_and_strip_punctuation'  # Add normalization
    )
    
#train_ds = prepare_imdb('train', tokenizer).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#val_ds = prepare_imdb('test', tokenizer).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # Using test as validation
train_ds = prepare_wikitext('train', tokenizer).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = prepare_wikitext('validation', tokenizer).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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
        return self.pe[:, :tf.shape(x)[1], :] # + x

# Dot Attention 
class DotProductAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads, d_model//num_heads)

    def call(self, x, mask=None, use_causal_mask=False):
        return self.mha(x, x, attention_mask=mask, use_causal_mask=False)

# Aggregation
class Aggregation(layers.Layer):
    def __init__(self, d_model, projection=True, noise_stddev=0.0):
        super().__init__()
        self.d_model = d_model
        self.projection = projection
        self.proj = layers.Dense(d_model, activation='linear', use_bias=False) if projection is not None else None
        self.noise = layers.GaussianNoise(noise_stddev, seed=None)
        
    def call(self, x, mask=None, training=None):
        x = self.proj(x) if self.projection else x
             
        if mask is not None:
            # Convert mask to float32 and use it to zero out future positions
            mask = tf.cast(mask, tf.float32)
            
            # Aggregate using matrix multiplication (avoids 4D tensor)
            x = tf.einsum('bij,bjf->bif', mask, x)  # (batch, seq_len, d_model*expansion)
        else:
            x = tf.reduce_sum(x, axis=1, keepdims=True)
            
        x = self.noise(x, training=training)
        return x  # (batch, seq_len, d_model)

# Autoregressive Model
class LanguageModel(tf.keras.Model):
    def __init__(self, attention_type):
        super().__init__()
        self.attention_type = attention_type
        self.embed = layers.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_enc = AdaptivePositionEncoding(SEQ_LENGTH, EMBED_DIM)

        if attention_type == "aggregate":
            self.attention = Aggregation(EMBED_DIM, projection=True, noise_stddev=0.1)
        else:
            self.attention = DotProductAttention(EMBED_DIM, num_heads=4)
             
        self.out = layers.Dense(VOCAB_SIZE)

    def call(self, inputs):
        x = self.embed(inputs)
        
        if self.attention_type == "aggregate":
            x = x * self.pos_enc(x)
        else:
            x = x + self.pos_enc(x)
        
        # Create causal mask with dynamic batch size
        batch_size = tf.shape(inputs)[0]
        mask = tf.linalg.band_part(
            tf.ones((SEQ_LENGTH, SEQ_LENGTH)), -1, 0)[tf.newaxis, ...]
        mask = tf.tile(mask, [batch_size, 1, 1])  # Shape (batch_size, SEQ_LENGTH, SEQ_LENGTH)
        
        attn_output = self.attention(x, mask=mask)  
        
        return self.out(attn_output)

# --------------------------
# 5. Training & Evaluation
# --------------------------
def train_model(attention_type):
    model = LanguageModel(attention_type)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4, clipvalue=1.0 ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Track metrics properly
    history = {'val_ppl':[]}

    # Custom callback for perplexity
    class PerplexityCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            ppl = math.exp(logs['val_loss'])
            history['val_ppl'].append(ppl)
            logs['val_ppl'] = ppl  # Also store in standard logs

    history_callback = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[PerplexityCallback()]
    )
    model.summary()
    
    # Combine all metrics
    final_history = {
        'training': history_callback.history,
        'perplexity': history
    }
    return {
        'history': final_history,
        'model': model, 
        "params": model.count_params()
    }

# Main Execution
results = {}
for attn_type in ["aggregate", "dot"]:
    print(f"\n{'='*40}\nTraining {attn_type} attention model\n{'='*40}")
    results[attn_type] = train_model(attn_type)
    print(f" | Params: {results[attn_type]['params']}")
