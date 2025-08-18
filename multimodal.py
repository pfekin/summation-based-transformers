import tensorflow_datasets as tfds
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.datasets import fetch_openml
import math

EMBED_DIM = 256

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

# Superposition
class Superposition(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def build(self, input_shape):
        # Check if input_shape[1] is defined. If not, set a default max_len
        max_len = input_shape[1] if input_shape[1] is not None else 300 # Use 300 as default if None
        self.pos_encoder = PositionalEmbedding(max_len, self.d_model)
        self.proj = layers.Dense(self.d_model, activation='relu', use_bias=False)

    def call(self, x, training=None):
        pos = self.pos_encoder(x) + 1.0 # positional encoding bias
        x = x * pos
        x = self.proj(x)

        x = tf.reduce_sum(x, axis=1)

        return x  

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.d_model) 

# Custom layer to process numerical features
class NumericalFeatureProcessor(layers.Layer):
    def __init__(self, d_model, num_features):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.dense_layers = []
        for _ in range(num_features):
            self.dense_layers.append(layers.Dense(d_model, activation="relu"))
            self.dense_layers.append(layers.Dense(d_model, activation="gelu"))

    def call(self, inputs):
        split_features = tf.split(inputs, num_or_size_splits=self.num_features, axis=1)
        processed = []
        layer_idx = 0
        for i in range(self.num_features):
            x = self.dense_layers[layer_idx](split_features[i])
            layer_idx += 1
            x = self.dense_layers[layer_idx](x)
            layer_idx += 1
            processed.append(x)
        return processed #layers.Add()(processed)

# Load dataset
data = tfds.load("civil_comments", split="train")
df = tfds.as_dataframe(data.take(10_000))

# Define features and target
texts = df["text"].values
numerical_features = df[["toxicity", "severe_toxicity", "obscene", "threat"]]
y = df["identity_attack"].astype(np.float32)

# Split data once for all models (maintain alignment)
(X_text_temp, X_text_test,
 X_num_temp, X_num_test,
 y_temp, y_test) = train_test_split(
    texts, numerical_features.values, y, test_size=0.2, random_state=42
)

# Final train/validation split
(X_text_train, X_text_val,
 X_num_train, X_num_val,
 y_train, y_val) = train_test_split(
    X_text_temp, X_num_temp, y_temp, test_size=0.25, random_state=42
)

#
# Train numerical model
#
model_num = Ridge(alpha=1.0)
model_num.fit(X_num_train, y_train)
y_pred_num = model_num.predict(X_num_test)


# Text Vectorization
vectorizer = layers.TextVectorization(
    max_tokens=5000,
    output_sequence_length=300,
    output_mode="int"
)
vectorizer.adapt(X_text_train)  

#
# Train attention multimodal modal
#

text_input = layers.Input(shape=(1,), dtype=tf.string, name="text_input")
num_input = layers.Input(shape=(4,), dtype=tf.float32, name="num_input")

# Text processing
x_text = vectorizer(text_input)
x_text = layers.Embedding(
    input_dim=len(vectorizer.get_vocabulary()) + 1, output_dim=EMBED_DIM)(x_text)  # Output shape: (batch_size, sequence_length, 64)
pos = PositionalEmbedding(300, EMBED_DIM)(x_text)
x_text = x_text + pos

attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x_text, x_text)
x_text = layers.Add()([x_text, attn]) # multi-Head Attention + Residual
x_text = layers.LayerNormalization()(x_text)
x_text = layers.GlobalAveragePooling1D()(x_text)

x_num = NumericalFeatureProcessor(EMBED_DIM, X_num_train.shape[1])(num_input)

concatenated = layers.concatenate([x_text] + x_num)

x = layers.Dense(256, activation="relu")(concatenated)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1)(x)

model = Model(inputs=[text_input, num_input], outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

# Convert data to TensorFlow datasets for training, validation, and prediction
train_dataset = tf.data.Dataset.from_tensor_slices(({'text_input': X_text_train, 'num_input': X_num_train}, y_train)).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices(({'text_input': X_text_val, 'num_input': X_num_val}, y_val)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices(({'text_input': X_text_test, 'num_input': X_num_test})).batch(64)


model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    verbose=1,
    callbacks=callbacks
)
print(model.summary())
y_pred_multi = model.predict(test_dataset)

#
# Train Representational Superposition Multimodal Model
#

text_input = layers.Input(shape=(1,), dtype=tf.string, name="text_input")
num_input = layers.Input(shape=(4,), dtype=tf.float32, name="num_input")

# Text Processing 
x_text = vectorizer(text_input)
x_text = layers.Embedding(
    input_dim=len(vectorizer.get_vocabulary()) + 1, output_dim=EMBED_DIM)(x_text)  # Output shape: (batch_size, sequence_length, EMBED_DIM)
x_text = Superposition(EMBED_DIM)(x_text) # Text categorical embeddings are aggregated

# Numerical Processing Branch using the custom layer
x_num = NumericalFeatureProcessor(EMBED_DIM, X_num_train.shape[1])(num_input)
x_num = layers.Add()(x_num)

# Linear transformation of text latent space + ReLU activation before add
x_text = layers.Dense(EMBED_DIM, activation='relu')(x_text)

# Linear transformation of numerical latent space + ReLU activation before add
x_num = layers.Dense(EMBED_DIM, activation='relu')(x_num)
x_num = layers.Dense(EMBED_DIM, activation='relu')(x_num)

# Text and numerical representational superpositions are summed into a single latent space 
single_latent = layers.Add()([x_text, x_num])

x = layers.Dense(256, activation="relu")(single_latent)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1)(x)

model = Model(inputs=[text_input, num_input], outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# Convert text data to TensorFlow datasets for training and prediction
train_dataset = tf.data.Dataset.from_tensor_slices(({"text_input": X_text_train, "num_input": X_num_train}, y_train)).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices(({"text_input": X_text_val, "num_input": X_num_val}, y_val)).batch(64)
test_dataset_aggr = tf.data.Dataset.from_tensor_slices(({"text_input": X_text_test, "num_input": X_num_test})).batch(64)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    verbose=1,
    callbacks=callbacks
)
print(model.summary())
y_pred_aggr_multi = model.predict(test_dataset_aggr)

# Numerical metrics
print("\nNumerical model (ridge):")
print(f"RMSE: {mean_squared_error(y_test, y_pred_num):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_num):.4f}")
print(f"R²: {r2_score(y_test, y_pred_num):.4f}")

# Multi-modal metrics
print("\nMulti-modal model (Multi-head attention):")
print(f"RMSE: {mean_squared_error(y_test, y_pred_multi):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_multi):.4f}")
print(f"R²: {r2_score(y_test, y_pred_multi):.4f}")

# Representational Superposition multi-modal metrics
print("\Representational Superposition multi-modal model:")
print(f"RMSE: {mean_squared_error(y_test, y_pred_aggr_multi):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_aggr_multi):.4f}")
print(f"R²: {r2_score(y_test, y_pred_aggr_multi):.4f}")
