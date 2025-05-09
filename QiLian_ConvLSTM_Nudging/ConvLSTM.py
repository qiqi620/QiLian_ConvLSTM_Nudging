# convlatm.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from configs import (input_dir, model_dir,plot_dir, len_out, generator_config,dropout_rate,
                     filters,kernel_size,learning_rate)
from data import DataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Input, Dense, ConvLSTM2D)

np.random.seed(20)
tf.random.set_seed(20)

# save
model_path = os.path.join(model_dir, 'ConvLSTM_model.h5')
# training history path
training_history_path = os.path.join(model_dir, 'ConvLSTM_history.pkl')
loss_plot_path = os.path.join(plot_dir,'ConvLSTM_loss.png')

train_generator = DataGenerator(
    X_path=os.path.join(input_dir, 'X_train.npy'),
    y_path=os.path.join(input_dir, 'y_train.npy'),
    batch_size=generator_config['batch_size'],
    time_steps=generator_config['time_steps'],
    len_out=generator_config['len_out'],
    shuffle=True
)

val_generator = DataGenerator(
    X_path=os.path.join(input_dir, 'X_val.npy'),
    y_path=os.path.join(input_dir, 'y_val.npy'),
    batch_size=generator_config['batch_size'],
    time_steps=generator_config['time_steps'],
    len_out=generator_config['len_out'],
    shuffle=False
)

# Get data shape information
# In order to build the model, you need to know the shape of the input data, which is obtained from the generator here
sample_X, sample_y = train_generator[0]
time_steps = sample_X.shape[1]     # 7
H = sample_X.shape[2]
W = sample_X.shape[3]
num_channels = sample_X.shape[4]
output_dim = sample_y.shape[1:]    # (H, W)

# ==============================
# model
# ==============================
def convlstm(time_steps, H, W, num_channels, len_out):
    # (None, time_steps, H, W, num_channels)
    inputs = Input(shape=(time_steps, H, W, num_channels))

    # Encoder
    outputs = ConvLSTM2D(filters=filters,
                         kernel_size=kernel_size,
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='elu',
                         recurrent_initializer='orthogonal',
                         return_sequences=True,
                         dropout=dropout_rate,
                         )(inputs)

    outputs, h, c = ConvLSTM2D(filters=filters,
                                kernel_size=kernel_size,
                                padding='same',
                                kernel_initializer='he_normal',
                                activation='elu',
                                recurrent_initializer='orthogonal',
                                return_sequences=True,
                                return_state=True,
                                dropout=dropout_rate,
                                )(outputs)

    # Decoder
    outs = []
    for i in range(len_out):
        x = tf.expand_dims(Dense(1)(h), axis=1)

        outputs, h, c = ConvLSTM2D(filters=filters,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     kernel_initializer='he_normal',
                                     activation='elu',
                                     recurrent_initializer='orthogonal',
                                     return_sequences=True,
                                     dropout=dropout_rate,
                                     return_state=True,
                                    )(x, initial_state=[h, c])
        outs.append(outputs)

    outputs = tf.concat(outs, axis=1)
    out = Dense(1)(outputs)
    out = tf.squeeze(out, axis=[1, -1])  # (batch_size, H, W)

    model = Model([inputs], out, name='ConvLSTM_model')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

try:
    model = convlstm(time_steps, H, W, num_channels, len_out)
    model.summary()
except Exception as e:
    print(f"An error occurred while building the model: {e}")
    exit(1)

# ==============================
# Model compilation and training
# ==============================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=generator_config['patience'],
    restore_best_weights=True,
    verbose=1
)
checkpoint = ModelCheckpoint(
    model_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

print("train ...")
try:
    history = model.fit(
        train_generator,
        epochs=generator_config['epochs'],
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )
except Exception as e:
    print(f"An error occurred while training the model: {e}")
    exit(1)

try:
    with open(training_history_path, 'wb') as f:
        pickle.dump(history.history, f)
except Exception as e:
    print(f"An error occurred while saving training history: {e}")
