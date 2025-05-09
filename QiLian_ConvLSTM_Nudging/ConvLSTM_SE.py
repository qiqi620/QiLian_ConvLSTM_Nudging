# ConvLSTM_SE.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from configs import (input_dir, model_dir,plot_dir, len_out, generator_config,dropout_rate,kernel_regularizer,
                     filters,kernel_size,learning_rate)
from data import DataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Input, Dense, Multiply, GlobalAveragePooling2D, ConvLSTM2D)

np.random.seed(20)
tf.random.set_seed(20)

# save
model_path = os.path.join(model_dir, 'ConvLSTM_SE_model.h5')
# training history path
training_history_path = os.path.join(model_dir, 'ConvLSTM_SE_history.pkl')
loss_plot_path = os.path.join(plot_dir,'ConvLSTM_SE_loss.png')

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

sample_X, sample_y = train_generator[0]
time_steps = sample_X.shape[1]     # 7
H = sample_X.shape[2]
W = sample_X.shape[3]
num_channels = sample_X.shape[4]
output_dim = sample_y.shape[1:]    # (H, W)

# ==============================
# model
# ==============================
def convlstm_se_att(time_steps, H, W, num_channels, len_out):
    inputs = Input(shape=(time_steps, H, W, num_channels))  # (batch_size,time_steps, height, width, channels)

    # encoder
    outputs = ConvLSTM2D(filters,
                         kernel_size,
                         padding='same',
                         kernel_initializer='he_normal',
                         return_state=False,
                         activation='elu',
                         recurrent_initializer='orthogonal',
                         return_sequences=True,
                         dropout=dropout_rate,
                         kernel_regularizer=kernel_regularizer
                         )(inputs)  # (batch_size, time_steps, H, W, 8)

    outputs, h, c = ConvLSTM2D(filters, kernel_size,
                               padding='same',
                               kernel_initializer='he_normal',
                               activation='elu',
                               recurrent_initializer='orthogonal',
                               return_sequences=True,
                               dropout=dropout_rate,
                               return_state=True,
                               kernel_regularizer=kernel_regularizer
                               )(outputs)

    # Decoder
    outs = []
    for i in range(len_out):
        # Dense layer followed by dimension expansion
        x = tf.expand_dims(Dense(1)(h), axis=1)

        # SE-block
        x_att = GlobalAveragePooling2D()(x[:, 0])  # eq.A1 (batch_size, 1)
        beta = Dense(x.shape[-1])(x_att)  # (batch_size, 1) # (None, 1)
        x_att = Dense(1)(Multiply()([x, beta]))  # eq.A3 #(None,1,79,207,1)

        # give weight for (f(x), x, att(x))
        x = tf.concat([x, x_att], axis=-1)  # f(x)+x, residual block
        x = Dense(1)(x)  # activate  #(None,1,79,207,1)

        outputs, h, c = ConvLSTM2D(filters,
                                     kernel_size,
                                     padding='same',
                                     kernel_initializer='he_normal',
                                     activation='elu',
                                     recurrent_initializer='orthogonal',
                                     return_sequences=True,
                                     dropout=dropout_rate,
                                     return_state=True,
                                     kernel_regularizer=kernel_regularizer
                                     )(x, initial_state=[h, c])  #(None,1,79,207,8),#(None,79,207,8),#(None,79,207,8)
        outs.append(outputs) #(None,1,79,207,8)
    outputs = tf.concat(outs, axis=1)  #(None,1,79,207,1)
    out = Dense(1)(outputs)  # Final output layer
    out = tf.squeeze(out, axis=[1, -1])  # (batch_size, H, W)

    # Define and compile the model
    model = Model([inputs],out, name='ConvLSTM_SE_model')
    model.compile(Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

try:
    model = convlstm_se_att(time_steps, H, W, num_channels, len_out)    # 选择训练模型
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

K.clear_session()

try:
    with open(training_history_path, 'wb') as f:
        pickle.dump(history.history, f)
except Exception as e:
    print(f"An error occurred while saving training history: {e}")