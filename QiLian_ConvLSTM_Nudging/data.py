# data.py
import os
import pickle
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from configs import (smc_path, dem_path, shp_path,
                     input_dir, plot_dir, start_date, num_days,
                     load_dem, load_data, create_mask,
                     normalize_data,inverse_normalize,generator_config)

font = {'family' : 'Times New Roman',
        'size'   : 14}
plt.rc('font', **font)

# GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow is using {len(physical_devices)} GPU(s): {[d.name for d in physical_devices]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU instead.")

np.random.seed(20)
tf.random.set_seed(20)

print("load data...")
dem = load_dem(dem_path)
smc = load_data(smc_path, num_days)

print("Creating mask...")
mask = create_mask(dem, shp_path)

# Visualizing the mask
def visualize_mask(mask, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray')
    plt.title('Visualizing the mask')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f" The mask has been saved to {save_path}")
    else:
        plt.show()

print("Visualizing the mask...")
visualize_mask(mask, save_path=os.path.join(plot_dir, 'mask.png'))

# Divide the data set (2017-2019 is the training set, 2020 is the validation set, and 2021 is the test set)
start_train_idx = (datetime.date(2017, 1, 1) - start_date).days
end_train_idx = (datetime.date(2020, 5, 31) - start_date).days
start_val_idx = (datetime.date(2020, 6, 1) - start_date).days
end_val_idx = (datetime.date(2021, 5, 31) - start_date).days
start_test_idx = (datetime.date(2021, 6, 1) - start_date).days
end_test_idx = (datetime.date(2021, 12, 31) - start_date).days

# Training set, validation set, test set
X_train, y_train = smc[start_train_idx:end_train_idx + 1], smc[start_train_idx:end_train_idx + 1]
X_val, y_val = smc[start_val_idx:end_val_idx + 1], smc[start_val_idx:end_val_idx + 1]
X_test, y_test = smc[start_test_idx:end_test_idx + 1], smc[start_test_idx:end_test_idx + 1]

# Add time step
time_steps = generator_config['time_steps']
X_train_seq, y_train_seq = [], []
X_val_seq, y_val_seq = [], []
X_test_seq, y_test_seq = [], []

for i in range(time_steps, X_train.shape[0]):
    sm_seq = X_train[i - time_steps:i]
    features = np.stack(sm_seq, axis=0)  # (time_steps, H, W)
    features = features[..., np.newaxis]  # (time_steps, height, width, 1)
    X_train_seq.append(features)
    y_train_seq.append(smc[start_train_idx + i])

for i in range(time_steps, X_val.shape[0]):
    sm_seq = X_val[i - time_steps:i]
    features = np.stack(sm_seq, axis=0)  # (time_steps, H, W)
    features = features[..., np.newaxis]  # (time_steps, height, width, 1)
    X_val_seq.append(features)
    y_val_seq.append(smc[start_val_idx + i])

for i in range(time_steps, X_test.shape[0]):
    sm_seq = X_test[i - time_steps:i]
    features = np.stack(sm_seq, axis=0)  # (time_steps, H, W)
    features = features[..., np.newaxis]  # (time_steps, height, width, 1)
    X_test_seq.append(features)
    y_test_seq.append(smc[start_test_idx + i])

# Convert to NumPy array
X_train_seq = np.array(X_train_seq) # (samples, time_steps, height, width, 1)
y_train_seq = np.array(y_train_seq) # (samples, height, width)
X_val_seq = np.array(X_val_seq)
y_val_seq = np.array(y_val_seq)
X_test_seq = np.array(X_test_seq)
y_test_seq = np.array(y_test_seq)

# Normalize data
arrays_to_normalize = (
    X_train_seq, X_val_seq, X_test_seq,
    y_train_seq, y_val_seq, y_test_seq,
    dem
)
scaled_arrays, scalers = normalize_data(*arrays_to_normalize)

# scaled_arrays
X_train_scaled = scaled_arrays[0]
X_val_scaled = scaled_arrays[1]
X_test_scaled = scaled_arrays[2]
y_train_scaled = scaled_arrays[3]
y_val_scaled = scaled_arrays[4]
y_test_scaled = scaled_arrays[5]
dem_scaled = scaled_arrays[6]

# Denormalized true value
y_test_inverse = inverse_normalize(scalers[5], y_test_scaled)
np.save(os.path.join(input_dir, 'y_test_inverse.npy'), y_test_inverse)

# save data
np.save(os.path.join(input_dir, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(input_dir, 'X_val.npy'), X_val_scaled)
np.save(os.path.join(input_dir, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(input_dir, 'y_train.npy'), y_train_scaled)
np.save(os.path.join(input_dir, 'y_val.npy'), y_val_scaled)
np.save(os.path.join(input_dir, 'y_test.npy'), y_test_scaled)

# save scalers
with open(os.path.join(input_dir, 'scalers.pkl'), 'wb') as f:
    pickle.dump({
        'scaler_X_train': scalers[0],
        'scaler_X_val': scalers[1],
        'scaler_X_test': scalers[2],
        'scaler_y_train': scalers[3],
        'scaler_y_val': scalers[4],
        'scaler_y_test': scalers[5],
        'scaler_dem': scalers[6]
    }, f)



# DataGenerator
class DataGenerator(Sequence):
    def __init__(self, X_path, y_path, batch_size, time_steps, len_out, shuffle=True):
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.len_out = len_out
        self.shuffle = shuffle
        self.indexes = np.arange(self.X.shape[0])
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X_batch = self.X[batch_indexes]
        y_batch = self.y[batch_indexes]

        return X_batch, y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)
