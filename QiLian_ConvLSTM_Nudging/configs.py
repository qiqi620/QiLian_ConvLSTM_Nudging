# configs.py
import os
import datetime
import rasterio
import numpy as np
import tensorflow as tf
import geopandas as gpd
from rasterio.features import geometry_mask
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2

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

# input data path
smc_path = 'data/smc/'
dem_path = 'data/dem/STRM_v41_3s_DEM_Qilianshan.tif'
shp_path = 'data/shp/qilian.shp'
utrue_path = 'data/utrue/'
# mmodel prediction # uf_path
cl1_path = 'data/Short_ConvLSTM/'
cla1_path = 'data/Short_ConvLSTM_SE/'
cl2_path = 'data/Long_ConvLSTM/'
cla2_path = 'data/Long_ConvLSTM_SE/'

# processed data path
input_dir = 'input/'
os.makedirs(input_dir, exist_ok=True)

# models saving path
model_dir = 'models/'
os.makedirs(model_dir, exist_ok=True)
# 'scalers.pkl'
scaler_path = os.path.join(input_dir, 'scalers.pkl')

# models prediction saving path
prediction_dir = 'data/'
os.makedirs(prediction_dir, exist_ok=True)

plot_dir = 'plot/'
os.makedirs(plot_dir, exist_ok=True)
fig_dir = 'fig/'
os.makedirs(fig_dir, exist_ok=True)

# Training parameters
time_steps = 14  # Iutput the time step
len_out = 1      # Output time step
batch_size = 16
epochs = 100
patience = 50
filters = 8
kernel_size = (5, 5)
stats_hs = 0
learning_rate = 0.001
dropout_rate = 0.3
kernel_regularizer=l2(1e-4)

# Define date range
start_date = datetime.date(2017, 1, 1)
end_date = datetime.date(2021, 12, 31)
num_days = (end_date - start_date).days + 1

# Data generator related configuration
generator_config = {
    'batch_size': batch_size,
    'time_steps': time_steps,
    'len_out': len_out,
    'shuffle': True,
    'patience': patience,
    'epochs':epochs
}

def load_dem(dem_path):
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan
    return dem

def load_data(data_path, num_days):
    data = []
    start_date = datetime.date(2017, 1, 1)
    for day in range(num_days):
        current_date = start_date + datetime.timedelta(days=day)
        filename = os.path.join(data_path, current_date.strftime('%Y%m%d') + '.tif')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"file not found: {filename}")
        with rasterio.open(filename) as src:
            img = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                img[img == nodata] = np.nan
            data.append(img)
    data = np.array(data)  #  (num_days, height, width)
    return data

def create_mask(dem, shp_path):
    with rasterio.open(dem_path) as src:
        dem_meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)

    gdf = gpd.read_file(shp_path)
    geometries = [feature["geometry"] for feature in gdf.__geo_interface__["features"]]

    mask = geometry_mask(geometries=geometries,
                         transform=transform,
                         invert=True,
                         out_shape=out_shape)
    return mask

def apply_mask(data, mask, fill_value=0):
    if data.ndim == 2:
        #  (height, width)
        masked_data = np.where(mask, data, fill_value)
    elif data.ndim == 3:
        #  (num_days, height, width)
        masked_data = np.where(mask, data, fill_value)
    elif data.ndim == 4:
        #  (time_steps, height, width, channels)
        masked_data = np.where(mask, data, fill_value)
    else:
        raise ValueError("Data dimension is not supported")
    #  NaN and Inf
    masked_data = np.nan_to_num(masked_data, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return masked_data

"""
Normalize multiple arrays
Return the normalized arrays and the corresponding MinMaxScaler
"""
def normalize_data(*arrays):
    scalers = []
    scaled_arrays = []
    for array in arrays:
        scaler = MinMaxScaler()  #  MinMaxScaler
        original_shape = array.shape
        flattened = array.reshape(-1, 1)
        scaled = scaler.fit_transform(flattened).reshape(original_shape)
        scaled_arrays.append(scaled)
        scalers.append(scaler)
    return scaled_arrays, scalers

"""
Inverse normalization
"""
def inverse_normalize(scaler, data):
    original_shape = data.shape
    flattened = data.reshape(-1, 1)
    inverse = scaler.inverse_transform(flattened)
    inverse = inverse.reshape(original_shape)
    return inverse