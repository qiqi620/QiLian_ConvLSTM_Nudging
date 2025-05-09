# QiLian_ConvLSTM_Nudging

# A hybrid ConvLSTM–Nudging model for predicting surface soil moisture in the Qilian Mountain Area

## Requirements

The code has been tested running under Python 3.10, with the following packages and their dependencies installed:
```
gdal==3.10.0
pyproj==3.7.0
rasterio==1.4.3
scikit-learn==1.6.0
numpy==1.26.0
tensorflow==2.10.0
```

## Usage

Firstly, run 'data.py' to perform data preprocessing and data set division.

Secondly, make model predictions:

- Run 'ConvLSTM.py' and 'ConvLSTM_SE.py' to train two models.
  
- Run  'Short_prediction.py' and 'Long_prediction.py' to make long-term and short-term forecasts.
  
Finally,run 'data_nudging.py' to perform nudging correction.

## Code reference

https://github.com/leelew/HybridHydro/LICENSE

https://github.com/surajp92/LSTM_Nudging

## Dataset
The Daily 0.05°×0.05° Land Surface Soil Moisture Dataset and digital elevation model (DEM) data of Qilian Mountain Area used in this repository can be downloaded from
 (http://data.tpdc.ac.cn) and (http://www.ncdc.ac.cn), both of which are publicly available.

## Citation
NOTES: The paper is not accepted yet. 
In case you use QiLian_ConvLSTM_Nudging in your research or work, please cite this GitHub codes:
https://github.com/qiqi620/QiLian_ConvLSTM_Nudging

Copyright (c) 2025, Qian Xiao
