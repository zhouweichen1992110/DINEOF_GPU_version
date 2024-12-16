# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 10:29:23 2024
DINEOF_Main.py
@author: zhouw
"""

'''
GPU Time: 25.0330 seconds
CPU Time: 126.7612 seconds
============================   GPU is faster at least 5.06 times ===================
'''

from scipy.io import loadmat
import numpy as np
from utils import select_set, init_missing, eof_multi, eof_core, normalize, eof_multi_GPU,denormalize,func_eofszb, eof_core_torch  # Import the EOF analysis function
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import torch
import time  # Add for time measurement

# Load the .mat file
mat_data = loadmat('./TestDataset.mat')

# Extract Dataset variable
Dataset = mat_data['Dataset']

# Predefined parameters
Lon_M = 97  # Number of longitude points
Lat_N = 121  # Number of latitude points
datalength = 131  # Number of timestamps
pixels = 11737  # Total number of pixels (Lat_N * Lon_M)
percentMonteCarlo = 10
nMonteCarlo = 10
initialization = 'column'
stopping = 0.1 # minimum threshold for early stop
rounds = 1000 # maximun number of iterations
Cum_over = 0.85 # EOFs explaining >85% variance

# Extract and reshape latitude and longitude
lon = Dataset[:, 1]  
lat = Dataset[:, 0]  
Lat = np.reshape(lat, (Lat_N, Lon_M))  
Lon = np.reshape(lon, (Lat_N, Lon_M))  

# Create 2D mesh grid from 1D longitude and latitude
Lonn = (Lon[0,:]).T
Latt = Lat[:,0]
Latt = np.flipud(Latt) 
Lon_grid, Lat_grid = np.meshgrid(Lonn, Latt) 

# Extract and reshape data values
Data1 = Dataset[:, 2:]
Data1 = np.reshape(Data1, (Lat_N, Lon_M, datalength))  

# # flipud lat in Data
Data = np.empty_like(Data1)
for t in range(Data1.shape[2]):
    Data[:,:,t] = np.flipud(Data1[:,:,t])

# Verify shapes for debugging
print("Lat shape:", Lat.shape) 
print("Lon shape:", Lon.shape) 
print("Data shape:", Data.shape) 


## Count valid data and coverage
count = 0
Coverage = np.full((Lat_N, Lon_M), np.nan)  
Data_Used = []
Lat_Used = []
Lon_Used = []

# Iterate through each grid cell
for i in range(Lat_N):  
    for j in range(Lon_M): 
        Data_tmp = Data[i, j, :]  
        valid_indices = np.where(~np.isnan(Data_tmp))[0]  
        Coverage[i, j] = len(valid_indices) / len(Data_tmp)  
        
        if Coverage[i, j] == 0:  
            Coverage[i, j] = np.nan
            Lat[i, j] = np.nan
            Lon[i, j] = np.nan
        else:
            count += 1  # Increment valid pixel count
            Data_Used.append(Data_tmp) 
            Lat_Used.append(Lat[i, j])  
            Lon_Used.append(Lon[i, j])  

# Convert lists to numpy arrays for further processing
Data_Used = np.array(Data_Used)
Lat_Used = np.array(Lat_Used)
Lon_Used = np.array(Lon_Used)

#% EOF Analysis, find the aimed EOFs
data = np.copy(Data_Used)

# Replace NaNs with zeros for EOF analysis
data[np.isnan(data)] = 0

#% Perform EOF analysis
PCA, EigenVector, EigenValue, Cum, Mean_Eof = func_eofszb(data)

# Find the number of EOFs explaining >85% variance
indices = np.where(Cum > Cum_over)[0] 
maxeof = indices[0] + 1

# Prepare data for validation
data = np.copy(Data_Used)

# Use the select_set function to randomly select data
data2, testData, testIndex = select_set(data, percentMonteCarlo, method='random')

# Normalize data (row-wise mean and standard deviation)
dataNorm, norm_params = normalize(data2, 'meanrows', 'stdrows')
normMeans = norm_params[0]
normStds = norm_params[1]

# Output results for debugging
print("Number of valid pixels:", count)
print("Coverage shape:", Coverage.shape)
print("Number of EOFs explaining >85% variance:", maxeof)
print("Cumulative variance (first 10 values):", Cum[:10])

#% Initialize validation errors array
valErrors_gpu = np.full((nMonteCarlo, maxeof), np.inf)  # Shape: (nMonteCarlo, maxeof)
valErrors_cpu = np.full((nMonteCarlo, maxeof), np.inf)  # Shape: (nMonteCarlo, maxeof)
maxeof = int(maxeof)

# Initialize missing data again using the best EOF model
dataInit, maskTest = init_missing(dataNorm, initialization)

start_gpu = time.time()
# Monte Carlo Validation Loop
for mc in range(nMonteCarlo):
    # Randomly select the validation set
    dataMC, valData, valIndex = select_set(dataNorm, percentMonteCarlo)

    # Initialize missing values in the selected data
    dataInit, mask = init_missing(dataMC, initialization)

    # Perform EOF estimation on the initialization (you will need to implement or call your EOF estimation method)
    outputs,roundcounts = eof_multi_GPU(dataInit, mask, maxeof, 'original', valIndex, stopping, rounds)

    # Calculate the validation error (Mean Squared Error) for each EOF
    valErrors_gpu[mc, :] = (np.mean((outputs - np.tile(valData, (maxeof, 1))) ** 2, axis=1)).T
    
mean_valErrors = np.mean(valErrors_gpu, axis=0)
best_eof_index_gpu = np.argmin(mean_valErrors) + 1
best_eof_index_gpu = int(best_eof_index_gpu)

# Measure time for GPU-based eof_core_torch
dataFilled_gpu, _, _ = eof_core_torch(dataInit, maskTest, best_eof_index_gpu, stopping, rounds)
end_gpu = time.time()


start_cpu = time.time()
# Monte Carlo Validation Loop
for mc in range(nMonteCarlo):
    # Randomly select the validation set
    dataMC, valData, valIndex = select_set(dataNorm, percentMonteCarlo)

    # Initialize missing values in the selected data
    dataInit, mask = init_missing(dataMC, initialization)

    # Perform EOF estimation on the initialization (you will need to implement or call your EOF estimation method)
    outputs,roundcounts = eof_multi(dataInit, mask, maxeof, 'original', valIndex, stopping, rounds)

    # Calculate the validation error (Mean Squared Error) for each EOF
    valErrors_cpu[mc, :] = (np.mean((outputs - np.tile(valData, (maxeof, 1))) ** 2, axis=1)).T


# # Find the best EOF model (minimizing the validation error)
mean_valErrors = np.mean(valErrors_cpu, axis=0)
best_eof_index_cpu = np.argmin(mean_valErrors) + 1
best_eof_index_cpu = int(best_eof_index_cpu)
# Measure time for CPU-based eof_core
dataFilled_cpu, _, _ = eof_core(dataInit, maskTest, best_eof_index_cpu, stopping, rounds)
end_cpu = time.time()


# Calculate GPU and CPU times
gpu_time = end_gpu - start_gpu
cpu_time = end_cpu - start_cpu
speedup = cpu_time / gpu_time
# Calculate test error (Mean Squared Error)
testError_cpu = np.mean((dataFilled_cpu.flatten()[testIndex] - testData) ** 2)
testError_gpu = np.mean((dataFilled_gpu.flatten()[testIndex] - testData) ** 2)

print("best_eof_index_gpu:", best_eof_index_gpu)
print("best_eof_index_gpu:", best_eof_index_cpu)
print("Test Error (CPU):", testError_cpu)
print("Test Error (GPU):", testError_gpu)
print(f"GPU Time: {gpu_time:.4f} seconds")
print(f"CPU Time: {cpu_time:.4f} seconds")

print(f"GPU is faster at least {speedup:.2f} times")

'''
GPU Time: 25.0330 seconds
CPU Time: 126.7612 seconds
GPU is faster at least 5.06 times
'''