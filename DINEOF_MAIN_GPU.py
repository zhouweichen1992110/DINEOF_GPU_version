# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:29:23 2024
DINEOF_Main.py
@author: zhouw
"""
from scipy.io import loadmat
import numpy as np
from utils import select_set, init_missing, eof_multi, eof_core, normalize, eof_multi_GPU,denormalize,func_eofszb, eof_core_torch  # Import the EOF analysis function
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import torch


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
nMonteCarlo = 1
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
valErrors = np.full((nMonteCarlo, maxeof), np.inf)  # Shape: (nMonteCarlo, maxeof)
maxeof = int(maxeof)

# Monte Carlo Validation Loop
for mc in range(nMonteCarlo):
    # Randomly select the validation set
    dataMC, valData, valIndex = select_set(dataNorm, percentMonteCarlo)

    # Initialize missing values in the selected data
    dataInit, mask = init_missing(dataMC, initialization)

    # Perform EOF estimation on the initialization (you will need to implement or call your EOF estimation method)
    outputs,roundcounts = eof_multi_GPU(dataInit, mask, maxeof, 'original', valIndex, stopping, rounds)

    # Calculate the validation error (Mean Squared Error) for each EOF
    valErrors[mc, :] = (np.mean((outputs - np.tile(valData, (maxeof, 1))) ** 2, axis=1)).T

# # Find the best EOF model (minimizing the validation error)
mean_valErrors = np.mean(valErrors, axis=0)
best_eof_index = np.argmin(mean_valErrors) + 1
best_eof_index = int(best_eof_index)

# Initialize missing data again using the best EOF model
dataInit, maskTest = init_missing(dataNorm, initialization)



# Measure time for GPU-based eof_core_torch

dataFilled_gpu, _, _ = eof_core_torch(dataInit, maskTest, best_eof_index, stopping, rounds)
dataFilled_gpu = denormalize(dataFilled_gpu, 'std', normStds, 'mean', normMeans)

# Calculate test error (Mean Squared Error)
testError_gpu = np.mean((dataFilled_gpu.flatten()[testIndex] - testData) ** 2)


print("Validation errors (mean):", mean_valErrors)
print("Best EOF index:", best_eof_index)
print("Test Error (GPU):", testError_gpu)



#%% Final filling of the dataset
data = np.copy(Data_Used)
if np.any(np.isnan(data)):  # Check if there are any missing values in the dataset
    # Dataset normalization (ignores NaN values)
    dataNorm, norm_params = normalize(data, 'meanrows', 'stdrows')
    normMeans = norm_params[0]
    normStds = norm_params[1]

    # Initialization
    dataInitFinal, maskFinal = init_missing(dataNorm, initialization)

    # Filling missing values using EOF estimation
    dataFilledFinal, _, _ = eof_core_torch(dataInitFinal, maskFinal, best_eof_index, stopping, rounds)

    # De-normalizing the filled dataset
    dataFilledFinal = denormalize(dataFilledFinal, 'std', normStds, 'mean', normMeans)

    # Replace the missing values in the original dataset with the filled values
    dataFilledOutput = np.copy(data)
    dataFilledOutput[np.isnan(data)] = dataFilledFinal[np.isnan(data)]


#%% Plotting the results
for ss in range(0, 131, 25):  # 131 is the timestamps
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 每张图包含两个子图

    # ============ Before DINEOF ============ #
    ax = axes[0]
    m = Basemap(projection='mill', 
                llcrnrlat=Lat_grid.min(), urcrnrlat=Lat_grid.max(),
                llcrnrlon=Lon_grid.min(), urcrnrlon=Lon_grid.max(),
                resolution='i', ax=ax)

    m.drawcoastlines()
    m.fillcontinents(color='lightgrey', lake_color='white')
    m.drawmapboundary(fill_color='white')
    m.drawparallels(np.arange(28, 34, 1), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(121, 126, 1), labels=[0, 0, 0, 1])

    # project lon&lat
    lon_proj, lat_proj = m(Lon_grid, Lat_grid)

    # ============ Before DINEOF ============ # 
    D = Data[:, :, ss]
    cs = m.pcolormesh(lon_proj, lat_proj, D, shading='auto', cmap='jet')
    cs.set_clim(0, 30)  
    cb = m.colorbar(cs, location='bottom', pad=0.4)
    cb.set_label('Data Value (0-30)')
    ax.set_title('Before DINEOF - Time Step {}'.format(ss + 1))

    # ============ After DINEOF ============ #
    ax = axes[1]
    m = Basemap(projection='mill', 
                llcrnrlat=Lat_grid.min(), urcrnrlat=Lat_grid.max(),
                llcrnrlon=Lon_grid.min(), urcrnrlon=Lon_grid.max(),
                resolution='i', ax=ax)

    m.drawcoastlines()
    m.fillcontinents(color='lightgrey', lake_color='white')
    m.drawmapboundary(fill_color='white')
    m.drawparallels(np.arange(28, 34, 1), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(121, 126, 1), labels=[0, 0, 0, 1])

    Data2 = np.full((Lat_N, Lon_M), np.nan)
    for ii in range(Lat_N):
        for jj in range(Lon_M):
            matches = np.where((np.abs(Lon[ii, jj] - Lon_Used) < 1e-6) &
                               (np.abs(Lat[ii, jj] - Lat_Used) < 1e-6))
            if matches[0].size > 0:
                Data2[ii, jj] = dataFilledOutput[matches[0][0], ss]

    cs = m.pcolormesh(lon_proj, lat_proj, Data2, shading='auto', cmap='jet')
    cs.set_clim(0, 30)  
    cb = m.colorbar(cs, location='bottom', pad=0.4)
    cb.set_label('Data Value (0-30)')
    ax.set_title('After DINEOF - Time Step {}'.format(ss + 1))

    # save & plt
    plt.tight_layout()
    plt.savefig(f"./figs_gpu/DINEOF_Comparison_TimeStep_{ss+1}.png")  
    # plt.show()  
    plt.close(fig)  
