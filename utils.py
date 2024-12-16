# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:08:37 2024

@author: zhouw
"""

import numpy as np
from scipy.sparse.linalg import svds
import torch



def denormalize(data, *params):
    """
    Reverse normalization or initialization applied to the data.

    Parameters:
        data (ndarray): Input data matrix to be denormalized.
        *params: Pairs of parameter names and values. Supported parameters:
            - 'mean': Mean value or vector used during normalization.
            - 'std': Standard deviation value or vector used during normalization.

    Returns:
        output (ndarray): Denormalized data matrix.
    """
    output = data.copy()  
    rows, columns = output.shape

    if len(params) % 2 != 0:
        raise ValueError("Number of parameters must be even (name-value pairs).")

    for i in range(0, len(params), 2):
        param_name = params[i].lower()
        param_value = params[i + 1]

        if param_name == 'mean':
            if np.isscalar(param_value):  
                output += param_value
            elif param_value.ndim == 1:  
                if param_value.size == rows:  
                    output += param_value[:, np.newaxis]
                elif param_value.size == columns:  
                    output += param_value[np.newaxis, :]
                else:
                    raise ValueError("Mean vector dimensions do not match the data.")
            else:
                raise ValueError("Invalid mean value dimensions.")
        
        elif param_name == 'std':
            if np.isscalar(param_value):  
                output *= param_value
            elif param_value.ndim == 1:  
                if param_value.size == rows:  
                    output *= param_value[:, np.newaxis]
                elif param_value.size == columns:  
                    output *= param_value[np.newaxis, :]
                else:
                    raise ValueError("Std vector dimensions do not match the data.")
            else:
                raise ValueError("Invalid std value dimensions.")

        else:
            print(f"Unrecognized parameter '{params[i]}' --> Ignored.")

    return output



def eof_multi(data, mask, neof, initeof, index, stop_criterion=1, max_iterations=np.inf):
    """
    Perform EOF analysis with multiple EOF configurations and validation.

    Parameters:
        data (ndarray): 2D data matrix with missing values initialized.
        mask (ndarray): Binary matrix of the same shape as `data`.
                        1 indicates missing values, 0 indicates known data.
        neof (int or list): Specifies EOF range:
            - Scalar: Maximum number of EOFs to use.
            - List of 2: [min_eof, max_eof].
            - List of 3: [min_eof, step, max_eof].
        initeof (str): Initialization method:
            - 'previous': Use results of the previous EOF calculation.
            - 'original': Use original data for each calculation.
        index (array-like): Indices of validation set for error evaluation.
        stop_criterion (float, optional): Convergence threshold. Default is 1.
        max_iterations (int, optional): Maximum number of iterations. Default is infinity.

    Returns:
        outputs (ndarray): Results matrix for each EOF configuration.
        round_count (ndarray, optional): Number of iterations for each EOF calculation.
    """
    # Parse EOF range
    if isinstance(neof, int):
        neofmin, neofmax, neofstep = 1, neof, 1
    elif len(neof) == 2:
        neofmin, neofmax, neofstep = neof[0], neof[1], 1
    elif len(neof) == 3:
        neofmin, neofstep, neofmax = neof
    else:
        raise ValueError("Invalid format for `neof`. Use [min], [min, max], or [min, step, max].")

    # Ensure neofmax does not exceed matrix dimensions
    neofmax = min(neofmax, min(data.shape))

    # Initialize outputs and round counts
    outputs = np.full((neofmax, len(index)), np.inf)  
    round_count = np.full(neofmax, np.inf)            

    # EOF calculation based on initialization method
    if initeof.lower() == 'previous':
        for i in range(neofmin, neofmax + 1, neofstep):
            data, eigenvalues, elapsed = eof_core(data, mask, eofs=i, stop_criterion=stop_criterion, max_iterations=max_iterations)
            outputs[i - 1, :] = data.flat[index]  
            round_count[i - 1] = elapsed         
    elif initeof.lower() == 'original':
        for i in range(neofmin, neofmax + 1, neofstep):
            data2, eigenvalues, elapsed = eof_core(data, mask, eofs=i, stop_criterion=stop_criterion, max_iterations=max_iterations)
            outputs[i - 1, :] = data2.flat[index]  
            round_count[i - 1] = elapsed          
    else:
        raise ValueError("Invalid `initeof` method. Use 'previous' or 'original'.")

    if round_count.ndim > 1:
        round_count = round_count.flatten()  

    return outputs, round_count


def eof_multi_GPU(data, mask, neof, initeof, index, stop_criterion=1, max_iterations=np.inf):
    """
    Perform EOF analysis with multiple EOF configurations and validation.

    Parameters:
        data (ndarray): 2D data matrix with missing values initialized.
        mask (ndarray): Binary matrix of the same shape as `data`.
                        1 indicates missing values, 0 indicates known data.
        neof (int or list): Specifies EOF range:
            - Scalar: Maximum number of EOFs to use.
            - List of 2: [min_eof, max_eof].
            - List of 3: [min_eof, step, max_eof].
        initeof (str): Initialization method:
            - 'previous': Use results of the previous EOF calculation.
            - 'original': Use original data for each calculation.
        index (array-like): Indices of validation set for error evaluation.
        stop_criterion (float, optional): Convergence threshold. Default is 1.
        max_iterations (int, optional): Maximum number of iterations. Default is infinity.

    Returns:
        outputs (ndarray): Results matrix for each EOF configuration.
        round_count (ndarray, optional): Number of iterations for each EOF calculation.
    """
    # Parse EOF range
    if isinstance(neof, int):
        neofmin, neofmax, neofstep = 1, neof, 1
    elif len(neof) == 2:
        neofmin, neofmax, neofstep = neof[0], neof[1], 1
    elif len(neof) == 3:
        neofmin, neofstep, neofmax = neof
    else:
        raise ValueError("Invalid format for `neof`. Use [min], [min, max], or [min, step, max].")

    # Ensure neofmax does not exceed matrix dimensions
    neofmax = min(neofmax, min(data.shape))

    # Initialize outputs and round counts
    outputs = np.full((neofmax, len(index)), np.inf)  
    round_count = np.full(neofmax, np.inf)            

    # EOF calculation based on initialization method
    if initeof.lower() == 'previous':
        for i in range(neofmin, neofmax + 1, neofstep):
            data, eigenvalues, elapsed = eof_core_torch(data, mask, eofs=i, stop_criterion=stop_criterion, max_iterations=max_iterations)
            outputs[i - 1, :] = data.flat[index]  
            round_count[i - 1] = elapsed         
    elif initeof.lower() == 'original':
        for i in range(neofmin, neofmax + 1, neofstep):
            data2, eigenvalues, elapsed = eof_core_torch(data, mask, eofs=i, stop_criterion=stop_criterion, max_iterations=max_iterations)
            outputs[i - 1, :] = data2.flat[index]  
            round_count[i - 1] = elapsed          
    else:
        raise ValueError("Invalid `initeof` method. Use 'previous' or 'original'.")

    if round_count.ndim > 1:
        round_count = round_count.flatten()  

    return outputs, round_count


def eof_core(data, mask, eofs, stop_criterion=1, max_iterations=np.inf):
    """
    Perform EOF analysis and estimate missing values.

    Parameters:
        data (ndarray): 2D data matrix with missing values already initialized.
        mask (ndarray): Binary matrix of the same shape as `data`.
                        1 indicates a missing value, 0 indicates known data.
        eofs (int, list, or ndarray): Specifies which EOFs to use:
            - Scalar: Number of consecutive EOFs to use.
            - Vector: Specific EOFs to use.
            - Matrix: Specifies EOFs for each iteration round (padded with zeros for variable EOF counts).
        stop_criterion (float, optional): Minimum change for convergence. Default is 1.
        max_iterations (int, optional): Maximum number of iterations. Default is infinity.

    Returns:
        output (ndarray): Data matrix with missing values estimated.
        eigenvalues (ndarray): Eigenvalues from the EOF analysis.
        varargout (int, optional): Number of iterations performed (if requested).
    """
    # Initialize parameters based on inputs
    if isinstance(eofs, int):
        if eofs > min(data.shape):
            value = np.arange(1, min(data.shape) + 1)  # Use all EOFs
        else:
            value = np.arange(1, eofs + 1)  # Use the first `eofs` EOFs
    else:
        value = np.array(eofs)

    # Ensure EOF selection vector is clean
    value = value[value > 0]  

    output = data.copy()  # Initialize output
    reference = np.zeros_like(data)  

    if value.ndim == 1:  
        round_count = 0
        while True:
            # Calculate SVD
            U, singular_values, Vt = svds(output, k=max(value))
            eigenvalues = np.diag(singular_values)

            # Estimate missing values
            reconstruction = (U[:, value - 1] @ eigenvalues[value - 1][:, value - 1] @ Vt[value - 1, :])
            output = data * (1 - mask) + reconstruction * mask

            # Check convergence
            change = np.sum((reference * mask - output * mask) ** 2)
            if change <= stop_criterion or round_count >= max_iterations:
                break

            reference = output.copy()  
            round_count += 1

        return output, eigenvalues, round_count

    else:  # EOF pruned calculation for variable EOF counts per round
        for round_num in range(min(value.shape[0], int(max_iterations))):
            # Select EOFs for this round
            index = value[round_num, :]
            index = index[index > 0]  # Remove zeros

            # Perform SVD
            U, singular_values, Vt = svds(output, k=max(index))
            eigenvalues = np.diag(singular_values)

            # Update missing value estimates
            reconstruction = (U[:, index - 1] @ eigenvalues[index - 1][:, index - 1] @ Vt[index - 1, :])
            output = data * (1 - mask) + reconstruction * mask

        return output, eigenvalues, round_count





def init_missing(data, method, value=None):
    """
    Initialize missing values in the data matrix.

    Parameters:
        data (ndarray): 2D data matrix with missing values as NaN.
        method (str): Method for initializing missing values. Options include:
            - 'total': Fill with global mean.
            - 'row': Fill with row-wise mean.
            - 'column': Fill with column-wise mean.
            - 'interpolation': Interpolate along rows (temporal dimension).
            - 'value': Fill with a specific value.
        value (float, optional): Value to fill when using the 'value' method.

    Returns:
        output (ndarray): Data matrix with missing values filled.
        mask (ndarray): Binary mask indicating filled values (1 for filled, 0 otherwise).
    """
    if data.ndim != 2:
        raise ValueError(f"Incomprehensible number of dimensions: {data.ndim}")

    output = data.copy()  
    mask = np.isnan(output)  

    if method.lower() == 'total':
        # Fill with global mean
        global_mean = np.nanmean(output)
        output[mask] = global_mean

    elif method.lower() == 'row':
        # Fill with row-wise means
        row_means = np.nanmean(output, axis=1)
        row_means[np.isnan(row_means)] = 0  
        for i in range(output.shape[0]):
            if np.isnan(output[i, :]).all():
                output[i, :] = row_means[i]  
            else:
                output[i, np.isnan(output[i, :])] = row_means[i]  

    elif method.lower() == 'column':
        # Fill with column-wise means
        col_means = np.nanmean(output, axis=0)
        col_means[np.isnan(col_means)] = 0  
        for j in range(output.shape[1]):
            output[np.isnan(output[:, j]), j] = col_means[j]

    elif method.lower() == 'interpolation':
        # Linear interpolation along rows
        for j in range(output.shape[1]):
            col = output[:, j]
            nan_indices = np.where(np.isnan(col))[0]
            if len(nan_indices) == 0:
                continue  

            # Fill gaps using linear interpolation
            valid_indices = np.where(~np.isnan(col))[0]
            valid_values = col[valid_indices]
            interpolated = np.interp(nan_indices, valid_indices, valid_values)
            output[nan_indices, j] = interpolated

    elif method.lower() == 'value':
        # Fill with a specific value
        if value is None:
            raise ValueError("Value parameter is missing for 'value' method!")
        output[mask] = value

    else:
        raise ValueError(f"Unknown method: {method}")

    return output, mask



def normalize(data, *params):
    """
    Normalize data with options for handling mean and standard deviation.

    Parameters:
        data (ndarray): Input data matrix, can contain NaN values.
        *params: Normalization options as strings. Options include:
            - 'mean': Remove global mean.
            - 'std': Remove global standard deviation.
            - 'meanrows': Remove row means.
            - 'meancols': Remove column means.
            - 'stdrows': Remove row standard deviations.
            - 'stdcols': Remove column standard deviations.

    Returns:
        output (ndarray): Normalized data.
        varargout (list): List of removed means and standard deviations, 
                          depending on the normalization options given.
    """
    output = data.copy()  
    rows, columns = output.shape
    varargout = []

    if not params:  
        total_mean = np.nanmean(output) 
        output -= total_mean
        total_std = np.nanstd(output)  
        if total_std == 0:
            total_std = 1 
        output /= total_std
        varargout = [total_mean, total_std]
    else:
        for param in params:
            param = param.lower()
            if param == 'mean':
                total_mean = np.nanmean(output)
                output -= total_mean
                varargout.append(total_mean)
            elif param == 'std':
                total_std = np.nanstd(output)
                if total_std == 0:
                    total_std = 1  
                output /= total_std
                varargout.append(total_std)
            elif param == 'meanrows':
                nan_rows = np.isnan(output).all(axis=1) 
                output[nan_rows] = 0 
                row_means = np.nanmean(output, axis=1)
                row_means[np.isnan(row_means)] = 0  
                output -= row_means[:, np.newaxis]
                varargout.append(row_means)
            elif param == 'meancols':
                nan_cols = np.isnan(output).all(axis=0)
                output[nan_cols] = 0
                col_means = np.nanmean(output, axis=0)
                col_means[np.isnan(col_means)] = 0  
                output -= col_means[np.newaxis, :]
                varargout.append(col_means)
            elif param == 'stdrows':
                nan_rows = np.isnan(output).all(axis=1) 
                output[nan_rows] = 1 
                row_stds = np.nanstd(output, axis=1)
                row_stds[np.isnan(row_stds)] = 1  
                row_stds[row_stds == 0] = 1  
                output /= row_stds[:, np.newaxis]
                varargout.append(row_stds)
            elif param == 'stdcols':
                nan_cols = np.isnan(output).all(axis=0)
                output[nan_cols] = 1
                col_stds = np.nanstd(output, axis=0)
                col_stds[np.isnan(col_stds)] = 1  
                col_stds[col_stds == 0] = 1  
                output /= col_stds[np.newaxis, :]
                varargout.append(col_stds)
            else:
                print(f"Unrecognized option '{param}' --> ignored!")

    return output, varargout



def select_set(data, percent, method='random', cloud_mask=None):
    """
    Randomly removes data for validation purposes.
    
    Parameters:
        data (ndarray): Input data (2D or 3D) with missing values as NaN.
        percent (float): Percentage of non-missing data to remove (0 to 100).
        method (str): Method of removal ('random' or 'cloud').
        cloud_mask (ndarray, optional): Cloud mask (2D or 3D binary matrix).
                                        Required if method is 'cloud'.
                                        
    Returns:
        output (ndarray): Modified data with validation values set to NaN.
        valData (list): Validation data values removed from the dataset.
        valIndex (list): Indices of the removed validation data.
    """
    if method.lower() == 'cloud' and cloud_mask is None:
        print('No cloud mask defined, using random set selection!')
        method = 'random'
    elif method.lower() not in ['random', 'cloud']:
        print('Incorrect method defined, using random set selection!')
        method = 'random'

    # Calculate total data points and number to remove
    n_data = data.size
    n_nan = np.isnan(data).sum()
    n_val_data = int((n_data - n_nan) * percent / 100)

    # Initialize validation data and indices
    val_data = []
    val_index = []

    if method.lower() == 'random':
        # Randomly remove data points
        flat_data = data.flatten()
        available_indices = np.where(~np.isnan(flat_data))[0]  
        selected_indices = np.random.choice(available_indices, n_val_data, replace=False)

        for idx in selected_indices:
            val_data.append(flat_data[idx])
            val_index.append(idx)
            flat_data[idx] = np.nan

        output = flat_data.reshape(data.shape)

    elif method.lower() == 'cloud':
        # Validate cloud mask dimensions
        if cloud_mask.ndim not in [2, 3]:
            raise ValueError("Cloud mask must be 2D or 3D binary matrix!")
        
        # Handle 2D or 3D cloud mask
        if cloud_mask.ndim == 2:
            cloud_mask = cloud_mask[:, :, np.newaxis]  
        n_clouds = cloud_mask.shape[2]

        output = data.copy()
        orig_data_missing = np.isnan(output).sum()

        cloud_counter = 0

        while np.isnan(output).sum() - orig_data_missing < n_val_data:
            # Select a cloud from the mask
            cloud = cloud_mask[:, :, cloud_counter % n_clouds]
            cloud_counter += 1

            # Crop the cloud to eliminate empty edges
            non_zero_rows = np.any(cloud, axis=1)
            non_zero_cols = np.any(cloud, axis=0)
            cloud = cloud[non_zero_rows][:, non_zero_cols]

            # Randomly place the cloud in the data
            rows, cols, time = output.shape
            cloud_rows, cloud_cols = cloud.shape
            row_anchor = np.random.randint(0, rows - cloud_rows + 1)
            col_anchor = np.random.randint(0, cols - cloud_cols + 1)
            time_anchor = np.random.randint(0, time)

            # Apply cloud only to valid locations
            region = output[row_anchor:row_anchor+cloud_rows, col_anchor:col_anchor+cloud_cols, time_anchor]
            if np.isnan(region).any():
                continue  

            # Save validation data and apply cloud
            val_indices = np.where(cloud)
            val_values = region[val_indices]
            val_data.extend(val_values)
            region[val_indices] = np.nan
            val_index.extend(zip(val_indices[0] + row_anchor, val_indices[1] + col_anchor, [time_anchor] * len(val_indices[0])))

            output[row_anchor:row_anchor+cloud_rows, col_anchor:col_anchor+cloud_cols, time_anchor] = region

    return output, np.array(val_data), np.array(val_index)




def eof_core_torch(data, mask, eofs, stop_criterion=1, max_iterations=float('inf'), device='cuda'):
    """
    Perform EOF analysis and estimate missing values using PyTorch for GPU computation.

    Parameters:
        data (ndarray or torch.Tensor): 2D data matrix with missing values already initialized.
        mask (ndarray or torch.Tensor): Binary matrix of the same shape as `data`.
                                         1 indicates a missing value, 0 indicates known data.
        eofs (int, list, or ndarray/torch.Tensor): Specifies which EOFs to use:
            - Scalar: Number of consecutive EOFs to use.
            - Vector: Specific EOFs to use.
            - Matrix: Specifies EOFs for each iteration round (padded with zeros for variable EOF counts).
        stop_criterion (float, optional): Minimum change for convergence. Default is 1.
        max_iterations (int, optional): Maximum number of iterations. Default is infinity.
        device (str, optional): Device to perform computations on. Default is 'cuda'.

    Returns:
        output (ndarray): Data matrix with missing values estimated.
        eigenvalues (ndarray): Eigenvalues from the EOF analysis.
        varargout (int, optional): Number of iterations performed (if requested).
    """
    # Convert inputs to torch.Tensor and move to specified device
    data = torch.tensor(data, dtype=torch.float32, device=device) if not isinstance(data, torch.Tensor) else data.to(device)
    mask = torch.tensor(mask, dtype=torch.float32, device=device) if not isinstance(mask, torch.Tensor) else mask.to(device)

    # Initialize parameters based on inputs
    if isinstance(eofs, int):
        if eofs > min(data.shape):
            value = torch.arange(1, min(data.shape) + 1, device=device)  # Use all EOFs
        else:
            value = torch.arange(1, eofs + 1, device=device)  # Use the first `eofs` EOFs
    else:
        value = torch.tensor(eofs, dtype=torch.int64, device=device) if not isinstance(eofs, torch.Tensor) else eofs.to(device)

    # Ensure EOF selection vector is clean
    value = value[value > 0]

    output = data.clone()  # Initialize output
    reference = torch.zeros_like(data, device=device)

    if value.ndim == 1:  
        round_count = 0
        while True:
            # Calculate SVD
            U, singular_values, Vt = torch.linalg.svd(output, full_matrices=False)
            eigenvalues = singular_values.diag_embed()

            # Estimate missing values
            reconstruction = (U[:, value - 1] @ eigenvalues[value - 1][:, value - 1] @ Vt[value - 1, :])
            output = data * (1 - mask) + reconstruction * mask

            # Check convergence
            change = torch.sum((reference * mask - output * mask) ** 2)
            if change.item() <= stop_criterion or round_count >= max_iterations:
                break

            reference = output.clone()
            round_count += 1

        # Convert outputs back to NumPy
        return output.cpu().numpy(), eigenvalues.cpu().numpy(), round_count

    else:  # EOF pruned calculation for variable EOF counts per round
        for round_num in range(min(value.shape[0], int(max_iterations))):
            # Select EOFs for this round
            index = value[round_num, :]
            index = index[index > 0]  # Remove zeros

            # Perform SVD
            U, singular_values, Vt = torch.linalg.svd(output, full_matrices=False)
            eigenvalues = singular_values.diag_embed()

            # Update missing value estimates
            reconstruction = (U[:, index - 1] @ eigenvalues[index - 1][:, index - 1] @ Vt[index - 1, :])
            output = data * (1 - mask) + reconstruction * mask

        # Convert outputs back to NumPy
        return output.cpu().numpy(), eigenvalues.cpu().numpy(), round_count




def func_eofszb(Eof_Data):
    """
    Perform Empirical Orthogonal Function (EOF) analysis on the given data.
    """
    # Determine size of input data
    m, n = Eof_Data.shape
    
    # Calculate mean along spatial dimension and normalize data
    Mean_Eof = np.mean(Eof_Data, axis=1, keepdims=True)
    Mean_Eof = np.tile(Mean_Eof, (1, n)) 
    Eof_Data_centered = Eof_Data - Mean_Eof  
    
    # Compute the covariance matrix (RSS_Eof)
    RSS_Eof = np.dot(Eof_Data_centered.T, Eof_Data_centered)  
    
    # Eigen decomposition
    EigenValue, EigenVector = np.linalg.eig(RSS_Eof)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = EigenValue.argsort()[::-1]
    EigenValue = EigenValue[idx]
    EigenVector = EigenVector[:, idx]
    
    # Clamp eigenvalues to non-negative to avoid sqrt issues
    EigenValue = np.maximum(EigenValue, 0)
    
    # Calculate spatial eigenvectors
    EigenVector_Spatial1 = np.dot(Eof_Data_centered, EigenVector)
    EigenVector_Spatial = np.zeros_like(EigenVector_Spatial1)
    for i in range(len(EigenValue)):
        if EigenValue[i] > 0:  
            EigenVector_Spatial[:, i] = EigenVector_Spatial1[:, i] / np.sqrt(EigenValue[i])
        else:
            EigenVector_Spatial[:, i] = 0  
    
    # Calculate PCA (Principal Component Analysis scores)
    PCA = np.dot(EigenVector_Spatial.T, Eof_Data_centered)
    
    # Scale eigenvalues
    EigenValue = EigenValue * len(EigenValue)
    
    # Calculate cumulative explained variance ratio
    Cum = np.cumsum(EigenValue) / np.sum(EigenValue)
    
    return PCA, EigenVector_Spatial, EigenValue, Cum, Mean_Eof
