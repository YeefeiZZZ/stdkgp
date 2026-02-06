import pandas as pd
import numpy as np
from scipy.signal import medfilt
import sys
import os

def traffic_data(data_path, trainsteps, numsensor, train_step, step_current, prediction_steps, sigma, speed_volume):
    """
    Load and preprocess traffic data.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    try:
        dataraw_read = pd.read_csv(data_path, sep=',')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    dataraw = np.float64(dataraw_read)
    # Apply median filter
    dataraw = np.apply_along_axis(medfilt, axis=0, arr=dataraw, kernel_size=5)
    
    # Add noise
    data = dataraw + sigma * np.random.rand(len(dataraw[:, 1]), len(dataraw[1, :]))
    
    # Create time index
    time = np.expand_dims(np.float64(list(range(len(data[:, 1])))) + 1, axis=1)

    # Prepare Training Data
    X_train = np.empty((0, trainsteps, 1))
    time_train = np.empty((0, 1))
    y_train = np.empty((0, 1))

    for s in range(train_step):
        index = step_current + s - 1
        # Check bounds
        if index + trainsteps >= len(data):
            break
        X_train = np.concatenate((X_train, data[index:index + trainsteps, numsensor].reshape(1, trainsteps, 1)), axis=0)
        time_train = np.concatenate((time_train, time[index + trainsteps, 0].reshape(1, 1)), axis=0)
        y_train = np.concatenate((y_train, data[index + trainsteps, numsensor].reshape(1, 1)), axis=0)

    # Prepare Test Data
    X_test = np.empty((0, trainsteps, 1))
    time_test = np.empty((0, 1))
    y_test = np.empty((0, 1))
    s_t = train_step + step_current
    
    for s in range(prediction_steps):
        # Check bounds
        if s_t + trainsteps + s - 1 >= len(data):
            break
        X_test = np.concatenate((X_test, data[s_t - 1 + s:s_t + trainsteps + s - 1, numsensor].reshape(1, trainsteps, 1)), axis=0)
        time_test = np.concatenate((time_test, time[s_t + trainsteps + s - 1, 0].reshape(1, 1)), axis=0)
        y_test = np.concatenate((y_test, data[s_t + trainsteps + s - 1, numsensor].reshape(1, 1)), axis=0)

    return X_train, y_train, time_train, X_test, time_test, y_test