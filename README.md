# Traffic Prediction with GP, NN, and ST-DKGP

This repository provides implementations for traffic speed and traffic volume prediction using three different modeling approaches:

* **Standard Gaussian Processes (GP)**
* **Neural Networks (NN):** CNN, RNN, ANN
* **Spatiotemporal Deep Kernel Gaussian Processes (ST-DKGP)**

The codebase is refactored from original research scripts to be modular, clean, and easy to run, making it suitable for both research and experimentation.

---

## Repository Structure

* `train_gp.py`: Training script for standard Gaussian Process Regression using Pyro.
* `train_nn.py`: Training script for pure Neural Networks (CNN, RNN, ANN).
* `train_stdkgp.py`: Training script for Spatiotemporal Deep Kernel GP (combining NN extractors with GP).
* `models.py`: Definitions of Neural Network architectures (CNN, RNN, ANN).
* `utils.py`: Data loading and preprocessing utilities.
* `requirements.txt`: Python dependencies.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset

The code expects a CSV file containing traffic data.

* **Default Behavior:** It looks for `dataset/Data_1.csv`.
* **Demo Mode:** If the data file is not found, the script will automatically generate synthetic "dummy" data so you can test the code immediately without errors.
* **Custom Data:** Use the `--data_path` argument to point to your own CSV file. The data should be a matrix where **rows** represent time steps and **columns** represent sensors.

---

## Usage

### 1. Standard Gaussian Process (`train_gp.py`)
Run standard GP regression on the data.

```bash
python train_gp.py --data_path ./dataset/Data_1.csv --kernel matern52 --data volume
```

### Arguments:
* '--kernel': Kernel type ('rbf', `matern32`, `matern52`). Default: matern52.
* `--data`: Type of data to process (`volume`, `speed`). Default: `volume`.
* `--data_path`: Path to the CSV dataset.
* `--lr`: Learning rate.

### 2. Neural Networks (`train_nn.py`)
Train a pure neural network (CNN, RNN, or ANN) for prediction.

```bash
python train_nn.py --extractor cnn --data volume --lr 0.001
```

### Arguments:
* '--extractor': Model architecture ('cnn', `rnn`, `ann`). Default: 'rnn'.
* `--data`: Type of data to process (`volume`, `speed`). Default: `volume`.
* `--data_path`: Path to the CSV dataset.
* `--lr`: Learning rate.

### 3. Spatiotemporal Deep Kernel GP (`train_stdkgp.py`)
Train the Deep Kernel Learning model where a Neural Network acts as a feature extractor for the GP.

```bash
python train_stdkgp.py --extractor cnn --kernel rbf --data volume
```

### Arguments:
* '--extractor': Model architecture ('cnn', `rnn`, `ann`). Default: 'rnn'.
* `--data`: Type of data to process (`volume`, `speed`). Default: `volume`.
* `--data_path`: Path to the CSV dataset.
* `--lr`: Learning rate.

## Requirements
*Python 3.8+

*PyTorch

*Pyro-ppl

*Pandas

*Numpy

*Matplotlib

*Scipy



