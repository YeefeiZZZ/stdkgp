import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp
import matplotlib.pyplot as plt
import argparse
import time
import os
import sys
from utils import traffic_data
from models import CNN, RNN, ANN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    parser = argparse.ArgumentParser(description='ST-DKGP Training script.')
    parser.add_argument('--kernel', type=str, choices=['rbf', 'matern32', 'matern52'],
                        default='rbf', help='Kernel type.')
    parser.add_argument('--extractor', type=str, choices=['cnn', 'rnn', 'ann'],
                        default='cnn', help='Feature extractor.')
    parser.add_argument('--data', type=str, choices=['volume', 'speed'],
                        default='volume', help='Type of data.')
    parser.add_argument('--data_path', type=str, default='./dataset/Santander/Data_1.csv',
                        help='Path to dataset CSV.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--steps', type=int, default=500, help='Training steps.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    trainsteps = 10
    train_step = 2000
    prediction_steps = 500
    n_run = 1
    
    RMSE_all = []
    time_all = []

    for n in range(n_run):
        start_time = time.time()
        RMSE_SUM = []
        
        # Only iterate 4 sensors in stdkgp example? Adjusted to 5 to match others or keep logic
        for n_sensor in range(4):
            print(f"Processing Sensor {n_sensor}...")
            
            X_train, y_train, time_train, X_test, time_test, y_test = traffic_data(
                args.data_path, trainsteps, n_sensor, train_step, 1, prediction_steps, 0, args.data
            )
            
            # Tensor prep
            y_test_t = torch.tensor(y_test.reshape(-1), dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train.reshape(-1), dtype=torch.float32).to(device)
            X_train_t = torch.tensor(X_train.reshape(X_train.shape[0], 1, trainsteps), dtype=torch.float32).to(device)
            X_test_t = torch.tensor(X_test.reshape(X_test.shape[0], 1, trainsteps), dtype=torch.float32).to(device)
            time_train_t = torch.tensor(time_train.reshape(time_train.shape[0], 1, 1), dtype=torch.float32).to(device)
            time_test_t = torch.tensor(time_test.reshape(time_test.shape[0], 1, 1), dtype=torch.float32).to(device)
            
            # Concatenate (Batch, Channels, Length/Features)
            X_train_t = torch.cat((X_train_t, time_train_t), 2)
            X_test_t = torch.cat((X_test_t, time_test_t), 2)

            pyro.clear_param_store()
            
            input_dim_nn = 16
            
            if args.kernel == 'rbf':
                base_kernel = gp.kernels.RBF(input_dim=input_dim_nn, lengthscale=torch.ones(1).to(device), active_dims=list(range(input_dim_nn)))
            elif args.kernel == 'matern32':
                base_kernel = gp.kernels.Matern32(input_dim=input_dim_nn, lengthscale=torch.ones(1).to(device), active_dims=list(range(input_dim_nn)))
            else:
                base_kernel = gp.kernels.Matern52(input_dim=input_dim_nn, lengthscale=torch.ones(1).to(device), active_dims=list(range(input_dim_nn)))
            
            # Periodic kernel on the time dimension (index 16)
            periodic_kernel = gp.kernels.Periodic(input_dim=1, period=(torch.ones(1)*100).to(device), active_dims=[input_dim_nn])
            
            kernel = gp.kernels.Product(base_kernel, periodic_kernel)

            # Feature Extractor
            if args.extractor == 'cnn':
                feature_extractor = CNN(output_dim=None).to(device)
            elif args.extractor == 'ann':
                feature_extractor = ANN(output_dim=None).to(device)
            elif args.extractor == 'rnn':
                feature_extractor = RNN(output_dim=None, device=device).to(device)

            # Deep Kernel
            deep_kernel = gp.kernels.Warping(kernel, iwarping_fn=feature_extractor)
            
            gpr = gp.models.GPRegression(X_train_t, y_train_t, deep_kernel, noise=torch.tensor(5.0).to(device)).to(device)
            
            optimizer = torch.optim.Adam(gpr.parameters(), lr=0.0001)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
            
            losses = []

            for i in range(args.steps):
                gpr.train()
                optimizer.zero_grad()
                loss = loss_fn(gpr.model, gpr.guide)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if (i + 1) % 100 == 0:
                    print(f'Iter {i+1}/{args.steps} - Loss: {loss.item():.3f}')
            
            # Eval
            gpr.eval()
            with torch.no_grad():
                output = gpr(X_test_t, noiseless=False)
                mean = output[0].detach().cpu().numpy()
                var = output[1].detach().cpu().numpy()
            
            RMSE = np.sqrt(np.mean((mean - y_test_t.cpu().numpy())**2))
            print(f"Sensor {n_sensor} ST-DKGP RMSE: {RMSE:.4f}")
            RMSE_SUM.append(RMSE)
            
            if n_sensor == 0:
                plt.figure(figsize=(10, 5))
                t_plot = np.arange(prediction_steps)
                plt.plot(t_plot, y_test_t.cpu().numpy(), "-b", label="Target")
                plt.plot(t_plot, mean, "-r", label="Prediction")
                plt.fill_between(t_plot, mean - 1.96 * np.sqrt(var), mean + 1.96 * np.sqrt(var), alpha=0.2, color='r')
                plt.title("ST-DKGP Evaluation")
                plt.legend()
                plt.show()

        end_time = time.time()
        time_all.append(end_time - start_time)
        RMSE_all.append(np.mean(RMSE_SUM))

    print(f"Average Time: {np.mean(time_all):.4f}s")
    print(f"Average RMSE: {np.mean(RMSE_all):.4f}")

if __name__ == '__main__':
    main()