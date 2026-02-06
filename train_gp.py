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

# Fix for OpenMP duplicate lib error on some systems
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    parser = argparse.ArgumentParser(description='GP Training script.')
    parser.add_argument('--kernel', type=str, choices=['rbf', 'matern32', 'matern52'],
                        default='matern52', help='Kernel type.')
    parser.add_argument('--data', type=str, choices=['volume', 'speed'],
                        default='volume', help='Type of data.')
    parser.add_argument('--data_path', type=str, default='./dataset/Santander/Data_1.csv',
                        help='Path to dataset CSV.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--steps', type=int, default=500, help='Training steps.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    trainsteps = 10
    train_step = 2000
    prediction_steps = 500
    n_run = 1 # Runs per sensor
    
    RMSE_all_sensor = []
    time_all = []
    
    # Initialize result containers
    # Note: Shapes adjusted based on logic, initialized empty
    prediction = np.empty((prediction_steps, 0))
    variance = np.empty((prediction_steps, 0))
    Y_test_all = np.empty((prediction_steps, 0))

    for n in range(n_run):
        start_time = time.time()
        RMSE_SUM = 0
        
        # Loop over first 5 sensors as in original code
        for n_sensor in range(5):
            print(f"Processing Sensor {n_sensor}...")
            
            X_train, y_train, time_train, X_test, time_test, y_test = traffic_data(
                args.data_path, trainsteps, n_sensor, train_step, 1, prediction_steps, 0, args.data
            )
            
            # Tensor conversion
            y_test_t = torch.tensor(y_test.reshape(-1), dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train.reshape(-1), dtype=torch.float32).to(device)
            X_train_t = torch.tensor(X_train.reshape(X_train.shape[0], trainsteps), dtype=torch.float32).to(device)
            X_test_t = torch.tensor(X_test.reshape(X_test.shape[0], trainsteps), dtype=torch.float32).to(device)
            time_train_t = torch.tensor(time_train.reshape(time_train.shape[0], 1), dtype=torch.float32).to(device)
            time_test_t = torch.tensor(time_test.reshape(time_test.shape[0], 1), dtype=torch.float32).to(device)

            # Concatenate time feature
            X_train_t = torch.cat((X_train_t, time_train_t), 1)
            X_test_t = torch.cat((X_test_t, time_test_t), 1)

            pyro.clear_param_store()
            
            # Kernel selection
            input_dim = 10 # 10 spatial/lag features
            if args.kernel == 'rbf':
                base_kernel = gp.kernels.RBF(input_dim=input_dim, lengthscale=torch.ones(1).to(device), active_dims=list(range(input_dim)))
            elif args.kernel == 'matern32':
                 base_kernel = gp.kernels.Matern32(input_dim=input_dim, lengthscale=torch.ones(1).to(device), active_dims=list(range(input_dim)))
            else: # matern52
                 base_kernel = gp.kernels.Matern52(input_dim=input_dim, lengthscale=torch.ones(1).to(device), active_dims=list(range(input_dim)))

            # Periodic kernel on time dimension (index 10)
            periodic_kernel = gp.kernels.Periodic(input_dim=1, period=(torch.ones(1)*100).to(device), active_dims=[input_dim])
            
            kernel = gp.kernels.Product(base_kernel, periodic_kernel)

            gpr = gp.models.GPRegression(X_train_t, y_train_t, kernel, noise=torch.tensor(5.0).to(device)).to(device)
            optimizer = torch.optim.Adam(gpr.parameters(), lr=args.lr)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

            losses = []
            
            # Training Loop
            for i in range(args.steps):
                gpr.train()
                optimizer.zero_grad()
                loss = loss_fn(gpr.model, gpr.guide)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if (i + 1) % 100 == 0:
                    print(f'Iter {i+1}/{args.steps} - Loss: {loss.item():.3f}')

            # Evaluation
            gpr.eval()
            with torch.no_grad():
                output = gpr(X_test_t, noiseless=False)
                mean = output[0].detach().cpu().numpy()
                var = output[1].detach().cpu().numpy()

            RMSE = np.sqrt(np.mean((mean - y_test_t.detach().cpu().numpy())**2))
            print(f"Sensor {n_sensor} GP RMSE: {RMSE:.4f}")
            RMSE_SUM += RMSE

            # Plotting (Optional - showing just for the last sensor to avoid blocking)
            if n_sensor == 0: # Only plot first sensor for brevity
                plt.figure(figsize=(10, 5))
                t_plot = np.arange(prediction_steps)
                plt.plot(t_plot, y_test_t.cpu().numpy(), "-b", label="Target")
                plt.plot(t_plot, mean, "-r", label="Prediction")
                plt.fill_between(t_plot, mean - 1.96 * np.sqrt(var), mean + 1.96 * np.sqrt(var), alpha=0.2, color='r')
                plt.title(f"GP Evaluation (Sensor {n_sensor})")
                plt.legend()
                plt.show()

        end_time = time.time()
        time_all.append(end_time - start_time)
        RMSE_average = RMSE_SUM / 5
        RMSE_all_sensor.append(RMSE_average)

    print(f"Average Time: {np.mean(time_all):.4f}s")
    print(f"Average RMSE over all sensors: {np.mean(RMSE_all_sensor):.4f}")

if __name__ == '__main__':
    main()