import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import time
import os
import sys
from utils import traffic_data
from models import CNN, RNN, ANN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    parser = argparse.ArgumentParser(description='NN Training script.')
    parser.add_argument('--extractor', type=str, choices=['cnn', 'rnn', 'ann'],
                        default='rnn', help='Model type.')
    parser.add_argument('--data', type=str, choices=['volume', 'speed'],
                        default='volume', help='Type of data.')
    parser.add_argument('--data_path', type=str, default='./dataset/Santander/Data_1.csv',
                        help='Path to dataset CSV.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--steps', type=int, default=700, help='Training steps.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    trainsteps = 10
    train_step = 96 * 7 * 5 # ~3000 steps
    prediction_steps = 96 * 7
    n_run = 1 
    
    RMSE_all_sensor = []
    time_all = []

    for n in range(n_run):
        start_time = time.time()
        RMSE_SUM = 0
        
        for n_sensor in range(5):
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
            
            # Append time feature
            X_train_t = torch.cat((X_train_t, time_train_t), 2)
            X_test_t = torch.cat((X_test_t, time_test_t), 2)

            # Model Init
            if args.extractor == 'cnn':
                model = CNN(output_dim=1).to(device)
            elif args.extractor == 'ann':
                model = ANN(output_dim=1).to(device)
            elif args.extractor == 'rnn':
                model = RNN(output_dim=1, device=device).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = nn.MSELoss()
            
            losses = []
            
            # Training
            for i in range(args.steps):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = loss_fn(outputs.flatten(), y_train_t)
                loss.backward()
                optimizer.step()
                
                # L2 Regularization (manual)
                l2_lambda = 0.01
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss += l2_lambda * l2_norm
                
                losses.append(loss.item())
                if (i + 1) % 100 == 0:
                    print(f'Iter {i+1}/{args.steps} - Loss: {loss.item():.3f}')

            # Evaluation
            model.eval()
            with torch.no_grad():
                output = model(X_test_t)
                mean = output.flatten().detach().cpu().numpy()
            
            RMSE = np.sqrt(np.mean((mean - y_test_t.cpu().numpy())**2))
            print(f"Sensor {n_sensor} NN RMSE: {RMSE:.4f}")
            RMSE_SUM += RMSE
            
            if n_sensor == 0:
                plt.figure(figsize=(10, 5))
                t_plot = np.arange(prediction_steps)
                plt.plot(t_plot, y_test_t.cpu().numpy(), "-b", label="Target")
                plt.plot(t_plot, mean, "-r", label="Prediction")
                plt.title(f"NN ({args.extractor.upper()}) Evaluation")
                plt.legend()
                plt.show()

        end_time = time.time()
        time_all.append(end_time - start_time)
        RMSE_all_sensor.append(RMSE_SUM / 5)

    print(f"Average Time: {np.mean(time_all):.4f}s")
    print(f"Average RMSE: {np.mean(RMSE_all_sensor):.4f}")

if __name__ == '__main__':
    main()