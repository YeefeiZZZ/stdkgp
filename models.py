import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, inplace=False, output_dim=None):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='same')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1)
        
        self.dropout1 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        
        # 8 channels * 7 length (approx after pooling) -> 16 hidden
        self.fc1 = nn.Linear(8 * 7, 16)
        
        # If output_dim is provided, add a final regression layer (for pure NN training)
        # If None, it acts as a feature extractor (for DKGP)
        self.output_dim = output_dim
        if self.output_dim is not None:
            self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        # x shape: (batch, 1, seq_len) or similar depending on usage
        if x.dim() == 3 and x.size(1) == 1: 
            # Standard format from data_loader: (batch, seq_len, 1) -> needs permute for Conv1d
            # But the original code sliced x[:,:,0:10] and treated it as space
            pass
            
        space = x[:, :, 0:10] # Input features
        # temporal = x[:, :, 10] # Time feature (if present)

        # Conv1d expects (Batch, Channels, Length)
        # Assuming input 'space' is (Batch, 1, 10) or (Batch, 10, 1) - Original code implies specific shape
        # Based on original code: self.conv1(space)
        
        layer1 = self.pool1(self.relu1(self.conv1(space)))
        layer2 = self.pool2(self.relu2(self.conv2(layer1)))
        layer3 = self.pool3(self.relu3(self.conv3(layer2)))
        layer4 = self.fc1(self.flatten(layer3))
        
        out = layer4
        
        if self.output_dim is not None:
            out = self.fc2(out)
        else:
            # For DKGP, we usually append temporal feature here if needed
            # Replicating original logic:
            if x.size(2) > 10:
                temporal = x[:, :, 10]
                out = torch.cat((out, temporal), 1)
                
        return out

class RNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16, layer_dim=3, output_dim=None, device='cpu'):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        self.output_dim = output_dim
        if output_dim is not None:
            self.fc = nn.Linear(hidden_dim, output_dim)
        # Original code for DKGP didn't have a final FC layer inside RNN class for regression
        # but pure NN training script did.

    def forward(self, x):
        space = x[:, :, 0:10]
        # temporal = x[:, :, 10]
        
        h0 = torch.zeros(self.layer_dim, space.size(0), self.hidden_dim).to(self.device)
        out, h0 = self.rnn(space, h0.detach())
        
        if self.output_dim is not None:
            # For pure NN training
            out = self.fc(out)
            out = out[:, -1, :]
        else:
            # For DKGP extractor
            out = out[:, -1, :]
            if x.size(2) > 10:
                temporal = x[:, :, 10]
                out = torch.cat((out, temporal), 1)
        
        return out

class ANN(nn.Module):
    def __init__(self, inplace=False, output_dim=None):
        super(ANN, self).__init__()
        self.linear1 = nn.Linear(in_features=10, out_features=64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=32, out_features=16)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        
        self.output_dim = output_dim
        if output_dim is not None:
            self.linear4 = nn.Linear(in_features=16, out_features=output_dim)

    def forward(self, x):
        space = x[:, :, 0:10]
        
        layer1 = self.relu1(self.linear1(space))
        layer2 = self.relu2(self.linear2(layer1))
        layer3 = self.relu3(self.linear3(layer2))
        
        out = layer3
        
        if self.output_dim is not None:
            out = self.linear4(layer3)
            out = out.reshape(out.size(0), -1)
        else:
             # For DKGP extractor
            if x.size(2) > 10:
                temporal = x[:, :, 10:11]
                out = torch.cat((out, temporal), 2)
                out = out.reshape(out.size(0), -1)
        
        return out