import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# 0.8-0.9 0.05-0.07
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')
from bayeformers import to_bayesian

# Загрузка и предобработка данных
file_path = r'Exp5_\data\cleaned_weather.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

features = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 
            'sh', 'H2OC', 'rho', 'wv', 'max. wv', 'wd', 'rain', 
            'raining', 'SWDR', 'PAR', 'max. PAR', 'Tlog']
data = data[features]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

def create_dataset(data_array, look_back=24):
    X, y = [], []
    for i in range(len(data_array) - look_back):
        X.append(data_array[i : i + look_back])
        y.append(data_array[i + look_back, 1])
    return np.array(X), np.array(y)

look_back = 24
X, y = create_dataset(data_scaled, look_back)
num_features = X.shape[2]
X_flat = X.reshape(X.shape[0], look_back * num_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesianRegressionModel(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size=64):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

training_fractions = [0.1, 0.2, 0.5, 0.8, 0.9]
layer_counts = [1, 2, 4, 6, 8, 10]
epochs = 50
batch_size = 32
learning_rate = 0.001

results = []

def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss_fn(model(batch_X), batch_y).backward()
        optimizer.step()

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor.to(device))
        test_variance = test_predictions.var().item()
        test_mae = mean_absolute_error(y_test_tensor.cpu().numpy(), test_predictions.cpu().numpy())
    return test_mae, test_variance

for frac in tqdm(training_fractions, desc="Training Progress"):
    train_size = int(len(X_flat) * frac)
    X_train_tensor = torch.tensor(X_flat[:train_size], dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y[:train_size], dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_flat[train_size:], dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y[train_size:], dtype=torch.float32).view(-1, 1).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for num_layers in layer_counts:
        model = BayesianRegressionModel(X_train_tensor.shape[1], num_layers, hidden_size=64).to(device)
        bayesian_model = to_bayesian(model, delta=0.05, freeze=True)
        optimizer = optim.Adam(bayesian_model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        for _ in range(epochs):
            train_model(bayesian_model, train_loader, optimizer, loss_fn)
        
        test_mae, test_variance = evaluate_model(bayesian_model, X_test_tensor, y_test_tensor)
        results.append({'training_fraction': frac, 'num_layers': num_layers, 'test_mae': test_mae, 'test_variance': test_variance})
        print(f"Слои: {num_layers}, Train Fraction: {frac}, MAE: {test_mae:.4f}, Variance: {test_variance:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("bayesian_experiment_results.csv", index=False)
print("Результаты сохранены в 'bayesian_experiment_results.csv'")