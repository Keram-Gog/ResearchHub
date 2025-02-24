import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import sys
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')

from bayeformers import to_bayesian

# ======== 1. Функция sMAPE ========
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

# ======== 2. Функция sparsity ========
def introduce_sparsity(X, fraction):
    X_sp = X.copy()
    mask = np.random.rand(*X_sp.shape) < fraction
    X_sp[mask] = np.nan
    return X_sp

# ======== 3. Модель Transformer ========
class BayesianTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, num_layers=2):
        super(BayesianTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)  # Приведение к d_model
        x = x.unsqueeze(0)  # Добавляем seq_len=1
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Убираем seq_len
        return self.output_layer(x)

# ======== 4. Загрузка данных ========
try:
    data = pd.read_csv('D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp2_CorrelationDataAnalysis\\data\\nifty_500.csv', sep=',')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

X = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])
y = data['Last Traded Price']
X = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======== Параметры эксперимента ========
size_options = [1.0, 0.5, 0.25, 0.1]
sparsity_options = [0.0, 0.25, 0.5, 0.9]
layer_options = [1, 2, 4, 6, 8, 10]
epochs = 100
learning_rate = 0.0001
weight_decay = 0.0001
results = []

# ======== Основной цикл эксперимента ========
for size in size_options:
    for sparsity in sparsity_options:
        # Применяем sparsity
        X_sp = introduce_sparsity(X_scaled, fraction=sparsity)
        X_sp = np.nan_to_num(X_sp, nan=0.0)
        
        train_size = int(len(X_sp) * size)
        if train_size < 1 or train_size >= len(X_sp):
            continue

        X_train, X_test = X_sp[:train_size], X_sp[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Преобразование в тензоры
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        input_dim = X_train_t.shape[1]
        
        # Пробуем различные количества слоев
        for num_layers in layer_options:
            model = BayesianTransformer(input_dim, num_layers=num_layers)
            bayesian_model = to_bayesian(model, delta=0.05, freeze=True)
            optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Обучение
            for epoch in range(epochs):
                bayesian_model.train()
                optimizer.zero_grad()
                pred = bayesian_model(X_train_t)
                loss = F.mse_loss(pred, y_train_t)
                loss.backward()
                optimizer.step()

            bayesian_model.eval()
            with torch.no_grad():
                test_pred = bayesian_model(X_test_t).numpy().ravel()

            # Оценка
            test_mae = mean_absolute_error(y_test, test_pred)
            test_smape = smape(y_test, test_pred)
            test_variance = np.var(test_pred)

            results.append({
                'size': size, 'sparsity': sparsity, 'num_layers': num_layers,
                'mae': test_mae, 'smape': test_smape, 'variance': test_variance
            })

            print(f"[BayesTransformer] Size={size}, Spars={sparsity}, Layers={num_layers} "
                  f"-> MAE={test_mae:.4f}, sMAPE={test_smape:.2f}%, Var={test_variance:.4f}")

# ======== Сохранение результатов ========
results_df = pd.DataFrame(results)
results_df.to_csv('BayesTransformer_results.csv', index=False)
print("\nЭксперименты завершены. Итоги в 'BayesTransformer_results.csv'.")
