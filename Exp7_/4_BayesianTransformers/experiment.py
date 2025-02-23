import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import sys

sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')
from bayeformers import to_bayesian
import bayeformers.nn as bnn

# 1. Загрузка данных
data = pd.read_csv(r'Exp7_\data\MSFT_Monthly_stock_prizes.csv')
data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

# Выбираем признаки и целевую переменную
features = ["Open", "High", "Low", "Volume"]
target = "Close"
X = data[features]
y = data[target]

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Определение параметров эксперимента
layer_options = [1, 2, 4, 6, 8, 10]
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]
epochs = 100
results = []

# 3. Класс трансформера
class DynamicTransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, num_heads=4, hidden_dim=128):
        super(DynamicTransformerModel, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.linear_in(x).unsqueeze(1)  # (batch_size, seq_len=1, hidden_dim)
        x = self.transformer(x).mean(dim=1)
        return self.fc(x)

# 4. Запуск эксперимента
for num_layers in layer_options:
    for test_size in test_size_options:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        
        model = DynamicTransformerModel(input_size=X_train.shape[1], num_layers=num_layers)
        bayesian_model = to_bayesian(model, delta=0.05, freeze=True)
        optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            bayesian_model.train()
            optimizer.zero_grad()
            loss = F.mse_loss(bayesian_model(X_train_tensor), y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Оценка модели
        bayesian_model.eval()
        with torch.no_grad():
            y_pred = bayesian_model(X_test_tensor).numpy()
            mae = mean_absolute_error(y_test, y_pred)
            variance = np.var(y_pred)
        
        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'mae': mae,
            'variance': variance
        })
        print(f"Слои: {num_layers}, Тестовая выборка: {test_size}, MAE: {mae:.4f}, Дисперсия: {variance:.4f}")

# 5. Сохранение результатов
results_df = pd.DataFrame(results)
results_df.to_csv("Exp7_\\transformer_results.csv", index=False)
print("Эксперимент завершен! Результаты сохранены в 'transformer_results.csv'")



''' Слои: 1, Тестовая выборка: 0.1, MAE: 71.0266, Дисперсия: 0.0009
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 1, Тестовая выборка: 0.2, MAE: 59.2247, Дисперсия: 0.0006
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 1, Тестовая выборка: 0.5, MAE: 70.5881, Дисперсия: 0.0003
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 1, Тестовая выборка: 0.8, MAE: 74.8667, Дисперсия: 0.0004
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 1, Тестовая выборка: 0.9, MAE: 75.5808, Дисперсия: 0.0004
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 2, Тестовая выборка: 0.1, MAE: 71.1929, Дисперсия: 0.0001
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 2, Тестовая выборка: 0.2, MAE: 59.4377, Дисперсия: 0.0001
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 2, Тестовая выборка: 0.5, MAE: 70.4656, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 2, Тестовая выборка: 0.8, MAE: 74.8539, Дисперсия: 0.0001
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 2, Тестовая выборка: 0.9, MAE: 74.6378, Дисперсия: 0.0002
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 4, Тестовая выборка: 0.1, MAE: 71.1301, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 4, Тестовая выборка: 0.2, MAE: 59.5165, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 4, Тестовая выборка: 0.5, MAE: 70.4081, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 4, Тестовая выборка: 0.8, MAE: 74.9215, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 4, Тестовая выборка: 0.9, MAE: 74.9476, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 6, Тестовая выборка: 0.1, MAE: 71.1684, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 6, Тестовая выборка: 0.2, MAE: 59.1313, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 6, Тестовая выборка: 0.5, MAE: 70.5481, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 6, Тестовая выборка: 0.8, MAE: 74.6585, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 6, Тестовая выборка: 0.9, MAE: 74.9454, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 8, Тестовая выборка: 0.1, MAE: 71.0240, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 8, Тестовая выборка: 0.2, MAE: 59.1869, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 8, Тестовая выборка: 0.5, MAE: 70.7961, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 8, Тестовая выборка: 0.8, MAE: 75.0543, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 8, Тестовая выборка: 0.9, MAE: 74.9805, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 10, Тестовая выборка: 0.1, MAE: 71.2087, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 10, Тестовая выборка: 0.2, MAE: 59.6271, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 10, Тестовая выборка: 0.5, MAE: 70.8076, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 10, Тестовая выборка: 0.8, MAE: 74.6673, Дисперсия: 0.0000
c:\Users\huawei\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Слои: 10, Тестовая выборка: 0.9, MAE: 75.2725, Дисперсия: 0.0000'''
