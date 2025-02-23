import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Подключаем путь к BayeFormers (укажите актуальный путь)
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')
from bayeformers import to_bayesian
import bayeformers.nn as bnn

# ---------------------------
# 1. Загрузка и предобработка данных
# ---------------------------
file_path = r'Exp5_\data\cleaned_weather.csv'  # Укажите актуальный путь к файлу
data = pd.read_csv(file_path)

# Преобразуем столбец 'date' в datetime и устанавливаем его как индекс
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Выбираем 20 метеорологических признаков
features = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 
            'sh', 'H2OC', 'rho', 'wv', 'max. wv', 'wd', 'rain', 
            'raining', 'SWDR', 'PAR', 'max. PAR', 'Tlog']
data = data[features]

# Нормализуем данные
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Функция для формирования обучающих примеров с использованием скользящего окна
def create_dataset(data_array, look_back=24):
    X, y = [], []
    # Предсказываем следующий шаг для столбца 'T' (индекс 1)
    for i in range(len(data_array) - look_back):
        X.append(data_array[i:i+look_back])
        y.append(data_array[i+look_back, 1])
    return np.array(X), np.array(y)

look_back = 24
X, y = create_dataset(data_scaled, look_back)
# X имеет форму (num_samples, look_back, num_features)
input_shape = (look_back, X.shape[2])

# ---------------------------
# 2. Определение трансформер-блока и модели в PyTorch
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, head_size, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=head_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(head_size)
        self.ff = nn.Sequential(
            nn.Linear(head_size, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, head_size)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(head_size)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, look_back, num_features, n_layers, head_size=64, num_heads=4, ff_dim=64, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.look_back = look_back
        # Проецируем входные признаки в пространство размерности head_size
        self.projection = nn.Linear(num_features, head_size)
        # Позиционное кодирование: используем nn.Embedding для позиции от 0 до look_back-1
        self.pos_embedding = nn.Embedding(look_back, head_size)
        # Список трансформер-блоков
        self.layers = nn.ModuleList([TransformerBlock(head_size, num_heads, ff_dim, dropout) for _ in range(n_layers)])
        # Глобальное усреднение по временной оси и выходной слой
        self.output_layer = nn.Linear(head_size, 1)
        
    def forward(self, x):
        # x: (batch, look_back, num_features)
        batch_size, seq_len, _ = x.shape
        x = self.projection(x)  # (batch, look_back, head_size)
        # Создаем индексы позиций
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)  # (batch, look_back, head_size)
        x = x + pos_emb
        for layer in self.layers:
            x = layer(x)
        # Глобальное усреднение по временной оси (mean по dim=1)
        x = x.mean(dim=1)  # (batch, head_size)
        out = self.output_layer(x)  # (batch, 1)
        return out

# ---------------------------
# 3. Эксперимент: перебор долей обучающей выборки и числа трансформерных блоков
# ---------------------------
training_fractions = [0.1, 0.2, 0.5, 0.8, 0.9]
layer_counts = [1, 2, 4, 6, 8, 10]
epochs = 50
batch_size = 32
learning_rate = 0.001

results = []

def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Перебор комбинаций параметров
for frac in training_fractions:
    train_size = int(len(X) * frac)
    X_train_np = X[:train_size]
    y_train_np = y[:train_size]
    X_test_np = X[train_size:]
    y_test_np = y[train_size:]
    
    # Преобразование numpy массивов в torch тензоры (с сохранением формы: (samples, look_back, num_features))
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1).to(device)
    
    for n_layers in layer_counts:
        input_dim = X_train_tensor.shape[2]  # число признаков
        model = TransformerModel(look_back, input_dim, n_layers, head_size=64, num_heads=4, ff_dim=64, dropout=0.1).to(device)
        
        # Преобразуем модель в байесовскую с помощью BayeFormers
        bayesian_model = to_bayesian(model, delta=0.05, freeze=True)
        
        optimizer = optim.Adam(bayesian_model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        bayesian_model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in get_batches(X_train_tensor, y_train_tensor, batch_size):
                optimizer.zero_grad()
                predictions = bayesian_model(batch_X)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
        
        bayesian_model.eval()
        with torch.no_grad():
            test_predictions = bayesian_model(X_test_tensor)
            test_predictions_np = test_predictions.cpu().numpy()
            y_test_np_cpu = y_test_tensor.cpu().numpy()
            test_mae = mean_absolute_error(y_test_np_cpu, test_predictions_np)
            test_variance = np.var(test_predictions_np)
        
        results.append({
            'training_fraction': frac,
            'n_layers': n_layers,
            'test_mae': test_mae,
            'test_variance': test_variance
        })
        print(f"Training Fraction: {frac}, Layers: {n_layers}, MAE: {test_mae:.4f}, Variance: {test_variance:.4f}")

# ---------------------------
# 4. Сохранение результатов эксперимента в CSV
# ---------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("bayesian_transformer_experiment_results.csv", index=False)
print("Результаты эксперимента сохранены в 'bayesian_transformer_experiment_results.csv'")
