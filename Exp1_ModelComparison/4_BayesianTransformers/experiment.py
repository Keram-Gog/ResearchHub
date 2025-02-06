import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import sys
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')

from bayeformers import to_bayesian
import bayeformers.nn as bnn

# 1. Загрузка и подготовка данных
try:
    data = pd.read_csv('Exp1_ModelComparison\\data\\student-mat.csv', sep=';')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['G3'])  # Используем G3 как целевую переменную
y = data['G3']

# Преобразование категориальных переменных в числовые (one-hot encoding)
X = pd.get_dummies(X)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Определение параметров эксперимента
layer_options = [1, 2, 4, 6, 8, 10, 12]  # Количество слоев
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Размер тестовой выборки
train_size = 100  # Фиксированный размер обучающей выборки
epochs = 100  # Число эпох обучения

# Результаты эксперимента
results = []

# 3. Класс модели с байесовским трансформером
class BayesianTransformerRegressionModel(nn.Module):
    def __init__(self, input_size, num_layers, num_heads=4, dropout=0.1):
        super(BayesianTransformerRegressionModel, self).__init__()
        self.embedding = nn.Linear(input_size, 256)
        self.transformer = bnn.BayesianTransformer(d_model=256, nhead=num_heads, num_encoder_layers=num_layers, dropout=dropout)
        self.fc = bnn.BayesLinear(256, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x, x)
        x = x.squeeze(1)
        return self.fc(x)

# 4. Запуск эксперимента
for num_layers in layer_options:
    for test_size in test_size_options:
        train_size = int(len(X_scaled) * (1 - test_size))
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        if len(X_train) < train_size:
            print(f"Пропуск эксперимента: train_size ({train_size}) превышает доступное число данных ({len(X_train)})")
            continue

        X_train_sub = torch.tensor(X_train[:train_size], dtype=torch.float32)
        y_train_sub = torch.tensor(y_train.values[:train_size], dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        input_size = X_train.shape[1]
        model = BayesianTransformerRegressionModel(input_size, num_layers)
        model = to_bayesian(model, delta=0.05, freeze=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train_sub)
            loss = criterion(predictions, y_train_sub)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            test_mae = mean_absolute_error(y_test, test_predictions.numpy())
            test_variance = np.var(test_predictions.detach().numpy())

        print(f"Predictions: {test_predictions[:10].detach().numpy()}")
        print(f"Variance of predictions: {test_variance}")

        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'mae': test_mae,
            'variance': test_variance
        })

        print(f"Слои: {num_layers}, Тестовая выборка: {test_size}, MAE: {test_mae:.4f}, Variance: {test_variance:.4f}")

# 5. Создание таблицы результатов
results_df = pd.DataFrame(results)
results_df.to_csv('Exp1_ModelComparison\\4_BayesianTransformers\\experiment_results_transformer.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results_transformer.csv'.")
