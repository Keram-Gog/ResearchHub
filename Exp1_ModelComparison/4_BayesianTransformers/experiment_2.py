import sys
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')
from bayeformers import to_bayesian
import bayeformers.nn as bnn

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# -- Для воспроизводимости:
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

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
        x = x.unsqueeze(0)      # Добавляем размерность seq_len=1
        x = self.transformer_encoder(x)
        x = x.squeeze(0)        # Убираем размерность seq_len
        return self.output_layer(x)

def main():
    # ======== 4. Загрузка данных ========
    try:
        data = pd.read_csv(
            r'D:\main for my it\my tasks\source\ResearchHub\Exp1_ModelComparison\data\student-mat.csv', 
            sep=';'
        )
        print("Данные успешно загружены!")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    # ======== 5. Подготовка X и y ========
    X = data.drop(columns=['G3'])
    y = data['G3']
    X = pd.get_dummies(X)

    # Масштабирование X
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Масштабирование y (ВАЖНО!)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Параметры перебора
    size_options = [1.0, 0.5, 0.25, 0.1]
    sparsity_options = [0.0, 0.25, 0.5, 0.9]
    layer_options = [1, 2, 4, 6, 8, 10]

    epochs = 100
    learning_rate = 0.0001
    weight_decay = 0.0001
    results = []

    # ======== 6. Цикл по (Size, Sparsity, Complexity) ========
    for size in size_options:
        for sparsity in sparsity_options:
            # Имитация sparsity и замена NaN средним значением
            X_sp = introduce_sparsity(X_scaled, fraction=sparsity)
            imputer = SimpleImputer(strategy='mean')
            X_sp = imputer.fit_transform(X_sp)
            
            train_size = int(len(X_sp) * size)
            if train_size < 1 or train_size >= len(X_sp):
                continue

            # Разделение на train/test
            X_train, X_test = X_sp[:train_size], X_sp[train_size:]
            y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            input_dim = X_train_t.shape[1]
            
            for num_layers in layer_options:
                # Создание и преобразование модели в байесовскую
                model = BayesianTransformer(input_size=input_dim, d_model=64, num_layers=num_layers)
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

                # Оценка
                bayesian_model.eval()
                with torch.no_grad():
                    test_pred = bayesian_model(X_test_t).numpy()
                    # Приводим предсказания к исходному масштабу
                    test_pred = scaler_y.inverse_transform(test_pred).ravel()
                    y_test_orig = scaler_y.inverse_transform(y_test_t.numpy()).ravel()

                test_mae = mean_absolute_error(y_test_orig, test_pred)
                test_smape = smape(y_test_orig, test_pred)
                test_variance = np.var(test_pred)

                results.append({
                    'size': size,
                    'sparsity': sparsity,
                    'num_layers': num_layers,
                    'mae': test_mae,
                    'smape': test_smape,
                    'variance': test_variance
                })

                print(f"[BayesTransformer] Size={size}, Spars={sparsity}, Layers={num_layers} "
                      f"-> MAE={test_mae:.4f}, sMAPE={test_smape:.2f}%, Var={test_variance:.4f}")

    # ======== 7. Сохранение результатов ========
    results_df = pd.DataFrame(results)
    results_df.to_csv('BayesTransformer_results.csv', index=False)
    print("\nЭксперименты завершены. Итоги в 'BayesTransformer_results.csv'.")

if __name__ == '__main__':
    main()
