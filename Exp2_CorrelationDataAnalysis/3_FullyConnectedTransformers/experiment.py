import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# -- Для воспроизводимости
np.random.seed(42)
torch.manual_seed(42)

# ======== 1. Функция sMAPE ========
def smape(y_true, y_pred):
    """
    Вычисляет Symmetric Mean Absolute Percentage Error (sMAPE), в %.
    Формула:
      sMAPE = 100 * (2 * |y - y_hat|) / (|y| + |y_hat| + 1e-8)
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

# ======== 2. Функция вносящая "разреженность" (sparsity) ========
def introduce_sparsity(X, fraction):
    """
    fraction - доля элементов, которые заменяются на NaN.
    Пример: fraction=0.25 означает, что 25% значений X станут NaN.
    """
    X_sp = X.copy()
    mask = np.random.rand(*X_sp.shape) < fraction
    X_sp[mask] = np.nan
    return X_sp

# ======== 3. Класс TransformerModel (Classic) ========
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers=2, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1):
        """
        input_dim      - число входных признаков
        num_layers     - число блоков EncoderLayer (complexity)
        d_model        - скрытая размерность в трансформере
        nhead          - количество голов в Multi-Head Attention
        dim_feedforward- размерность скрытого слоя внутри EncoderLayer
        dropout        - вероятность дропаута
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Линейная проекция входных данных в d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Параметр для псевдо-позиционного кодирования
        # Допустим, у нас seq_len=1, но всё равно заведём PE для демонстрации
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Создаём слои трансформера (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Выходной "головной" блок
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Ожидаем x формы (batch_size, input_dim).
        Сформируем "seq_len=1" для подачи в TransformerEncoder:
          -> (batch_size, 1, input_dim)
        Затем проекция на (batch_size, 1, d_model).
        """
        batch_size = x.size(0)
        
        # Делаем фиктивную размерность seq_len=1
        x = x.unsqueeze(1)  # теперь (batch_size, 1, input_dim)
        
        # Линейная проекция + добавление позиционного кодирования (broadcast)
        x = self.input_projection(x)  # (batch_size, 1, d_model)
        x = x + self.positional_encoding  # (1,1,d_model) => broadcast на (batch_size,1,d_model)
        
        # Пропускаем через TransformerEncoder
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        
        # Усредняем (или просто берём x[:,0,:]) - здесь seq_len=1, так что mean = то же самое
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Прогоняем через выходной слой
        x = self.output_layer(x)  # (batch_size, 1)
        
        return x

def main():
    # ======== 4. Загрузка данных ========
    try:
        data = pd.read_csv('D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp2_CorrelationDataAnalysis\\data\\nifty_500.csv', sep=',')
        print("Данные успешно загружены!")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    # ======== 5. Подготовка X и y ========
    X = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])  # Признаки
    y = data['Last Traded Price']                                       # Целевой столбец

    # One-Hot для категориальных
    X = pd.get_dummies(X)
    
    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======== 6. Параметры перебора ========
    size_options = [1.0, 0.5, 0.25, 0.1]       # 100%, 50%, 25%, 10%
    sparsity_options = [0.0, 0.25, 0.5, 0.9]   # 0%, 25%, 50%, 90%
    layer_options = [1, 2, 4, 6, 8, 10]        # Complexity (число слоёв)
    
    epochs = 100
    learning_rate = 0.0001
    weight_decay = 0.0001

    results = []

    # ======== 7. Цикл по (Size, Sparsity, Complexity) ========
    for size in size_options:
        for sparsity in sparsity_options:
            # -- Имитация sparsity
            X_sp = introduce_sparsity(X_scaled, fraction=sparsity)
            # -- Простейшая "заплатка" NaN => 0
            X_sp = np.nan_to_num(X_sp, nan=0.0)
            
            train_size = int(len(X_sp) * size)
            if train_size < 1 or train_size >= len(X_sp):
                print(f"[Пропуск] size={size} -> train_size={train_size} недопустимо.")
                continue
            
            # -- Разделяем (без shuffle) первые train_size в train, остальные в test
            X_train = X_sp[:train_size]
            y_train = y[:train_size]
            X_test  = X_sp[train_size:]
            y_test  = y[train_size:]
            
            # -- Тензоры
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

            input_dim = X_train_t.shape[1]

            for num_layers in layer_options:
                # 1) Создаём модель-трансформер
                model = TransformerModel(input_dim=input_dim, num_layers=num_layers)
                
                # 2) Настройка оптимизатора
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                criterion = nn.MSELoss()
                
                # ======== 8. Обучение ========
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    pred = model(X_train_t)
                    loss = criterion(pred, y_train_t)
                    loss.backward()
                    optimizer.step()
                
                # ======== 9. Оценка ========
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test_t).numpy().ravel()
                
                test_mae = mean_absolute_error(y_test, test_pred)
                test_variance = np.var(test_pred)
                test_smape = smape(y_test, test_pred)
                
                results.append({
                    'size': size,
                    'sparsity': sparsity,
                    'num_layers': num_layers,
                    'mae': test_mae,
                    'smape': test_smape,
                    'variance': test_variance
                })
                
                print(f"[TransformerClassic] Size={size}, Spars={sparsity}, Layers={num_layers} "
                      f"-> MAE={test_mae:.4f}, sMAPE={test_smape:.2f}%, Var={test_variance:.4f}")
    
    # ======== 10. Сохраняем результаты ========
    results_df = pd.DataFrame(results)
    results_df.to_csv('Transformer_classic_results.csv', index=False)
    print("\nЭксперименты завершены. Итоги в 'Transformer_classic_results.csv'.")

# Запуск
if __name__ == '__main__':
    main()
