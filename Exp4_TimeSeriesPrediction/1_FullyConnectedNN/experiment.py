import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

# -- Для воспроизводимости:
np.random.seed(42)
torch.manual_seed(42)

# ======== 1. Функция sMAPE ========
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

# ======== 2. Функция, вносящая разреженность (sparsity) ========
def introduce_sparsity(X, fraction):
    X_sp = X.copy()
    mask = np.random.rand(*X_sp.shape) < fraction
    X_sp[mask] = np.nan
    return X_sp

# ======== 3. Кастомный Dataset для временных рядов ========
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======== 4. Класс полносвязной модели ========
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=6, hidden_units=128):
        super(FullyConnectedModel, self).__init__()
        layers = []
        current_size = input_size
        curr_hidden = hidden_units
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, curr_hidden))
            layers.append(nn.ReLU())
            current_size = curr_hidden
            curr_hidden = max(curr_hidden // 2, 1)  # чтобы не стало 0
        layers.append(nn.Linear(current_size, output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

# ======== 5. Функция для тренировки модели ========
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1))  # выпрямляем вход
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ======== 6. Функция для тестирования модели ========
def test_model(model, test_loader, device, scaler_y):
    model.eval()
    predictions = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs.view(inputs.size(0), -1))
            predictions.extend(outputs.cpu().numpy())
            y_true.extend(labels.numpy())
    predictions = np.array(predictions).flatten()
    y_true = np.array(y_true).flatten()
    
    # Обратное преобразование предсказаний и истинных значений
    predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()
    y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
    
    mean_pred = np.mean(predictions_orig)
    var_pred = np.var(predictions_orig)
    mae = mean_absolute_error(y_true_orig, predictions_orig)
    smape_val = smape(y_true_orig, predictions_orig)
    
    print(f"Mean Prediction: {mean_pred:.4f}")
    print(f"Prediction Variance: {var_pred:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"sMAPE: {smape_val:.2f}%")
    
    return mean_pred, var_pred, mae, smape_val

# ======== 7. Функция заполнения пропусков ========
def fill_missing_data(data):
    return data.interpolate(method="linear", limit_direction="forward", axis=0)

# ======== 8. Функция разбиения данных ========
def split_data(data, test_size, random_seed=42):
    np.random.seed(random_seed)
    total_size = data.shape[0]
    test_count = int(total_size * test_size)
    indices = np.arange(total_size)
    test_indices = np.random.choice(indices, size=test_count, replace=False)
    train_indices = np.setdiff1d(indices, test_indices)
    return data[train_indices], data[test_indices]

# ======== 9. Основной блок эксперимента ========
if __name__ == '__main__':
    # Загрузка и подготовка данных
    file_path = r"D:\main for my it\my tasks\source\ResearchHub\Exp4_TimeSeriesPrediction\data\Microsoft_Stock.csv"
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").set_index("Date")
    numerical_columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data[numerical_columns]
    data = fill_missing_data(data)
    
    # Масштабирование входных данных
    scaler_X = StandardScaler()
    data_scaled = scaler_X.fit_transform(data)
    
    # Масштабирование целевого признака (Close)
    scaler_y = StandardScaler()
    data_close = data["Close"].values.reshape(-1, 1)
    scaler_y.fit(data_close)
    
    # Формирование временных последовательностей
    seq_len = 30
    X_seq, y_seq = [], []
    for i in range(len(data_scaled) - seq_len):
        X_seq.append(data_scaled[i:i + seq_len])
        # Прогнозируем значение "Close", масштабированное отдельно
        y_seq.append(scaler_y.transform(data["Close"].values[i + seq_len].reshape(-1, 1))[0, 0])
    X_seq = np.array(X_seq)  # форма: (num_samples, seq_len, features)
    y_seq = np.array(y_seq).reshape(-1, 1)
    
    # Преобразуем последовательности в тензоры
    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.float32)
    
    # Параметры эксперимента
    layer_options = [1, 2, 4, 6, 8, 10]
    test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]
    sparsity_options = [0.0, 0.25, 0.5, 0.9]
    epochs = 50
    results = []
    
    for num_layers in layer_options:
        for test_size in test_size_options:
            for sparsity in sparsity_options:
                print(f"\nЭксперимент: {num_layers} слоёв, test_size={test_size}, sparsity={sparsity}")
                # Инъекция пропусков в данные
                X_seq_np = X_seq.numpy()
                X_seq_sp = introduce_sparsity(X_seq_np, fraction=sparsity)
                # Заполнение пропусков с помощью SimpleImputer
                orig_shape = X_seq_sp.shape
                X_seq_sp_flat = X_seq_sp.reshape(-1, orig_shape[-1])
                imputer = SimpleImputer(strategy='mean')
                X_seq_sp_imputed = imputer.fit_transform(X_seq_sp_flat).reshape(orig_shape)
                
                # Разбиение данных на train/test
                X_train_np, X_test_np = split_data(X_seq_sp_imputed, test_size)
                y_train_np, y_test_np = split_data(y_seq.numpy(), test_size)
                X_train = torch.tensor(X_train_np, dtype=torch.float32)
                y_train = torch.tensor(y_train_np, dtype=torch.float32)
                X_test = torch.tensor(X_test_np, dtype=torch.float32)
                y_test = torch.tensor(y_test_np, dtype=torch.float32)
                
                # Подготовка DataLoader
                train_dataset = TimeSeriesDataset(X_train, y_train)
                test_dataset = TimeSeriesDataset(X_test, y_test)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # Параметры модели
                input_size = X_train.size(1) * X_train.size(2)  # выпрямляем последовательность
                output_size = 1
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model = FullyConnectedModel(input_size=input_size, output_size=output_size, num_layers=num_layers).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
                
                print("Обучение модели...")
                train_model(model, train_loader, criterion, optimizer, epochs, device)
                
                print("Тестирование модели...")
                mean_pred, var_pred, mae, smape_val = test_model(model, test_loader, device, scaler_y)
                
                results.append({
                    'num_layers': num_layers,
                    'test_size': test_size,
                    'sparsity': sparsity,
                    'mean_prediction': mean_pred,
                    'variance_prediction': var_pred,
                    'mae': mae,
                    'smape': smape_val
                })
    
    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(r'Exp4_TimeSeriesPrediction/1_FullyConnectedNN/experiment_results_.csv', index=False)
    print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results.csv'.")
