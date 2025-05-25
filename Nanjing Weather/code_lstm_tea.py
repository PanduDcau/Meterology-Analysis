import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Read the NetCDF file containing temperature data
nc_file = '2m_temperature_nanjing.nc'
ds = xr.open_dataset(nc_file)

# Extract data for Nanjing by finding the closest lat/lon point
# Nanjing coordinates: 32.06°N, 118.79°E
lat_idx = np.argmin(np.abs(ds.latitude.values - 32.06))
lon_idx = np.argmin(np.abs(ds.longitude.values - 118.79))
nanjing_temp = ds.t2m.isel(latitude=lat_idx, longitude=lon_idx)

# Convert xarray DataArray to pandas DataFrame
df = nanjing_temp.to_dataframe().reset_index()
df = df.set_index('valid_time')  # Set the time column as index
df = df[['t2m']]  # Select only the temperature column
df.columns = ['temperature']  # Rename the column for clarity
# Convert temperature from Kelvin to Celsius
df['temperature'] = df['temperature'] - 273.15

# Preprocess the data: normalize using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)


# Function to create sequences for time series prediction
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


# Create sequences with length 3 for prediction
seq_length = 3  # Use 3 time steps to predict the next one
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and testing sets
# Use the last 10 days (240 hours) as the test set
train_size = len(X) - 240
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert numpy arrays to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Create DataLoader for batch processing during training
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model, loss function, and optimizer
model = LSTM(input_size=1, hidden_size=50, num_layers=1, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X.unsqueeze(2))
        # Compute loss
        loss = criterion(outputs, batch_y.unsqueeze(1))
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
model.eval()
with torch.no_grad():
    train_predict = model(X_train.unsqueeze(2)).squeeze().numpy()
    test_predict = model(X_test.unsqueeze(2)).squeeze().numpy()

# Inverse transform the predictions and actual values
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_train = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Calculate Root Mean Squared Error (RMSE)
train_rmse = np.sqrt(np.mean((train_predict - y_train) ** 2))
test_rmse = np.sqrt(np.mean((test_predict - y_test) ** 2))
print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

# Visualize the results
plt.figure(figsize=(15, 6))
plt.plot(df.index[-240:], y_test, label='Actual')
plt.plot(df.index[-240:], test_predict, label='Predicted')
plt.title('LSTM Model: Actual vs Predicted Temperature for Last 10 Days')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
