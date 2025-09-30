import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Create Dummy Dataset (for testing)
# -----------------------------
# 5 units, 100 cycles each, 5 sensors
data = []
for unit in range(1, 6):
    for cycle in range(1, 101):
        sensors = np.random.rand(5)  # 5 sensors for simplicity
        data.append([unit, cycle] + list(sensors))

train_data = pd.DataFrame(data, columns=['unit_number', 'time_in_cycles',
                                         'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5'])

# -----------------------------
# Step 2: Calculate RUL
# -----------------------------
max_cycles = train_data.groupby('unit_number')['time_in_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycle']
train_data = train_data.merge(max_cycles, on='unit_number', how='left')
train_data['RUL'] = train_data['max_cycle'] - train_data['time_in_cycles']

# -----------------------------
# Step 3: Scale sensor features
# -----------------------------
sensor_features = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
scaler = MinMaxScaler()
train_data[sensor_features] = scaler.fit_transform(train_data[sensor_features])

# -----------------------------
# Step 4: Create sequences
# -----------------------------
SEQ_LEN = 20
def create_sequences(df, seq_len, sensor_features):
    sequences = []
    labels = []
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        sensor_values = unit_data[sensor_features].values
        rul_values = unit_data['RUL'].values
        for i in range(len(sensor_values) - seq_len + 1):
            sequences.append(sensor_values[i:i+seq_len])
            labels.append(rul_values[i+seq_len-1])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(train_data, SEQ_LEN, sensor_features)
print("Shape of X:", X.shape, "Shape of y:", y.shape)

# -----------------------------
# Step 5: Build LSTM Model
# -----------------------------
model = Sequential([
    LSTM(50, input_shape=(SEQ_LEN, len(sensor_features))),
    Dropout(0.2),
    Dense(20, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# -----------------------------
# Step 6: Train Model
# -----------------------------
history = model.fit(X, y, epochs=5, batch_size=16, validation_split=0.1, shuffle=True)

# -----------------------------
# Step 7: Plot Loss
# -----------------------------
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

# -----------------------------
# Step 8: Predict RUL for first sequence
# -----------------------------
sample_X = X[0].reshape(1, SEQ_LEN, len(sensor_features))
predicted_rul = model.predict(sample_X)
print("Predicted RUL:", predicted_rul[0][0])

