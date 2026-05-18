import joblib
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_dl_model(input_path='data/processed/features_data.csv', model_dir='models/'):
    print("--- EasternMartin Technologies: Deep Learning Engine (Track B) ---")

    if not os.path.exists(input_path):
        print(f"[ERROR] Features data not found at {input_path}.")
        return

    print("[INFO] Loading dataset for Neural Network...")
    df = pd.read_csv(input_path)

    # 1. Prepare Features & Target
    y = df['AQI']
    columns_to_drop = ['AQI', 'Date']
    if 'AQI_Bucket' in df.columns:
        columns_to_drop.append('AQI_Bucket')
    X = df.drop(columns=columns_to_drop)

    # One-hot encoding
    X = pd.get_dummies(X, columns=['City'], drop_first=True)
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # 2. 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # 3. Data Scaling (Crucial for Neural Networks)
    print("[INFO] Scaling data (Normalizing for ANN)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Building the Artificial Neural Network (ANN)
    print("[INFO] Designing Neural Network Architecture...")
    model = Sequential()
    # Input Layer + Hidden Layer 1
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    # Hidden Layer 2
    model.add(Dense(32, activation='relu'))
    # Output Layer (1 Neuron for AQI Prediction)
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 5. Training with Epochs
    print("\n[PROCESS] Firing up the Neural Network... (Starting 100 Epochs)")
    
    # Optional: Early stopping prevents overfitting if the model stops learning
    # early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,          # The magic number requested by faculty
        batch_size=32,       # Processes 32 rows at a time
        verbose=1            # Set to 1 to see the epoch progress bar in terminal
    )

    # 6. Evaluation on Test Data
    print("\n[PROCESS] Evaluating Deep Learning Accuracy...")
    dl_predictions = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, dl_predictions)
    r2 = r2_score(y_test, dl_predictions)

    print("\n" + " " * 30)
    print("    DL Model Health Report (Challenger)    ")
    print("="*40)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R2 Score                 : {r2:.4f}")
    print("="*40 + "\n")

    # 7. Save Model & Scaler
    os.makedirs(model_dir, exist_ok=True)
    dl_model_path = os.path.join(model_dir, 'aqi_ann_model.keras')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    # Save Keras Model
    model.save(dl_model_path)
    # Save Scaler using joblib
    joblib.dump(scaler, scaler_path)

    print(f"[SUCCESS] Deep Learning Brain saved at: {dl_model_path}")
    print(f"[SUCCESS] Data Scaler saved at: {scaler_path}")

if __name__=="__main__":
    train_dl_model()