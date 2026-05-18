import joblib
import pandas as pd 
import numpy as np
import os 
import re
# Deep Learning Import for loading the keras model
from tensorflow.keras.models import load_model

class AQIPredictor:
    def __init__(self, 
                 ml_model_path='models/aqi_stacking_model.pkl', 
                 dl_model_path='models/aqi_ann_model.keras',
                 features_path='models/model_features.pkl',
                 scaler_path='models/scaler.pkl'):
        
        print("--- EasternMartin Technologies: Initializing Dual-Engine Prediction System ---")

        # 1. Check if all required files exist
        missing_files = []
        for path in [ml_model_path, dl_model_path, features_path, scaler_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            raise FileNotFoundError(f"[ERROR] Missing files: {missing_files}. Please run both trainer scripts first.")
        
        # 2. Load ML Brain and Structure
        self.ml_model = joblib.load(ml_model_path)
        self.model_features = joblib.load(features_path)
        print("[SUCCESS] Machine Learning Engine Loaded.")

        # 3. Load DL Brain and Scaler
        self.dl_model = load_model(dl_model_path)
        self.scaler = joblib.load(scaler_path)
        print("[SUCCESS] Deep Learning Engine (ANN) & Scaler Loaded.")
    
    def predict(self, input_data):
        """
        Input: Dictionary of features (PM2.5, PM10, City, etc.)
        Output: Dictionary with both ML and DL predictions
        """
        # 1. Feature Engineering (Lags and Rolls)
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2']
        for p in pollutants:
            if p in input_data:
                if f'{p}_Lag1' not in input_data:
                    input_data[f'{p}_Lag1'] = input_data[p]
                if f'{p}_Roll3' not in input_data:
                    input_data[f'{p}_Roll3'] = input_data[p]

        # 2. Convert dictionary to DataFrame
        df_input = pd.DataFrame([input_data])

        # 3. Encoding Cities
        df_input = pd.get_dummies(df_input)

        # 4. Clean column names (Remove special chars)
        df_input = df_input.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        # 5. Align columns exactly as they were during training
        for col in self.model_features:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[self.model_features]

        # ==========================================
        # ENGINE 1: Machine Learning Prediction
        # ==========================================
        ml_prediction = self.ml_model.predict(df_input)
        ml_result = round(float(ml_prediction[0]), 2)

        # ==========================================
        # ENGINE 2: Deep Learning Prediction
        # ==========================================
        # ANN needs scaled data!
        scaled_input = self.scaler.transform(df_input)
        dl_prediction = self.dl_model.predict(scaled_input, verbose=0) # verbose=0 to hide terminal spam
        dl_result = round(float(dl_prediction[0][0]), 2)

        # ==========================================
        # DECISION LOGIC: Who wins?
        # Research states ML is better for tabular data, so we set it as the primary 'Recommended' result.
        # But we pass both so the teacher can see the comparison.
        # ==========================================
        
        return {
            'ML_AQI': ml_result,
            'DL_AQI': dl_result,
            'Recommended_AQI': ml_result, 
            'Status': 'Success'
        }

if __name__=="__main__":
    # Test Run: Checking if both brains are responding
    engine = AQIPredictor()
    sample_input = {
        'PM2.5': 250.0,
        'PM10': 400.0,
        'NO2': 80.0,
        'CO': 2.5,
        'SO2': 11.0,
        'Month': 11,
        'Day': 15,
        'DayOfWeek': 3,
        'AQI_Lag1': 380.0,
        'City': 'Delhi'
    }

    print("\n[TEST PERFORMING] Sending data to both AI Engines...")
    result = engine.predict(sample_input)
    
    print("\n" + "="*40)
    print("      Dual-Engine Output Report      ")
    print("="*40)
    print(f"Classical ML Output (Champion): {result['ML_AQI']}")
    print(f"Deep Learning Output (Challenger): {result['DL_AQI']}")
    print("="*40 + "\n")