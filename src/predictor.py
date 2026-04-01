import joblib
import pandas as pd 
import numpy as np
import os 

class AQIPredictor:
    def __init__(self, model_path='models/aqi_stacking_model.pkl', features_path='models/model_features.pkl'):
        print("--- EasternMartin Technologies: Initializing Prediction Engine ---")

        # Brain (Model) load karna 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AI Brain Missing at {model_path}! Train the model first.")
        
        self.model = joblib.load(model_path)
        self.model_features = joblib.load(features_path)
        print("[SUCCESS] AI Brain and Features Structure Loaded.")
    
    def predict(self, input_data):
        """
        Input: Dictonary of features (PM2.5, PM10, City, etc.)
        Output: Predicted AQI
        """

        # dictonary ko data frame me convert karna hai 
        df_input = pd.DataFrame([input_data])

        # Encoding: Training ke waqt jo city columns bane the, wahi yahan bhi cahiye 
        df_input = pd.get_dummies(df_input)

        # Missing columns ko 0 se bharna (jo columns trainning me the par input me nhi hai)
        for col in self.model_features:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # columns ko usi sequence me set karna jo trainning ke waqt tha 
        df_input = df_input[self.model_features]

        # final Prediction
        prediction = self.model.predict(df_input)
        return round(prediction[0], 2)

if __name__=="__main__":
    # test run: Kya humara brain response de raha hai ?
    # hume wahi columns dene honge jo tranning me use hue 
    engine = AQIPredictor()
    sample_input = {
        'PM2.5': 80.5,
        'PM10': 120.0,
        'NO2': 45.2,
        'CO': 1.2,
        'SO2': 10.5,
        'Month': 4,
        'DayOfWeek': 2,
        'AQI_Lag1': 150.0,
        'City': 'Delhi'
    }

    result = engine.predict(sample_input)
    print(f"[TEST PERFORMING] Predicted AQI for sample input: {result}")