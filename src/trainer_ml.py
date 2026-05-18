import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import os
import re

def train_model(input_path='data/processed/features_data.csv', model_dir='models/'):
    print("--- EasternMartin Technologies: ML Training Engine (Track A) ---")

    # 1. Load data 
    if not os.path.exists(input_path):
        print(f"[ERROR] Features data not found at {input_path}.")
        return
    
    print("[INFO] Loading feature-engineered dataset...")
    df = pd.read_csv(input_path)

    # 2. Prepare features X and target y
    y = df['AQI']
    columns_to_drop = ['AQI', 'Date']
    if 'AQI_Bucket' in df.columns:
        columns_to_drop.append('AQI_Bucket')
    X = df.drop(columns=columns_to_drop)

    # 3. One-hot encoding (Cities)
    print("[INFO] Encoding categorical variables (City)...")
    X = pd.get_dummies(X, columns=['City'], drop_first=True)
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # 4. The 70/15/15 Split Logic (Faculty Demand)
    print("[INFO] Executing 70-15-15 Data Split...")
    # Pehle 30% data alag nikalte hain (Validation + Test ke liye)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    # Ab us 30% ko aadha-aadha (15% Val, 15% Test) baant dete hain
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"       -> Training Set (70%): {X_train.shape[0]} rows")
    print(f"       -> Validation Set (15%): {X_val.shape[0]} rows")
    print(f"       -> Testing Set (15%): {X_test.shape[0]} rows")

    # 5. Define the stacking architecture
    print("\n[INFO] Initializing Stacking Regressor...")
    base_models = [
        ('lr', LinearRegression(n_jobs=-1)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ] 
    meta_model = Ridge()
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=3)

    # 6. Train the model
    print("\n[PROCESS] Training the ML Engine... (Your Ryzen is handling this. Please wait...)")
    stacking_model.fit(X_train, y_train)

    # 7. Evaluate model performance on Unseen Test Data (15%)
    print("\n[PROCESS] Evaluating Model Accuracy on Test Data...")
    predictions = stacking_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + " " * 30)
    print("    ML Model Health Report (Champion)    ")
    print("="*40)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R2 Score                 : {r2:.4f}")
    print("="*40 + "\n")

    # 8. Save the Brain and Column Structure
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'aqi_stacking_model.pkl')
    features_path = os.path.join(model_dir, 'model_features.pkl')

    joblib.dump(stacking_model, model_path)
    joblib.dump(list(X_train.columns), features_path)

    print(f"[SUCCESS] ML Brain saved Securely at: {model_path}")

if __name__=="__main__":
    train_model()