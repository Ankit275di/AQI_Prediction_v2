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

def train_model(input_path='data/processed/features_data.csv', model_dir='models/'):
    print("--- EasternMartin Technologies: AI Training Engine ---")

    # load data 
    if not os.path.exists(input_path):
        print(f"[ERROR] Features data not found at {input_path}. Run features.py first.")
        return
    
    print("[INFO] Loading feature-engineered dataset...")
    df = pd.read_csv(input_path)

    # prepare features x and target y
    # target is what we want to predict (AQI)
    y = df['AQI']

    # features are everything else. we drop 'date' because ML models need numbers, and we already extracted month/DateofWeek in the previous step.
    # X = df.drop(columns=['AQI', 'Date']) --> off machine
    columns_to_drop = ['AQI', 'Date']
    if 'AQI_Bucket' in df.columns:
        columns_to_drop.append('AQI_Bucket')
    X = df.drop(columns=columns_to_drop)

    # one hot encoding converts cities into binary columns (0s and 1s)
    # AI Models dont understand text like city names 
    print("[INFO] Encoding categorical variables (City) ...")
    X = pd.get_dummies(X, columns=['City'], drop_first=True)

    # ---------- yeah new edit hai ---------------------
    import re
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    # --------------------------------------------------

    # split data into training (80%) and testing(20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Training AI on {X_train.shape[0]} rows. Testing on {X_test.shape[0]} rows.")

    # define the stacking architecture
    print("[INFO] Initilizing Stacking Regressor (Linear Regression + XGBoost + LightGBmM + Random Forest Regression)...")
    base_models = [
        ('lr', LinearRegression (n_jobs=-1)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ] 

    # the meta model that learns how to best combine the base models 
    meta_model = Ridge()

    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=3)

    # train the model
    print("\n[PROCESS] Traning the AI ... (Your Ryzen processor is handling the heavy lifting now. Please wait...)")
    stacking_model.fit(X_train, y_train)

    # evaluate model performance 
    print("\n[PROCESS] Evaluating Model Accuracy...")
    predictions = stacking_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + " " * 30)
    print("    Model Health Report     ")
    print("="*30)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R2 Score                 : {r2:.4f}")
    print("="*30 + "\n")

    # Save the Brain and Column Structure
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'aqi_stacking_model.pkl')
    features_path = os.path.join(model_dir, 'model_features.pkl')

    # save the actual model 
    joblib.dump(stacking_model, model_path)
    # save the exact column names so our web app inputs match the tranning inputs 
    joblib.dump(list(X_train.columns), features_path)

    print(f"[SUCCESS] AI Brain saved Securely at: {model_path}")
    print(f"[SUCCESS] Feature structure saved at: {features_path}")

if __name__=="__main__":
    train_model()