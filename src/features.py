import pandas as pd
import os 

def engineering_features(input_path='data/processed/city_day_clean.csv', output_path='data/processed/features_data.csv'):
    print(f"---EasternMartin Technologies: Feature Engineering Engine ---")

    # 1. File Check 
    if not os.path.exists(input_path):
        print(f"[ERROR] Clean data not found at {input_path}. Run data_purifier.py first.")
        return False
    
    print(f"[INFO] Loading purified data ...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # CRITICAL: Time-series data ko humeasha date and city ke according se sort karna cahiye 
    df = df.sort_values(by=['City', 'Date'])

    # 2. Temporal Features (WAqt ka Hisab)
    print("[INFO] Extracting Temporal Features (Month, DayOfWeek)...")
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # pichle din ke main pollutants 
    important_pollutants = ['PM2.5', 'PM10', 'NO2', 'CO']
    for col in important_pollutants:
        if col in df.columns:
            df[f'{col}_Lag1'] = df.groupby('City')[col].shift(1)

    # rolling averages (trend catch karne ke liye)
    # pichle 3 dino ka average nikalange taaki sudden spikes filter ho jayen 
    print("[INFO] Gathering Rolling Averages (3-day window)...")
    for col in important_pollutants:
        if col in df.columns:
            df['f{col}_Roll3'] = df.groupby('City')[col].transform(lambda x:x.rolling(window=3, min_periods=1).mean())
    
    # 5. clean up NaNs created by Shifting 
    # kyuki shift karne se pahle din ka data khali (NaN) ho jata hai, usey drop karna zaroori hai 
    initial_rows = len(df)
    df = df.dropna()
    print(f'[INFO] Dropped {initial_rows - len(df)} rows due to Lag initilization.')

    # save the feature - rich dataset 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Feature-engineered data saved to: {output_path}")
    print(f"[INFO] Final dataset shape for training: {df.shape}")
    return True

if __name__=="__main__":
    engineering_features()