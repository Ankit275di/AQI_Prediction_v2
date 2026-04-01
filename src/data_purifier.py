import pandas as pd 
import os

def purify_data(input_path='data/raw/city_day.csv', output_path='data/processed/city_day_clean.csv'):
    print(f"--- EasternMartin Technologies: Data Purifiaction Engine ---")

    # check if raw data exists 
    if not os.path.exists(input_path):
        print(f"[ERROR] Raw data not found at {input_path}")
        print(f"Please ensure DATASET is inside the {input_path} folder.")
        return False
    
    df = pd.read_csv(input_path)
    print(f"[INFO] Initial dataset loaded. Rows: {len(df)}")

    # 2. Target Variable Cleanup
    # we cannot train a model if the actual AQI (answer) is missing
    initial_len = len(df)
    df = df.dropna(subset=['AQI'])
    print(f"[INFO] Dropped {initial_len - len(df)} rows missing the target AQI.")

    # Standradize the time for sequence handling 
    df['Date'] = pd.to_datetime(df['Date'])
    df=df.sort_values(by=['City', 'Date'])
    
    # filling the blank areas or cols
    # We'll use linear interpolation grouped by city so delhi's data dosent mix with mumbai's
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

    print('[INFO] Applying linear interpolation to missing chemical values...')
    for col in pollutants:
        if col in df.columns:
            df[col] = df.groupby('City')[col].transform(lambda x:x.interpolate(method='linear', limit=3))

            # for remaning massive gaps, fill with the city's mdeian value for that chemical 
            df[col]=df.groupby('City')[col].transform(lambda x:x.fillna(x.median()))

    # 4. save the purified data 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Purified data saved to: {output_path}")
    print(f"[INFO] Final clean rows ready for feature engineering: {len(df)}")
    return True

if __name__ == "__main__":
    purify_data() 