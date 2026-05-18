from flask import Flask, render_template, request
from src.predictor import AQIPredictor
import datetime
import traceback

# initilizing the flask app
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

# initilize the AQI Engine 
predictor = AQIPredictor()

@app.route('/')
def home():
    # get current date info for ui
    now = datetime.datetime.now()
    context = {
        "day": now.strftime("%A"),
        "date": now.strftime("%d %b %y")
    }
    return render_template('index.html', **context)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Form se data lena (Ozone O3 is here)
        input_data = {
            'City': request.form.get('city'),
            'AQI_Lag1': float(request.form.get('lag1', 0)),
            'PM2.5': float(request.form.get('pm25', 0)),
            'PM10': float(request.form.get('pm10', 0)),
            'NO2': float(request.form.get('no2', 0)),
            'CO': float(request.form.get('co', 0)),
            'SO2': float(request.form.get('so2', 0)),
            'O3': float(request.form.get('o3', 0))
        }

        # 2. Engine se dictionary receive karna (FIXED: changed 'engine' to 'predictor')
        results = predictor.predict(input_data)
        
        # 3. Dictionary se exact numbers nikalna aur Float mein convert karna (FIXED)
        final_aqi = float(results['Recommended_AQI'])
        dl_aqi = float(results['DL_AQI'])

        # 4. Category logic
        if final_aqi <= 50:
            category = "Good"
        elif final_aqi <= 100:
            category = "Moderate"
        elif final_aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif final_aqi <= 200:
            category = "Unhealthy"
        elif final_aqi <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"

        # (Optional) Mock trend data for the chart, until LSTM backend is fully connected
        mock_trend = [round(final_aqi + (i * 2.5), 2) for i in range(12)]

        # 5. UI ko data bhejna
        return render_template('index.html', 
                               prediction=final_aqi, 
                               category=category, 
                               dl_prediction=dl_aqi,
                               trend_data=mock_trend)

    except Exception as e:
        # Deep Error Logging
        error_trace = traceback.format_exc()
        print("\n--- [CRITICAL ERROR TRACE] ---")
        print(error_trace)
        print("------------------------------\n")
        return f"Error in Prediction: {str(e)} <br><br> <b>Check VS Code Terminal for exact line number!</b>"

if __name__ == "__main__":
    print("[SYSTEM] Starting Local Development Server...")
    app.run(debug=True, port=5000)