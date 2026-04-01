from flask import Flask, render_template
from src.predictor import AQIPredictor
import datetime

# initilizing the flask app
app = Flask(__name__, template_folder='wb/templates', static_folder='web.static')

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

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # get data from the form 
        data = {
            'City': request.form.get('city'),
            'PM2.5': float(request.form.get('pm2.5')),
            'PM10': float(request.form.get('pm10')),
            'NO2': float(request.form.get('no2')),
            'CO': float(request.form.get('co')),
            'SO2': float(request.form.get('so2')),
            'AQI_Lag1': float(request.form.get('lag1')),
            'Month': datetime.datetime.now().month,
            'Day': datetime.datetime.now().day,
            'DayOfWeek': datetime.datetime.now().weekday()
        }

        # fire the AI Brain
        prediction = predictor.predict(data)

        # determine color category 
        color = "#2ecc71" # good category
        status = "Good"
        if prediction > 50: color, status = "#f1c40f", "Satisfactory"
        if prediction > 100: color, status = "#e67e22", "Moderate"
        if prediction > 200: color, status = "#e74c3c", "Poor"
        if prediction > 300: color, status = "#9b59b6", "Very Poor"
        if prediction > 400: color, status = "#7f8c8d", "Severe"

        return render_template('index.html', 
                               prediction=prediction,
                               status=status,
                               color=color,
                               city=data['City'])
    except Exception as e:
        return f"Error in Prediction: {str(e)}"


if __name__ == "__main__":
    print("[SYSTEM] Starting Local Development Server...")
    app.run(debug=True, port=5000)
