from flask import Falsk, render_template

# initilizing the flask app
app = Flask(__name__, template_folder='wb/templates', static_folder='web.static')

@app.route('/')
def home():
    # we gonna build the actual HTML later
    return "<h1>EasternMartin Technologies: AQI Predictor Engine V2 is Online</h1>"

if __name__ == "__main__":
    print("[SYSTEM] Starting Local Development Server...")
    app.run(debug=True, port=5000)
