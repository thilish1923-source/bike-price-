from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("D:/jai/bike_price/bike_price_model.pkl")

# Load dataset again to get unique options
df = pd.read_csv("D:/jai/bike_price/Used_Bikes.csv")
city_list = sorted(df['city'].unique())
owner_list = sorted(df['owner'].unique())
brand_list = sorted(df['brand'].unique())

@app.route('/')
def home():
    return render_template('index.html', city=city_list, owner=owner_list, brand=brand_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Take inputs from form
        city = request.form['city']
        owner = request.form['owner']
        brand = request.form['brand']
        kms_driven = float(request.form['kms_driven'])
        age = float(request.form['age'])
        power = float(request.form['power'])

        # Create a DataFrame for model input
        input_df = pd.DataFrame([{
            'city': city,
            'owner': owner,
            'brand': brand,
            'kms_driven': kms_driven,
            'age': age,
            'power': power
        }])

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_text = f"Estimated Bike Price: ₹{round(prediction, 2)}"

    except Exception as e:
        prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text, city=city_list, owner=owner_list, brand=brand_list)

if __name__ == "__main__":
    app.run(debug=True)
