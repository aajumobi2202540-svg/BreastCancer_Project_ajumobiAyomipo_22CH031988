from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model/breast_cancer_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

FEATURE_NAMES = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get inputs from user
    radius = float(request.form['radius_mean'])
    texture = float(request.form['texture_mean'])
    perimeter = float(request.form['perimeter_mean'])
    area = float(request.form['area_mean'])
    smoothness = float(request.form['smoothness_mean'])

    # Create DataFrame
    input_data = pd.DataFrame([[radius, texture, perimeter, area, smoothness]], columns=FEATURE_NAMES)
    
    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    result = "Benign" if prediction == 1 else "Malignant"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
