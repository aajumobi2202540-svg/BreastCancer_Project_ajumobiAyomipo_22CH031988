from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/house_price_model.pkl", "rb"))
le = pickle.load(open("model/label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    
    overallqual = int(data["OverallQual"])
    grlivarea = float(data["GrLivArea"])
    totalbsmtsf = float(data["TotalBsmtSF"])
    garagecars = int(data["GarageCars"])
    yearbuilt = int(data["YearBuilt"])
    neighborhood = le.transform([data["Neighborhood"]])[0]

    features = np.array([[overallqual, grlivarea, totalbsmtsf,
                           garagecars, yearbuilt, neighborhood]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction=f"₦{prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
