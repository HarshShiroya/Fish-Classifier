
from flask import Flask, request, render_template
import pickle
import joblib

app = Flask(__name__)

# Load the model
classifier = joblib.load("models/fish_species_classifier.pkl")
sc = joblib.load("models/scaler.pkl")

@app.route('/')
def home():
    return render_template("input.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming the form data is numerical and directly usable for prediction
    # Extract input features from the form
    input_features = [float(x) for x in request.form.values()]

    new_data_scaled = sc.transform([input_features])
    prediction = classifier.predict(new_data_scaled)
    
    # Modify as necessary to fit the model's output
    return render_template("result.html", prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
