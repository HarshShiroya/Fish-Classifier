
from flask import Flask, request, render_template
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

    # Extract input features from the form
    input_features = [float(x) for x in request.form.values()]

    # Scaling the input for the classifier
    new_data_scaled = sc.transform([input_features])
    prediction = classifier.predict(new_data_scaled)
    
    
    return render_template("result.html", prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
