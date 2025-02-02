from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Initialize the Flask app
app = Flask(__name__)


# Load the saved model and scaler
model = pickle.load(open('accident_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json(force=True)


        # Prepare the input data for prediction (assuming it's a dictionary)
        features = [
            data["Traffic_Density"],
            data["Speed_Limit"],
            data["Number_of_Vehicles"],
            data["Driver_Alcohol"],
            data["Driver_Age"],
            data["Driver_Experience"]
        ]
       
        # Convert the features into a numpy array
        features = np.array(features).reshape(1, -1)


        # Scale the features
        features = scaler.transform(features)


        # Predict using the model
        prediction = model.predict(features)


        # Return the prediction as a JSON response
        return jsonify({'Prediction': int(prediction[0])})


    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Set the custom IP address and port here
    app.run(host='0.0.0.0', port=5000, debug=True)
