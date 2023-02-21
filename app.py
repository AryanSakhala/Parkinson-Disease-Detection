from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

encoder = LabelEncoder()



data = pd.read_csv('Crop_recommendation.csv')
y = data['label']
encoder.fit(y)

# Initialize the Flask application
app = Flask(__name__)

model = load_model('my_model.h5')

# Define a function to make predictions
def make_predictions(N, P, K, temperature, humidity, ph, rainfall):
    # Create a numpy array from the user's inputs
    user_inputs = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Predict the label of the new data point
    y_pred_prob = model.predict(user_inputs)

    # Find the index of the predicted class
    y_pred = np.argmax(y_pred_prob, axis=-1)

    # Decode the label
    label = encoder.classes_[y_pred]
    
    return label

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user's inputs from the form
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # Make a prediction using the model
    prediction = make_predictions(N, P, K, temperature, humidity, ph, rainfall)
    
    # Render the results template with the prediction
    return render_template('result.html', prediction=prediction)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
