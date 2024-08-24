from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = joblib.load('iris_classifier.pkl')  
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy',allow_pickle=True) 


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
   
    features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    prediction = model.predict(features)
    predicted_species = label_encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f'Predicted Species: {predicted_species}')

if __name__ == '__main__':
    app.run(debug=True)
