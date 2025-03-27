from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained logistic regression model
model = pickle.load(open('diabetes_logistic_regression_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        pregnancies = 0
        
        if gender.lower() == 'woman':
            pregnancies = int(request.form['pregnancies'])
        
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        # Prepare input data
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        # Get prediction
        prediction = model.predict(features)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        
        return render_template('index.html', result=result)
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
