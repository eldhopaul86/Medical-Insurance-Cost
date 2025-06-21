from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn import metrics
import pickle
import os

app = Flask(__name__)

# Global variables to store models
linear_model = None
rf_model = None
poly_model = None
poly_features = None
label_encoders = {}

# USD to INR conversion rate (you can update this or fetch from an API)
USD_TO_INR_RATE = 83.0  # Approximate rate, update as needed

def load_and_train_models():
    """Load data and train all models"""
    global linear_model, rf_model, poly_model, poly_features, label_encoders
    
    # Load data
    df = pd.read_csv('insurance.csv')
    
    # Store original data for encoding reference
    original_df = df.copy()
    
    # Convert categorical columns
    df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')
    
    # Label encoding
    for col in ['sex', 'smoker', 'region']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare data for Linear Regression
    x_linear = df.drop(['charges'], axis=1)
    y = df['charges']
    x_train_linear, x_test_linear, y_train, y_test = train_test_split(x_linear, y, random_state=42, test_size=0.2)
    
    # Train Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(x_train_linear, y_train)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=4, random_state=1, n_jobs=-1)
    rf_model.fit(x_train_linear, y_train)
    
    # Prepare data for Polynomial Regression (excluding sex and region as in original code)
    x_poly = df.drop(['charges', 'sex', 'region'], axis=1)
    poly_features = PolynomialFeatures(degree=2)
    x_poly_transformed = poly_features.fit_transform(x_poly)
    x_train_poly, x_test_poly, y_train_poly, y_test_poly = train_test_split(x_poly_transformed, y, test_size=0.2, random_state=0)
    
    # Train Polynomial Regression
    poly_model = LinearRegression()
    poly_model.fit(x_train_poly, y_train_poly)
    
    print("All models trained successfully!")
    
    # Print model scores for verification
    print(f"Linear Regression R2 Score: {linear_model.score(x_test_linear, y_test):.3f}")
    rf_pred = rf_model.predict(x_test_linear)
    print(f"Random Forest R2 Score: {metrics.r2_score(y_test, rf_pred):.3f}")
    print(f"Polynomial Regression R2 Score: {poly_model.score(x_test_poly, y_test_poly):.3f}")

def convert_usd_to_inr(usd_amount):
    """Convert USD to INR"""
    return usd_amount * USD_TO_INR_RATE

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        model_type = request.form['model_type']
        
        # Encode categorical variables
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
        region_encoded = label_encoders['region'].transform([region])[0]
        
        # Make predictions based on selected model
        if model_type == 'linear':
            # Linear Regression prediction
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
            prediction_usd = linear_model.predict(input_data)[0]
            model_name = "Linear Regression"
            
        elif model_type == 'random_forest':
            # Random Forest prediction
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
            prediction_usd = rf_model.predict(input_data)[0]
            model_name = "Random Forest"
            
        elif model_type == 'polynomial':
            # Polynomial Regression prediction (only age, bmi, children, smoker)
            input_data = np.array([[age, bmi, children, smoker_encoded]])
            input_poly = poly_features.transform(input_data)
            prediction_usd = poly_model.predict(input_poly)[0]
            model_name = "Polynomial Regression"
        
        # Convert USD to INR
        prediction_inr = convert_usd_to_inr(prediction_usd)
        
        return render_template('result.html', 
                             prediction=round(prediction_usd, 2),
                             prediction_inr=f"₹{round(prediction_inr, 2):,}",
                             model_name=model_name,
                             input_data={
                                 'age': age,
                                 'sex': sex,
                                 'bmi': bmi,
                                 'children': children,
                                 'smoker': smoker,
                                 'region': region
                             })
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        age = float(data['age'])
        sex = data['sex']
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker']
        region = data['region']
        model_type = data.get('model_type', 'linear')
        
        # Encode categorical variables
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
        region_encoded = label_encoders['region'].transform([region])[0]
        
        # Make prediction
        if model_type == 'linear':
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
            prediction_usd = linear_model.predict(input_data)[0]
        elif model_type == 'random_forest':
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
            prediction_usd = rf_model.predict(input_data)[0]
        elif model_type == 'polynomial':
            input_data = np.array([[age, bmi, children, smoker_encoded]])
            input_poly = poly_features.transform(input_data)
            prediction_usd = poly_model.predict(input_poly)[0]
        
        # Convert to INR
        prediction_inr = convert_usd_to_inr(prediction_usd)
        
        return jsonify({
            'prediction_usd': round(prediction_usd, 2),
            'prediction_inr': round(prediction_inr, 2),
            'model_used': model_type,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == '__main__':
    # Load and train models when the app starts
    load_and_train_models()
    app.run(debug=True)