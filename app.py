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
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables to store models
linear_model = None
rf_model = None
poly_model = None
poly_features = None
label_encoders = {}
expected_categories = {}

# USD to INR conversion rate (you can update this or fetch from an API)
USD_TO_INR_RATE = 83.0  # Approximate rate, update as needed

def load_and_train_models():
    """Load data and train all models with better error handling"""
    global linear_model, rf_model, poly_model, poly_features, label_encoders, expected_categories
    
    try:
        # Check if insurance.csv exists
        if not os.path.exists('insurance.csv'):
            logger.error("insurance.csv file not found!")
            return False
            
        # Load data
        df = pd.read_csv('insurance.csv')
        logger.info(f"Loaded data with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Store original data for encoding reference
        original_df = df.copy()
        
        # Store expected categories for validation
        expected_categories = {
            'sex': df['sex'].unique().tolist(),
            'smoker': df['smoker'].unique().tolist(),
            'region': df['region'].unique().tolist()
        }
        logger.info(f"Expected categories: {expected_categories}")
        
        # Convert categorical columns
        df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')
        
        # Label encoding with better error handling
        for col in ['sex', 'smoker', 'region']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            logger.info(f"Label encoder for {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
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
        
        logger.info("All models trained successfully!")
        
        # Print model scores for verification
        logger.info(f"Linear Regression R2 Score: {linear_model.score(x_test_linear, y_test):.3f}")
        rf_pred = rf_model.predict(x_test_linear)
        logger.info(f"Random Forest R2 Score: {metrics.r2_score(y_test, rf_pred):.3f}")
        logger.info(f"Polynomial Regression R2 Score: {poly_model.score(x_test_poly, y_test_poly):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in load_and_train_models: {str(e)}")
        return False

def validate_input(age, sex, bmi, children, smoker, region):
    """Validate input parameters"""
    errors = []
    
    # Validate age
    if not (18 <= age <= 100):
        errors.append("Age must be between 18 and 100")
    
    # Validate sex
    if sex not in expected_categories.get('sex', ['male', 'female']):
        errors.append(f"Sex must be one of: {expected_categories.get('sex', ['male', 'female'])}")
    
    # Validate BMI
    if not (15 <= bmi <= 50):
        errors.append("BMI must be between 15 and 50")
    
    # Validate children
    if not (0 <= children <= 10):
        errors.append("Number of children must be between 0 and 10")
    
    # Validate smoker
    if smoker not in expected_categories.get('smoker', ['yes', 'no']):
        errors.append(f"Smoker must be one of: {expected_categories.get('smoker', ['yes', 'no'])}")
    
    # Validate region
    if region not in expected_categories.get('region', ['northeast', 'northwest', 'southeast', 'southwest']):
        errors.append(f"Region must be one of: {expected_categories.get('region', ['northeast', 'northwest', 'southeast', 'southwest'])}")
    
    return errors

def safe_label_encode(value, column):
    """Safely encode a value using the label encoder"""
    try:
        if column not in label_encoders:
            raise ValueError(f"No label encoder found for column: {column}")
        
        encoder = label_encoders[column]
        
        # Check if the value is in the encoder's classes
        if value not in encoder.classes_:
            logger.warning(f"Value '{value}' not found in encoder classes for {column}: {encoder.classes_}")
            # Return the encoding for the first class as a fallback
            return encoder.transform([encoder.classes_[0]])[0]
        
        return encoder.transform([value])[0]
    except Exception as e:
        logger.error(f"Error encoding {value} for column {column}: {str(e)}")
        raise

def convert_usd_to_inr(usd_amount):
    """Convert USD to INR"""
    return usd_amount * USD_TO_INR_RATE

@app.route('/')
def index():
    # Check if models are loaded
    if linear_model is None:
        success = load_and_train_models()
        if not success:
            return render_template('error.html', error="Failed to load machine learning models. Please ensure insurance.csv is available.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if models are loaded
        if linear_model is None:
            return render_template('error.html', error="Models not loaded. Please refresh the page.")
        
        # Get form data with validation
        try:
            age = float(request.form['age'])
            sex = request.form['sex'].lower().strip()
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = request.form['smoker'].lower().strip()
            region = request.form['region'].lower().strip()
            model_type = request.form['model_type']
        except (ValueError, KeyError) as e:
            return render_template('error.html', error=f"Invalid input data: {str(e)}")
        
        logger.info(f"Received input: age={age}, sex={sex}, bmi={bmi}, children={children}, smoker={smoker}, region={region}, model={model_type}")
        
        # Validate input
        validation_errors = validate_input(age, sex, bmi, children, smoker, region)
        if validation_errors:
            return render_template('error.html', error="Input validation failed: " + "; ".join(validation_errors))
        
        # Encode categorical variables safely
        try:
            sex_encoded = safe_label_encode(sex, 'sex')
            smoker_encoded = safe_label_encode(smoker, 'smoker')
            region_encoded = safe_label_encode(region, 'region')
        except Exception as e:
            return render_template('error.html', error=f"Error encoding categorical variables: {str(e)}")
        
        logger.info(f"Encoded values: sex={sex_encoded}, smoker={smoker_encoded}, region={region_encoded}")
        
        # Make predictions based on selected model
        try:
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
            else:
                return render_template('error.html', error=f"Invalid model type: {model_type}")
        
        except Exception as e:
            return render_template('error.html', error=f"Error making prediction: {str(e)}")
        
        # Convert USD to INR
        prediction_inr = convert_usd_to_inr(prediction_usd)
        
        logger.info(f"Prediction successful: USD=${prediction_usd:.2f}, INR=₹{prediction_inr:.2f}")
        
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
        logger.error(f"Unexpected error in predict: {str(e)}")
        return render_template('error.html', error=f"An unexpected error occurred: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        # Check if models are loaded
        if linear_model is None:
            return jsonify({
                'error': 'Models not loaded',
                'success': False
            })
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'success': False
            })
        
        # Extract and validate data
        try:
            age = float(data['age'])
            sex = data['sex'].lower().strip()
            bmi = float(data['bmi'])
            children = int(data['children'])
            smoker = data['smoker'].lower().strip()
            region = data['region'].lower().strip()
            model_type = data.get('model_type', 'linear')
        except (ValueError, KeyError) as e:
            return jsonify({
                'error': f'Invalid input data: {str(e)}',
                'success': False
            })
        
        # Validate input
        validation_errors = validate_input(age, sex, bmi, children, smoker, region)
        if validation_errors:
            return jsonify({
                'error': 'Input validation failed: ' + '; '.join(validation_errors),
                'success': False
            })
        
        # Encode categorical variables
        try:
            sex_encoded = safe_label_encode(sex, 'sex')
            smoker_encoded = safe_label_encode(smoker, 'smoker')
            region_encoded = safe_label_encode(region, 'region')
        except Exception as e:
            return jsonify({
                'error': f'Error encoding categorical variables: {str(e)}',
                'success': False
            })
        
        # Make prediction
        try:
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
            else:
                return jsonify({
                    'error': f'Invalid model type: {model_type}',
                    'success': False
                })
        except Exception as e:
            return jsonify({
                'error': f'Error making prediction: {str(e)}',
                'success': False
            })
        
        # Convert to INR
        prediction_inr = convert_usd_to_inr(prediction_usd)
        
        return jsonify({
            'prediction_usd': round(prediction_usd, 2),
            'prediction_inr': round(prediction_inr, 2),
            'model_used': model_type,
            'success': True
        })
    
    except Exception as e:
        logger.error(f"Unexpected error in api_predict: {str(e)}")
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}',
            'success': False
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        models_loaded = all([
            linear_model is not None,
            rf_model is not None,
            poly_model is not None,
            poly_features is not None,
            len(label_encoders) == 3
        ])
        
        return jsonify({
            'status': 'healthy' if models_loaded else 'unhealthy',
            'models_loaded': models_loaded,
            'expected_categories': expected_categories
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    # Load and train models when the app starts
    logger.info("Starting Flask application...")
    success = load_and_train_models()
    if not success:
        logger.error("Failed to load models. The application may not work correctly.")
    else:
        logger.info("Models loaded successfully!")
    
    app.run(debug=True)