"""
Student Performance Prediction - Flask API
RESTful API for model predictions
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
def load_models():
    """Load trained models and preprocessors"""
    try:
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/label_encoders.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return model, scaler, encoders, feature_cols
    except FileNotFoundError:
        return None, None, None, None

model, scaler, encoders, feature_cols = load_models()

def prepare_features(data_dict):
    """Convert input data to model-ready format"""
    input_df = pd.DataFrame([data_dict])
    
    # Encode categorical variables
    for col, encoder in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col])
            except:
                input_df[col] = 0
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Select and order features
    input_df = input_df[feature_cols]
    
    # Scale numerical features
    numerical_cols = input_df.select_dtypes(include=[np.number]).columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    return input_df

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        'name': 'Student Performance Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Make a single prediction',
            '/predict_batch': 'POST - Make batch predictions',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Make a single prediction"""
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare features
        features_df = prepare_features(data)
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0].tolist()
        
        # Map prediction to label
        labels = ['At Risk', 'Moderate', 'High Achiever']
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'prediction_label': labels[prediction],
            'probabilities': {
                'At Risk': probabilities[0],
                'Moderate': probabilities[1],
                'High Achiever': probabilities[2]
            },
            'confidence': max(probabilities)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Make batch predictions"""
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'students' not in data:
            return jsonify({'error': 'No students data provided'}), 400
        
        students = data['students']
        results = []
        labels = ['At Risk', 'Moderate', 'High Achiever']
        
        for student in students:
            features_df = prepare_features(student)
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0].tolist()
            
            results.append({
                'student_id': student.get('student_id', 'unknown'),
                'prediction': int(prediction),
                'prediction_label': labels[prediction],
                'confidence': max(probabilities)
            })
        
        # Summary statistics
        predictions = [r['prediction'] for r in results]
        
        return jsonify({
            'success': True,
            'total_students': len(results),
            'summary': {
                'At Risk': predictions.count(0),
                'Moderate': predictions.count(1),
                'High Achiever': predictions.count(2)
            },
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """Get feature importance"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = feature_cols
            
            # Sort by importance
            sorted_idx = np.argsort(importance)[::-1]
            
            top_features = []
            for i in range(min(10, len(sorted_idx))):
                top_features.append({
                    'feature': features[sorted_idx[i]],
                    'importance': float(importance[sorted_idx[i]])
                })
            
            return jsonify({
                'success': True,
                'feature_importance': top_features
            })
        else:
            return jsonify({'error': 'Model does not provide feature importance'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if model is None:
        print("❌ Models not loaded! Please run 'python train_model.py' first.")
    else:
        print("✅ Models loaded successfully!")
        print("🚀 Starting Flask API server...")
        print("📍 API available at: http://localhost:5000")
        print("📝 Test with: curl http://localhost:5000/health")
        app.run(debug=True, host='0.0.0.0', port=5000)