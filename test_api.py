"""
Test script for Flask API
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.json()}")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    test_student = {
        "age": 17,
        "sex": "M",
        "address": "U",
        "school": "GP",
        "G1": 14,
        "G2": 15,
        "failures": 0,
        "studytime": 3,
        "absences": 5,
        "internet": "yes",
        "activities": "yes",
        "famsup": "yes"
    }
    
    response = requests.post(f"{API_URL}/predict", json=test_student)
    print(f"\nSingle Prediction Result:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint"""
    test_batch = {
        "students": [
            {"student_id": "S001", "age": 16, "G1": 18, "G2": 19, "failures": 0, "absences": 2},
            {"student_id": "S002", "age": 17, "G1": 8, "G2": 7, "failures": 2, "absences": 15},
            {"student_id": "S003", "age": 18, "G1": 12, "G2": 11, "failures": 1, "absences": 8}
        ]
    }
    
    response = requests.post(f"{API_URL}/predict_batch", json=test_batch)
    print(f"\nBatch Prediction Result:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_feature_importance():
    """Test feature importance endpoint"""
    response = requests.get(f"{API_URL}/feature_importance")
    print(f"\nFeature Importance:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing Student Performance Prediction API")
    print("="*50)
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Feature Importance", test_feature_importance)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")