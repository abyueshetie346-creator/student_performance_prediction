#!/bin/bash

echo "=========================================="
echo "Student Performance Prediction System"
echo "=========================================="

# Create directories
mkdir -p data models

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Train models
echo "🤖 Training models..."
python train_model.py

# Ask user which interface to launch
echo ""
echo "Select deployment option:"
echo "1) Streamlit Web App (Recommended)"
echo "2) Flask API"
echo "3) Both"
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo "🚀 Launching Streamlit app..."
        streamlit run app.py
        ;;
    2)
        echo "🚀 Launching Flask API..."
        python flask_api.py
        ;;
    3)
        echo "🚀 Launching both interfaces..."
        streamlit run app.py --server.port 8501 &
        python flask_api.py &
        echo "Streamlit: http://localhost:8501"
        echo "Flask API: http://localhost:5000"
        wait
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac