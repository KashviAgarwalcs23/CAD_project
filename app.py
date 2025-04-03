from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cad_model.pkl')

# Load model (only once)
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise SystemExit(1)  # Stop the app if model fails to load

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Age': float(request.form['age']),
            'Sex': request.form['sex'],
            'BMI': float(request.form['bmi']),
            'DM': int(request.form.get('dm', 0)),
            'HTN': int(request.form.get('htn', 0)),
            'Current Smoker': int(request.form.get('smoker', 0)),
            'LDL': float(request.form['ldl']),
            'HDL': float(request.form['hdl']),
            'EF-TTE': float(request.form['ef']),
            'VHD': request.form['vhd'],
            'Total_Cholesterol': float(request.form['ldl']) + float(request.form['hdl']),
            'Cholesterol_Ratio': float(request.form['ldl']) / max(0.01, float(request.form['hdl'])),
            'BP_BMI_Interaction': (float(request.form['bp']) * float(request.form['bmi'])) / 100
        }
        
        # Predict
        df = pd.DataFrame([data])
        processed_data = preprocessor.transform(df)
        risk_score = model.predict_proba(processed_data)[0][1]
        
        return jsonify({
            'risk': f"{risk_score:.1%}",
            'interpretation': "High risk" if risk_score > 0.7 else "Moderate risk" if risk_score > 0.3 else "Low risk"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
