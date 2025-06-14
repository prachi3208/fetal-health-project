from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess the dataset
df = pd.read_csv('fetal_health.csv')
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Train models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Get model from request (default: logistic)
    model_name = data.get('model', 'logistic').lower()
    
    # Extract feature values in correct order
    try:
        input_features = np.array([[
            data['baseline value'],
            data['accelerations'],
            data['fetal_movement'],
            data['uterine_contractions'],
            data['light_decelerations'],
            data['severe_decelerations'],
            data['prolongued_decelerations'],
            data['abnormal_short_term_variability'],
            data['mean_value_of_short_term_variability'],
            data['percentage_of_time_with_abnormal_long_term_variability'],
            data['mean_value_of_long_term_variability'],
            data['histogram_width'],
            data['histogram_min'],
            data['histogram_max'],
            data['histogram_number_of_peaks'],
            data['histogram_number_of_zeroes'],
            data['histogram_mode'],
            data['histogram_mean'],
            data['histogram_median'],
            data['histogram_variance'],
            data['histogram_tendency']
        ]])
    except KeyError as e:
        return jsonify({'error': f'Missing input feature: {str(e)}'}), 400

    # Scale input
    input_scaled = scaler.transform(input_features)

    # Model selection
    if model_name == 'logistic':
        model = log_model
    elif model_name == 'knn':
        model = knn_model
    elif model_name == 'randomforest':
        model = rf_model
    else:
        return jsonify({'error': 'Invalid model name. Choose from logistic, knn, randomforest'}), 400

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    health_map = {
        1.0: 'Normal',
        2.0: 'Suspect',
        3.0: 'Pathological'
    }

    return jsonify({
        'prediction': int(prediction),
        'health_status': health_map.get(prediction, "Unknown")
    })


if __name__ == '__main__':
    app.run(debug=True)
