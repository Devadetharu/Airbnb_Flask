from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    This function takes JSON input, processes it into the correct format,
    runs it through the model, and returns the prediction.
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract the feature values (example assumes 5 features are required)
        # Ensure the keys match the features used to train the model
        required_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        input_data = []

        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing value for required feature: {feature}'}), 400
            input_data.append(data[feature])

        # Convert input data to a numpy array
        input_array = np.array([input_data])  # Shape: (1, num_features)

        # Make prediction
        prediction = model.predict(input_array)

        # Return the prediction
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Example health-check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return "The model server is running!", 200


# Run the app
if __name__ == '__main__':
    app.run(debug=True)