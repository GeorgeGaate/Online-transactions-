from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import joblib

# Load saved model artifact
model_artifact = joblib.load("model.joblib")
features = model_artifact["ordered_columns"]
rf_model = model_artifact["rf_model"]
model_threshold = model_artifact["rf_model_threshold"]

# Pre-process input transaction data
def pre_process(data_list, features):
    processed = []
    for data in data_list:
        data["amountOrig"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
        data["amountDest"] = data["oldbalanceDest"] - data["newbalanceDest"]
        data["errorBalanceOrig"] = data["amount"] - data["amountOrig"]
        data["errorBalanceDest"] = data["amount"] - data["amountDest"]
        for cat_f in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
            data[cat_f] = 1 if data["type"] == cat_f else 0
        processed.append([data[f] for f in features])
    return processed

# Predict fraud or genuine
def make_prediction(input_payload, model, threshold):
    prediction_probabilities = model.predict_proba(input_payload)
    predictions = [1 if p[1] > threshold else 0 for p in prediction_probabilities]
    return predictions

# Convert prediction to labels
def post_process(output):
    label_dict = {0: "Genuine", 1: "Fraud"}
    return [label_dict[o] for o in output]

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Home route to confirm API is live
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fraud Detection API is live!"})

# Define prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("data")
        if not data:
            abort(400, description="Missing 'data' in request JSON.")
        input_array = pre_process(data, features)
        output = make_prediction(input_array, rf_model, model_threshold)
        final_output = post_process(output)
        return jsonify({"predictions": final_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
