from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Update these paths to where your saved files are located
MODEL_PATH = r"C:\Users\91897\AppData\Local\Programs\Python\Python312\saved_model.pkl"
SCALER_PATH = r"C:\Users\91897\AppData\Local\Programs\Python\Python312\scaler.pkl"

def load_model_or_scaler(path):
    try:
        obj = joblib.load(path)
        print(f"Loaded object of type {type(obj)} from {path}")
        return obj
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        raise

model = load_model_or_scaler(MODEL_PATH)
scaler = load_model_or_scaler(SCALER_PATH)

# Verify scaler object has transform method
if not hasattr(scaler, "transform"):
    raise TypeError(f"Scaler loaded is not a scaler object. Got {type(scaler)} instead.")

def preprocess_and_predict(df):
    # Filter good spots only
    df = df[df["Flags"] == 0].copy()

    # Calculate VALUE feature
    ch1 = df["CH1_SIG_Median"] - df["CH1_BKD_Median"]
    ch2 = df["CH2_SIG_Median"] - df["CH2_BKD_Median"]
    df["VALUE"] = np.log2((ch2 + 1e-5) / (ch1 + 1e-5))

    # Select features
    features = ["CH1_SIG_Median", "CH1_BKD_Median", "CH2_SIG_Median", "CH2_BKD_Median", "VALUE"]
    X = df[features]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict labels
    preds = model.predict(X_scaled)

    df["Predicted_Label"] = preds
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if not uploaded_file:
            result = {"error": "No file uploaded. Please upload a CSV file."}
        else:
            try:
                df = pd.read_csv(uploaded_file)
                df_pred = preprocess_and_predict(df)
                positive_count = df_pred["Predicted_Label"].sum()
                verdict = "Down Syndrome Detected" if positive_count > 0 else "No Down Syndrome Detected"
                result = {"verdict": verdict}
            except Exception as e:
                result = {"error": f"Error processing the file: {str(e)}"}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
