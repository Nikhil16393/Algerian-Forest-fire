import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# -----------------------------
# Load Model & Scaler
# -----------------------------
MODEL_PATH = r"C:\Bootcamp_new\Algerian Fire Prediction\Models\Ridge.pkl"
SCALER_PATH = r"C:\Bootcamp_new\Algerian Fire Prediction\Models\scaler.pkl"

with open(MODEL_PATH, "rb") as f:
    ridge_model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    standard_scaler = pickle.load(f)


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    """Render the main prediction page."""
    return render_template("home.html", results=None)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    """Handle prediction request."""
    if request.method == "GET":
        return render_template("home.html", results=None)

    # POST
    try:
        required_fields = [
            "Temperature", "RH", "Ws", "Rain",
            "FFMC", "DMC", "ISI", "Classes", "Region"
        ]

        # Check missing fields
        missing = [f for f in required_fields if request.form.get(f) in (None, "")]
        if missing:
            return render_template(
                "home.html",
                results=None,
                error=f"Missing fields: {', '.join(missing)}"
            ), 400

        # Convert inputs to float in correct order
        values = [float(request.form.get(f)) for f in required_fields]

        # Shape: (1, 9)
        features = np.array([values])

        # Scale + predict
        scaled_features = standard_scaler.transform(features)
        prediction = ridge_model.predict(scaled_features)[0]

        return render_template("home.html", results=prediction)

    except ValueError:
        # User typed non-numeric input
        return render_template(
            "home.html",
            results=None,
            error="Please enter only numeric values in all fields."
        ), 400

    except Exception as e:
        # Any other unexpected error
        return render_template(
            "home.html",
            results=None,
            error=f"Something went wrong: {e}"
        ), 500


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
