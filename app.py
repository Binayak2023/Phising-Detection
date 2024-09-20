from flask import Flask, request, render_template
import numpy as np
import joblib
import warnings
from feature import FeatureExtraction  # Ensure the feature extraction class is correct

warnings.filterwarnings('ignore')

# Load the saved Random Forest model
model_path = r'best_forest_model.pkl'
with open(model_path, 'rb') as file:
    rf_model = joblib.load(file)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get URL from form
        url = request.form["url"]

        # Extract features for the given URL
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, -1)  # Ensure the input shape is correct (1, num_features)

        # Make predictions
        y_pred = rf_model.predict(x)[0]  # Predict whether it's phishing or not
        y_pro_phishing = rf_model.predict_proba(x)[0, 1]  # Probability of phishing
        y_pro_non_phishing = rf_model.predict_proba(x)[0, 0]  # Probability of non-phishing

        # Determine the message based on the prediction
        if y_pred == 1:  # If the prediction is phishing
            pred = "It is {0:.2f}% likely to be a phishing site.".format(y_pro_phishing * 100)
        else:  # If the prediction is not phishing
            pred = "It is {0:.2f}% safe to go.".format(y_pro_non_phishing * 100)

        # Return result to the template
        return render_template('index.html', xx=round(y_pro_phishing * 100, 2), url=url, pred=pred)

    # Default to rendering the index page
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)
