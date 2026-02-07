from flask import Flask, render_template, request
from joblib import load
import traceback

app = Flask(__name__)

# Load your trained fake job model
# Make sure this file exists in the same folder
model = load(r'model\fake_job_model.pkl')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        job_text = request.form.get("job_text", "").strip()

        if not job_text:
            return render_template(
                "result.html",
                error="Please paste a job description.",
                color="#cc0000"
            )

        # Get probabilities: [real, fake]
        proba = model.predict_proba([job_text])[0]
        prob_real = round(float(proba[0]) * 100, 2)
        prob_fake = round(float(proba[1]) * 100, 2)

        diff = abs(prob_real - prob_fake)

        show_both = False

        # Decide label + color
        if diff <= 5:
            show_both = True
            if prob_real > prob_fake:
                result = "⚠️ Uncertain – slightly more likely REAL"
            elif prob_fake > prob_real:
                result = "⚠️ Uncertain – slightly more likely FAKE"
            else:
                result = "⚠️ Perfect 50-50 – Uncertain"
            color = "#d97706"  # orange for uncertain
        else:
            if prob_real > prob_fake:
                result = "✅ This Job Posting is REAL"
                color = "#008000"  # green
            else:
                result = "🚨 This Job Posting is FAKE"
                color = "#cc0000"  # red

        return render_template(
            "result.html",
            result=result,
            prob_real=prob_real,
            prob_fake=prob_fake,
            show_both=show_both,
            color=color
        )

    except Exception as e:
        traceback.print_exc()
        return render_template(
            "result.html",
            error="Something went wrong while predicting.",
            color="#cc0000"
        )

if __name__ == "__main__":
    app.run(debug=True)
