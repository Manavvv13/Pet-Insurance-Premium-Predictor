from flask import Flask, request, render_template
import pandas as pd
from catboost import CatBoostRegressor
import joblib

app = Flask(__name__)

model = CatBoostRegressor()
model.load_model("catboost_premium_model.cbm")

# Load the breed label encoder
breed_encoder = joblib.load("breed_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            form = request.form

            deductible = float(form["deductible"])
            age = float(form["age"])
            species = int(form["species"])
            breed_name = form["breed"]
            breed = breed_encoder.transform([breed_name])[0]

            amt_claims_total = float(form["claims_total"])
            amt_claims_yr1 = float(form["claims_yr1"])
            amt_claims_yr2 = float(form["claims_yr2"])

            num_claims_yr1 = int(form["num_yr1"])
            num_claims_yr2 = int(amt_claims_total > 0 and amt_claims_yr2 > 0)

            has_chronic = int(form["chronic"])
            vaccinated = int(form["vaccinated"])
            vet_visits = int(form["visits"])
            surgery = int(form["surgery"])

            # Compute averages
            avg_yr1 = amt_claims_yr1 / num_claims_yr1 if num_claims_yr1 > 0 else 0
            avg_yr2 = amt_claims_yr2 / num_claims_yr2 if num_claims_yr2 > 0 else 0
            total_claims = amt_claims_yr1 + amt_claims_yr2
            total_num_claims = num_claims_yr1 + num_claims_yr2
            avg_total = total_claims / total_num_claims if total_num_claims > 0 else 0

            input_data = {
                "Deductible": deductible,
                "AgeYr1": age,
                "Species": species,
                "Breed": breed,
                "AmtClaimsTotal": amt_claims_total,
                "AmtClaimsYr1": amt_claims_yr1,
                "AmtClaimsYr2": amt_claims_yr2,
                "AvgClaimsYr1": avg_yr1,
                "AvgClaimsYr2": avg_yr2,
                "AvgClaimsTotal": avg_total,
                "NumClaimsYr1": num_claims_yr1,
                "HasChronicDisease": has_chronic,
                "Vaccinated": vaccinated,
                "NumVetVisits": vet_visits,
                "SurgeryHistory": surgery
            }

            df = pd.DataFrame([input_data])
            prediction = model.predict(df)[0]

            return render_template("index.html", prediction=round(prediction, 2), form_data=form)

        except Exception as e:
            return f"❌ Error: {str(e)}", 500

    return render_template("index.html", prediction=None, form_data={})

if __name__ == "__main__":
    app.run(debug=True)
