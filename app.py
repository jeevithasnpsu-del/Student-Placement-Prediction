from flask import Flask, request, render_template
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction function
@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    cgpa = float(request.form["CGPA"])
    internships = int(request.form["Internships"])
    projects = int(request.form["Projects"])
    workshops = int(request.form["Workshops"])
    aptitude = float(request.form["Aptitude"])
    softskills = float(request.form["SoftSkills"])
    extracurricular = int(request.form["Extracurricular"])
    training = int(request.form["Training"])
    ssc = float(request.form["SSC"])
    hsc = float(request.form["HSC"])

    # Create input array
    input_data = np.array([[cgpa, internships, projects, workshops,
                            aptitude, softskills,
                            extracurricular, training,
                            ssc, hsc]])

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)[0][1] * 100

    if probability < 40:
        suggestion = "Your placement chances are low. Improve CGPA, Aptitude, and gain more internships."

    elif probability < 70:
        suggestion = "Your placement chances are moderate. Improving projects and skills can increase chances."

    else:
        suggestion = "Your profile looks strong. You have high chances of placement."



    if prediction[0] == 1:
        result = f"Student will be Placed (Probability: {probability:.2f}%)"
    else:
        result = f"Student will NOT be Placed (Probability: {probability:.2f}%)"


    # Return result to HTML
    return render_template(
        "index.html",
        prediction_text=result,
        probability=round(probability, 2),
        suggestion=suggestion
    )


# Run app
if __name__ == "__main__":
    app.run(debug=True)