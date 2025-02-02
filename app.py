# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from fpdf import FPDF
import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ["glioma","meningioma","notumor","pituitary"]
disease_dic2= ["Bacterial Pneumonia","Normal","Viral Pneumonia"]
#{'Cyst': 0, 'Normal': 1, 'Stone': 2, 'Tumor': 3}
disease_dic3= ["Cyst","Normal","Stone","Tumor"]

from model_predict  import pred_brain_tumer
from model_predict_pnemonia  import pred_pnemonia
from model_predict_kidney import pred_kidney
# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'AI-Powered HealthCare Diagnosis System'
    return render_template('index.html', title=title)

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# render crop recommendation form page
disease_details = {
    "glioma": {
        "class_rust": "glioma",
        "disease_name": "Glioma",
        "stage": "Stage 2 or Stage 3 (commonly)",
        "condition": "Aggressive tumor affecting the glial cells.",
        "symptoms": "Headache, seizures, nausea, cognitive impairment.",
        "possible_causes": "Genetic mutations, radiation exposure.",
        "diagnosis": "MRI scans, CT scans, biopsy.",
        "treatment": "Surgery, radiation therapy, chemotherapy.",
        "estimated_cost": "Approx. $50,000 - $150,000 USD.",
        "recovery_period": "6 months to 2 years, depending on treatment.",
        "precaution": "Regular follow-ups, MRI scans, a healthy diet, and avoiding radiation exposure."
    },
    "meningioma": {
        "class_rust": "meningioma",
        "disease_name": "Meningioma",
        "stage": "Usually benign but can be aggressive in some cases.",
        "condition": "Tumor in the meninges (protective brain layers).",
        "symptoms": "Headache, vision problems, hearing loss, memory issues.",
        "possible_causes": "Genetic factors, exposure to radiation, hormonal changes.",
        "diagnosis": "MRI scans, CT scans, X-rays.",
        "treatment": "Observation, surgery, radiation therapy.",
        "estimated_cost": "Approx. $30,000 - $70,000 USD.",
        "recovery_period": "3 months to 1 year.",
        "precaution": "Monitor symptoms, maintain regular check-ups, and avoid head trauma."
    },
    "notumor": {
        "class_rust": "notumor",
        "disease_name": "No Tumor Detected",
        "stage": "N/A",
        "condition": "Normal brain structure.",
        "symptoms": "No symptoms of concern.",
        "possible_causes": "N/A",
        "diagnosis": "N/A",
        "treatment": "No treatment required.",
        "estimated_cost": "N/A",
        "recovery_period": "N/A",
        "precaution": "Continue a healthy lifestyle, regular check-ups if symptoms persist."
    },
    "pituitary": {
        "class_rust": "pituitary",
        "disease_name": "Pituitary Tumor",
        "stage": "Stage 1 or higher, depending on tumor size.",
        "condition": "Tumor in the pituitary gland causing hormonal imbalances.",
        "symptoms": "Headache, vision problems, fatigue, hormonal disorders.",
        "possible_causes": "Genetic factors, family history of pituitary conditions.",
        "diagnosis": "MRI scans, hormone level tests, biopsy.",
        "treatment": "Surgery, hormone therapy, radiation therapy (if required).",
        "estimated_cost": "Approx. $40,000 - $90,000 USD.",
        "recovery_period": "6 months to 1.5 years.",
        "precaution": "Regular monitoring of hormone levels, MRI scans, and proper medication adherence."
    }
}
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Brain Tumor Detection'
    
    if request.method == 'POST':
        # Capture form fields
        name = request.form.get('name')
        email = request.form.get('email')
        age = request.form.get('age')
        phone = request.form.get('phone')
        gender = request.form.get('gender')

        # Print user details for debugging (you can remove this later)
        print(f"Name: {name}, Email: {email}, Age: {age}, Phone: {phone}")

        # Handle file upload for image
        file = request.files.get('file')
        if file:
            img = Image.open(file)
            img.save('output.png')

            # Prediction
            prediction = pred_brain_tumer("output.png")
            prediction = (str(disease_dic[prediction]))

            # Check prediction and fetch disease details
            if prediction in disease_details:
                details = disease_details[prediction]
            else:
                details = {
                    "disease_name": "Unknown Condition",
                    "stage": "N/A",
                    "condition": "No information available for this condition.",
                    "symptoms": "N/A",
                    "possible_causes": "N/A",
                    "diagnosis": "N/A",
                    "treatment": "N/A",
                    "estimated_cost": "N/A",
                    "recovery_period": "N/A",
                    "precaution": "No information available."
                }

            # Render the disease result page with additional user info
            return render_template(
                'disease-result.html',
                prediction=prediction,
                details=details,
                title=title,
                name=name,
                email=email,
                age=age,
                phone=phone,
                gender=gender
            )
    
    return render_template('disease.html', title=title)
# Define a dictionary for pneumonia details
pneumonia_details = {
    "Bacterial Pneumonia": {
        "disease_name": "Bacterial Pneumonia",
        "condition": "Infection in the lungs caused by bacteria, leading to inflammation.",
        "symptoms": "Fever, cough with yellow or green mucus, chest pain, shortness of breath.",
        "possible_causes": "Streptococcus pneumoniae bacteria, weakened immune system.",
        "diagnosis": "Chest X-rays, blood tests, sputum culture.",
        "treatment": "Antibiotics, oxygen therapy, rest, and fluids.",
        "estimated_cost": "Approx. $5,000 - $15,000 USD (depending on severity and location).",
        "recovery_period": "1 to 3 weeks with proper treatment.",
        "precaution": "Vaccination, good hygiene, avoiding close contact with infected individuals, and strengthening immunity."
    },
    "Normal": {
        "disease_name": "Normal Lungs",
        "condition": "No signs of pneumonia. Healthy lung function.",
        "symptoms": "No symptoms of concern.",
        "possible_causes": "N/A",
        "diagnosis": "No further tests required.",
        "treatment": "No treatment required.",
        "estimated_cost": "N/A",
        "recovery_period": "N/A",
        "precaution": "Maintain a healthy lifestyle, avoid smoking, and get regular check-ups."
    },
    "Viral Pneumonia": {
        "disease_name": "Viral Pneumonia",
        "condition": "Lung infection caused by viruses, leading to inflammation and breathing difficulties.",
        "symptoms": "Cough, fever, chills, fatigue, shortness of breath.",
        "possible_causes": "Influenza virus, SARS-CoV-2, respiratory syncytial virus (RSV).",
        "diagnosis": "Chest X-rays, PCR tests, blood tests.",
        "treatment": "Antiviral medications, supportive care (rest, fluids, oxygen therapy).",
        "estimated_cost": "Approx. $3,000 - $10,000 USD.",
        "recovery_period": "1 to 2 weeks (mild cases) or up to several weeks for severe cases.",
        "precaution": "Vaccination, wearing masks, avoiding crowded places, maintaining proper hygiene."
    }
}
@app.route('/disease-predict_pnemonia', methods=['GET', 'POST'])
def disease_prediction_pnemonia():
    title = 'Brain Tumer Detection'

    if request.method == 'POST':
        # Capture user details from form
        name = request.form.get('name')
        email = request.form.get('email')
        age = request.form.get('age')
        phone = request.form.get('phone')
        gender = request.form.get('gender')

        file = request.files.get('file')

        img = Image.open(file)
        img.save('output.png')

        prediction = pred_pnemonia("output.png")
        prediction = (str(disease_dic2[prediction]))

        # Check the prediction and retrieve details
        details = pneumonia_details[prediction]

        # Render the template with the retrieved details and user info
        return render_template(
            'disease-result_pnemonia.html',
            prediction=prediction,
            details=details,
            title=title,
            name=name,
            email=email,
            age=age,
            phone=phone,
            gender=gender
        )

    return render_template('disease_pnemonia.html', title=title)

kidney_details = {
    "Cyst": {
        "disease_name": "Kidney Cyst",
        "condition": "Fluid-filled sac that forms on or inside the kidney.",
        "symptoms": "Back or side pain, fever, frequent urination, blood in urine.",
        "possible_causes": "Age-related changes, genetic disorders (e.g., polycystic kidney disease).",
        "diagnosis": "Ultrasound, CT scan, MRI, urine tests.",
        "treatment": "Observation, drainage, or surgery if causing complications.",
        "estimated_cost": "Approx. $2,000 - $10,000 USD.",
        "recovery_period": "1 to 3 weeks after treatment.",
        "precaution": "Maintain proper hydration, avoid smoking, and monitor blood pressure regularly."
    },
    "Normal": {
        "disease_name": "Healthy Kidneys",
        "condition": "Normal kidney function with no abnormalities.",
        "symptoms": "No symptoms.",
        "possible_causes": "N/A",
        "diagnosis": "No further tests required.",
        "treatment": "No treatment required.",
        "estimated_cost": "N/A",
        "recovery_period": "N/A",
        "precaution": "Stay hydrated, eat a balanced diet, and avoid excessive salt intake."
    },
    "Stone": {
        "disease_name": "Kidney Stone",
        "condition": "Hard deposits made of minerals and salts that form inside the kidneys.",
        "symptoms": "Severe pain in the back or side, nausea, vomiting, blood in urine.",
        "possible_causes": "Dehydration, high salt or calcium intake, genetic factors.",
        "diagnosis": "CT scan, ultrasound, X-rays, blood and urine tests.",
        "treatment": "Pain relievers, hydration, lithotripsy, or surgical removal.",
        "estimated_cost": "Approx. $3,000 - $15,000 USD (depending on treatment type).",
        "recovery_period": "1 to 4 weeks after treatment.",
        "precaution": "Drink plenty of water, reduce salt intake, and avoid high oxalate foods."
    },
    "Tumor": {
        "disease_name": "Kidney Tumor",
        "condition": "Abnormal growth of cells in the kidney, possibly cancerous.",
        "symptoms": "Blood in urine, back pain, weight loss, fatigue.",
        "possible_causes": "Smoking, genetic mutations, chronic kidney disease.",
        "diagnosis": "CT scan, MRI, biopsy, blood tests.",
        "treatment": "Surgery (nephrectomy), radiation therapy, chemotherapy.",
        "estimated_cost": "Approx. $20,000 - $50,000 USD.",
        "recovery_period": "3 months to 1 year, depending on severity.",
        "precaution": "Regular check-ups, healthy lifestyle, and avoid smoking or alcohol."
    }
}
@app.route('/disease-predict_kidney', methods=['GET', 'POST'])

def disease_prediction_kidney():
    title = 'Kidney Disease Detection'

    if request.method == 'POST':
        # Capture form fields for name, email, phone number, age, and gender
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        age = request.form.get('age')
        gender = request.form.get('gender')  # New gender field

        # Print user details for debugging (you can remove this later)
        print(f"Name: {name}, Email: {email}, Phone: {phone}, Age: {age}, Gender: {gender}")

        # Handle file upload for image
        file = request.files.get('file')
        if file:
            img = Image.open(file)
            img.save('output.png')

            # Prediction
            prediction = pred_kidney("output.png")
            prediction = str(disease_dic3[prediction])

            # Check prediction and fetch disease details
            if prediction in kidney_details:
                details = kidney_details[prediction]
            else:
                details = {
                    "disease_name": "Unknown Condition",
                    "condition": "No information available for this condition.",
                    "symptoms": "N/A",
                    "possible_causes": "N/A",
                    "diagnosis": "N/A",
                    "treatment": "N/A",
                    "estimated_cost": "N/A",
                    "recovery_period": "N/A",
                    "precaution": "No information available."
                }

            # Render the disease result page with additional user info
            return render_template(
                'disease-result_kidney.html',
                prediction=prediction,
                details=details,
                title=title,
                name=name,
                email=email,
                phone=phone,
                age=age,
                gender=gender  # Pass gender to the template
            )
    
    # Render the form page if the method is not POST
    return render_template('disease_kidney.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
