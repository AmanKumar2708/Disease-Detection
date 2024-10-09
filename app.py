import os
import pandas as pd
import pickle
import joblib
import streamlit as st
from streamlit_option_menu import option_menu



# Set page configuration
st.set_page_config(page_title="Disease Prediction",
                   layout="wide",
                   page_icon="🧑‍⚕")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define custom CSS for improved aesthetics
custom_css = """
<style>
    body {
        background-color: #e9ecef; /* Light gray background */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #007bff; /* Bootstrap primary color */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px 24px; /* Increased padding for buttons */
        font-size: 18px; /* Increased font size for buttons */
        transition: background-color 0.3s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker shade on hover */
    }
    .stTitle {
        color: #0056b3; /* Title color */
        font-size: 32px; /* Increased title size */
        text-align: center; /* Centered title */
        margin-bottom: 20px; /* Margin below title */
    }
    /* Applying borders to all input elements */
    input, .stTextInput, .stSelectbox, .stSlider, .stCheckbox, .stRadio, .stTextArea, .stNumberInput {
        border: 1.85px solid #19005e; /* Purple border */
        border-radius: 5px;
        padding: 10px; /* Padding inside input fields */
        font-size: 16px; /* Font size for inputs */
        margin-top: 10px; /* Margin above inputs */
        display: block; /* Ensuring they take up full width */
        width: calc(100% - 22px); /* Full width minus padding */
    }
    input:hover, .stTextInput:hover, .stSelectbox:hover, .stSlider:hover, .stCheckbox:hover, .stRadio:hover, .stTextArea:hover, .stNumberInput:hover {
        border-color: #007bff; /* Change border color on hover */
    }
    .stSidebar {
        background-color: #f8f9fa; /* Light background for sidebar */
        padding: 20px; /* Padding for sidebar content */
        border-radius: 8px; /* Rounded corners for sidebar */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
    }
    # .stMarkdown {
    #     margin: 20px 0; /* Margin for markdown sections */
    #     padding: 20px; /* Padding for markdown sections */
    #     background-color: #ffffff; /* White background for content */
    #     border-radius: 8px; /* Rounded corners for content */
    #     box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow for content */
    # }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


# loading the saved models

diabetes_model = pickle.load(open('C:/Users/27ama/multiple-disease-prediction-streamlit-app/colab_files_to_train_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('C:/Users/27ama/multiple-disease-prediction-streamlit-app/colab_files_to_train_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('C:/Users/27ama/multiple-disease-prediction-streamlit-app/colab_files_to_train_models/parkinsons_model.sav', 'rb'))

liver_model = joblib.load(open('C:/Users/27ama/multiple-disease-prediction-streamlit-app/colab_files_to_train_models/liver.sav', 'rb'))

kidneys_model = joblib.load(open('C:/Users/27ama/multiple-disease-prediction-streamlit-app/colab_files_to_train_models/chronic_model.sav', 'rb'))

breastcancer_model = joblib.load(open('C:/Users/27ama/multiple-disease-prediction-streamlit-app/colab_files_to_train_models/breast_cancer.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction','Liver Prediction','Kidney Prediction','Breast Cancer Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'lungs'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using Machine Learning')

    # User's name input
    name = st.text_input("Patient's Name:")

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        # Check for empty values and handle them
        if any(val == "" for val in user_input):
            st.error("Please fill in all fields with valid numbers.")
        else:
            try:
                # Convert input values to appropriate types
                user_input = [float(x) for x in user_input]

                diab_prediction = diabetes_model.predict([user_input])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'

            except ValueError as e:
                st.error(f"Invalid input: {e}")

    st.success(f"{name}, {diab_diagnosis}")


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using Machine Learning')

    # User's name input
    name = st.text_input("Patient's Name:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thalassemia')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        # Check for empty values and handle them
        if any(val == "" for val in user_input):
            st.error("Please fill in all fields with valid numbers.")
        else:
            try:
                # Convert input values to appropriate types
                user_input = [float(x) if x.replace('.', '', 1).isdigit() else x for x in user_input]

                heart_prediction = heart_disease_model.predict([user_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'

            except ValueError as e:
                st.error(f"Invalid input: {e}")

    st.success(f"{name}, {heart_diagnosis}")


# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using Machine Learning")

    # User's name input
    name = st.text_input("Patient's Name:")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', "")
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', "")
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', "")
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)', "")
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', "")
    with col1:
        RAP = st.text_input('MDVP:RAP', "")
    with col2:
        PPQ = st.text_input('MDVP:PPQ', "")
    with col3:
        DDP = st.text_input('Jitter:DDP', "")
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer', "")
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', "")
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3', "")
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5', "")
    with col3:
        APQ = st.text_input('MDVP:APQ', "")
    with col4:
        DDA = st.text_input('Shimmer:DDA', "")
    with col5:
        NHR = st.text_input('NHR', "")
    with col1:
        HNR = st.text_input('HNR', "")
    with col2:
        RPDE = st.text_input('RPDE', "")
    with col3:
        DFA = st.text_input('DFA', "")
    with col4:
        spread1 = st.text_input('spread1', "")
    with col5:
        spread2 = st.text_input('spread2', "")
    with col1:
        D2 = st.text_input('D2', "")
    with col2:
        PPE = st.text_input('PPE', "")

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        # Check for empty values and handle them
        if any(val == "" for val in user_input):
            st.error("Please fill in all fields with valid numbers.")
        else:
            try:
                user_input = [float(x) for x in user_input]

                parkinsons_prediction = parkinsons_model.predict([user_input])

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's disease"
                
                st.success(f"{name}, {parkinsons_diagnosis}")

            except ValueError as e:
                st.error(f"Invalid input: {e}")


#Liver Prediciton
if selected == 'Liver Prediction':
    st.title("Liver Disease Prediction using Machine Learning")
    
    # Get user inputs
    name = st.text_input(" Patient's Name:")
    col1, col2, col3 = st.columns(3)

    # Gender selection (correction in logic)
    with col1:
        display = ("Male", "Female")
        options = list(range(len(display)))
        gender = st.selectbox("Gender", options, format_func=lambda x: display[x])
        Sex = 0 if gender == 0 else 1  # Male: 0, Female: 1

    with col2:
        age = st.number_input("Enter your age")  # 2 
    with col3:
        Total_Bilirubin = st.number_input("Enter your Total Bilirubin")  # 3

    with col1:
        Direct_Bilirubin = st.number_input("Enter your Direct Bilirubin")  # 4
    with col2:
        Alkaline_Phosphotase = st.number_input("Enter your Alkaline Phosphotase")  # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Enter your Alamine Aminotransferase")  # 6

    with col1:
        Aspartate_Aminotransferase = st.number_input("Enter your Aspartate Aminotransferase")  # 7
    with col2:
        Total_Protiens = st.number_input("Enter your Total Proteins")  # 8
    with col3:
        Albumin = st.number_input("Enter your Albumin")  # 9

    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Enter your Albumin and Globulin Ratio")  # 10

    liver_diagnosis = ''

    # Prediction button
# Ensure that Liver_model is a model and not a DataFrame
    if st.button("Liver's Test Result"):

        user_input = [Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]

        user_input = [float(x) for x in user_input]

        liver_prediction = liver_model.predict([user_input])

        if liver_prediction[0] == 1:
            liver_diagnosis = "The person has Liver's disease"
        else:
            liver_diagnosis = "The person does not have Liver's disease"

    st.success(liver_diagnosis)

import pandas as pd
import streamlit as st

# Chronic Kidney Disease Prediction Page
if selected == 'Kidney Prediction':
    st.title("Chronic Kidney Disease Prediction using Machine Learning")
    
    # User's name input
    name = st.text_input("Patient's Name:")

    # Columns for input fields
    col1, col2, col3 = st.columns(3)

    # Input fields
    with col1:
        age = st.slider("Enter your age", 1, 100, 25)
        al = st.slider("Enter your Albumin", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        rbc = 1 if rbc == "Normal" else 0
        pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
        pc = 1 if pc == "Normal" else 0
        bgr = st.slider("Enter your Blood Glucose Random", 50, 200, 120)
        sod = st.slider("Enter your Sodium", 100, 200, 140)
        pcv = st.slider("Enter your Packed Cell Volume", 20, 60, 40)
        appet = st.selectbox("Appetite", ["Good", "Poor"])
        appet = 1 if appet == "Good" else 0

    with col2:
        bp = st.slider("Enter your Blood Pressure", 50, 200, 120)
        su = st.slider("Enter your Sugar", 0, 5, 0)
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
        pcc = 1 if pcc == "Present" else 0
        bu = st.slider("Enter your Blood Urea", 10, 200, 60)
        pot = st.slider("Enter your Potassium", 2, 7, 4)
        wc = st.slider("Enter your White Blood Cell Count", 2000, 20000, 10000)
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
        dm = 1 if dm == "Yes" else 0
        pe = st.selectbox("Pedal Edema", ["Yes", "No"])
        pe = 1 if pe == "Yes" else 0

    with col3:
        sg = st.slider("Enter your Specific Gravity", 1.0, 1.05, 1.02)
        ba = st.selectbox("Bacteria", ["Present", "Not Present"])
        ba = 1 if ba == "Present" else 0
        sc = st.slider("Enter your Serum Creatinine", 0, 10, 3)
        hemo = st.slider("Enter your Hemoglobin", 3, 17, 12)
        rc = st.slider("Enter your Red Blood Cell Count", 2, 8, 4)
        htn = st.selectbox("Hypertension", ["Yes", "No"])
        htn = 1 if htn == "Yes" else 0
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
        cad = 1 if cad == "Yes" else 0
        ane = st.selectbox("Anemia", ["Yes", "No"])
        ane = 1 if ane == "Yes" else 0

    # Code for prediction
    kidney_prediction_dig = ''

    # Button to make prediction
    if st.button("Predict Chronic Kidney Disease"):
        # Create a DataFrame with user inputs
        user_input = pd.DataFrame({
            'age': [age],
            'bp': [bp],
            'sg': [sg],
            'al': [al],
            'su': [su],
            'rbc': [rbc],
            'pc': [pc],
            'pcc': [pcc],
            'ba': [ba],
            'bgr': [bgr],
            'bu': [bu],
            'sc': [sc],
            'sod': [sod],
            'pot': [pot],
            'hemo': [hemo],
            'pcv': [pcv],
            'wc': [wc],
            'rc': [rc],
            'htn': [htn],
            'dm': [dm],
            'cad': [cad],
            'appet': [appet],
            'pe': [pe],
            'ane': [ane]
        })
        
        # Extract values as a 1D array
        user_input_values = user_input.values.reshape(1, -1)  # Reshape to match the model input shape

        # Perform prediction
        kidney_prediction = kidneys_model.predict(user_input_values)

        # Display result
        if kidney_prediction[0] == 1:
            kidney_prediction_dig = "We are really sorry to say, but it seems like you have kidney disease."
        else:
            kidney_prediction_dig = "Congratulations, you don't have kidney disease."
        
        st.success(f"{name}, {kidney_prediction_dig}")

# Breast Cancer Prediction Page

if selected == 'Breast Cancer Prediction':
    st.title("Breast Cancer Prediction using Machine Learning")
    name = st.text_input("Patient's Name:")
    # Columns
    # No inputs from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.slider("Enter your Radius Mean", 6.0, 30.0, 15.0)
        texture_mean = st.slider("Enter your Texture Mean", 9.0, 40.0, 20.0)
        perimeter_mean = st.slider("Enter your Perimeter Mean", 43.0, 190.0, 90.0)

    with col2:
        area_mean = st.slider("Enter your Area Mean", 143.0, 2501.0, 750.0)
        smoothness_mean = st.slider("Enter your Smoothness Mean", 0.05, 0.25, 0.1)
        compactness_mean = st.slider("Enter your Compactness Mean", 0.02, 0.3, 0.15)

    with col3:
        concavity_mean = st.slider("Enter your Concavity Mean", 0.0, 0.5, 0.2)
        concave_points_mean = st.slider("Enter your Concave Points Mean", 0.0, 0.2, 0.1)
        symmetry_mean = st.slider("Enter your Symmetry Mean", 0.1, 1.0, 0.5)

    with col1:
        fractal_dimension_mean = st.slider("Enter your Fractal Dimension Mean", 0.01, 0.1, 0.05)
        radius_se = st.slider("Enter your Radius SE", 0.1, 3.0, 1.0)
        texture_se = st.slider("Enter your Texture SE", 0.2, 2.0, 1.0)

    with col2:
        perimeter_se = st.slider("Enter your Perimeter SE", 1.0, 30.0, 10.0)
        area_se = st.slider("Enter your Area SE", 6.0, 500.0, 150.0)
        smoothness_se = st.slider("Enter your Smoothness SE", 0.001, 0.03, 0.01)

    with col3:
        compactness_se = st.slider("Enter your Compactness SE", 0.002, 0.2, 0.1)
        concavity_se = st.slider("Enter your Concavity SE", 0.0, 0.05, 0.02)
        concave_points_se = st.slider("Enter your Concave Points SE", 0.0, 0.03, 0.01)

    with col1:
        symmetry_se = st.slider("Enter your Symmetry SE", 0.1, 1.0, 0.5)
        fractal_dimension_se = st.slider("Enter your Fractal Dimension SE", 0.01, 0.1, 0.05)

    with col2:
        radius_worst = st.slider("Enter your Radius Worst", 7.0, 40.0, 20.0)
        texture_worst = st.slider("Enter your Texture Worst", 12.0, 50.0, 25.0)
        perimeter_worst = st.slider("Enter your Perimeter Worst", 50.0, 250.0, 120.0)

    with col3:
        area_worst = st.slider("Enter your Area Worst", 185.0, 4250.0, 1500.0)
        smoothness_worst = st.slider("Enter your Smoothness Worst", 0.07, 0.3, 0.15)
        compactness_worst = st.slider("Enter your Compactness Worst", 0.03, 0.6, 0.3)

    with col1:
        concavity_worst = st.slider("Enter your Concavity Worst", 0.0, 0.8, 0.4)
        concave_points_worst = st.slider("Enter your Concave Points Worst", 0.0, 0.2, 0.1)
        symmetry_worst = st.slider("Enter your Symmetry Worst", 0.1, 1.0, 0.5)

    with col2:
        fractal_dimension_worst = st.slider("Enter your Fractal Dimension Worst", 0.01, 0.2, 0.1)

        # Code for prediction
    breast_cancer_result = ''

    # Button
    if st.button("Predict Breast Cancer"):
        # Create a DataFrame with user inputs
        user_input = pd.DataFrame({
            'radius_mean': [radius_mean],
            'texture_mean': [texture_mean],
            'perimeter_mean': [perimeter_mean],
            'area_mean': [area_mean],
            'smoothness_mean': [smoothness_mean],
            'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean],
            'concave points_mean': [concave_points_mean],  # Update this line
            'symmetry_mean': [symmetry_mean],
            'fractal_dimension_mean': [fractal_dimension_mean],
            'radius_se': [radius_se],
            'texture_se': [texture_se],
            'perimeter_se': [perimeter_se],
            'area_se': [area_se],
            'smoothness_se': [smoothness_se],
            'compactness_se': [compactness_se],
            'concavity_se': [concavity_se],
            'concave points_se': [concave_points_se],  # Update this line
            'symmetry_se': [symmetry_se],
            'fractal_dimension_se': [fractal_dimension_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst],
            'area_worst': [area_worst],
            'smoothness_worst': [smoothness_worst],
            'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst],
            'concave points_worst': [concave_points_worst],  # Update this line
            'symmetry_worst': [symmetry_worst],
            'fractal_dimension_worst': [fractal_dimension_worst],
        })

        # Perform prediction
        breast_cancer_prediction = breastcancer_model.predict(user_input)
        # Display result
        if breast_cancer_prediction[0] == 1:
          
            breast_cancer_result = "The model predicts that you have Breast Cancer."
        else:
           
            breast_cancer_result = "The model predicts that you don't have Breast Cancer."

        st.success(breast_cancer_result)