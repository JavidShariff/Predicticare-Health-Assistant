# required packages:
# 1) pandas                     4) streamlit          7)matplotlib
# 2) scikit-learn               5) pandas             8)seaborn
# 3) streamlit_option_menu      6) numpy




import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



# Set page configuration
st.set_page_config(page_title="PredictiCare",
                   layout="wide",
                   page_icon="🏥")

# loading the saved models


diabetes_model = pickle.load(open("D:/devTitans/sav/diabetes_model.sav", 'rb'))
heart_model = pickle.load(open('D:/devTitans/sav/heart_model.sav', 'rb'))
insurance_model = pickle.load(open('D:/devTitans/sav/insurance_prediction.sav', 'rb'))

#working_dir=os.path.dirname(os.path.abspath(__file__))
folder_path="D:/devTitans/dataset"






# sidebar for navigation
with st.sidebar:
    selected = option_menu('PredictiCare',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Insurance Prediction','Data Visualization'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction')

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

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        #sex = st.text_input('Sex')
        try:
            sex = ['Male','Female']
            gender = st.selectbox("Sex",options = sex)
        except Exception as e:
            st.error(f"{e}")
        if gender == 'Male':
            gen = 1
        else:
            gen = 0

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST Depression Induced By Exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST Segment')

    with col3:
        ca = st.text_input('Major Vessels Colored By Flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, gen, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

        st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Insurance Prediction":

    # page title
    st.title("Insurance Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        try:
            sex = ['Male','Female']
            gender = st.selectbox("Sex",options = sex)
        except Exception as e:
            st.error(f"{e}")
        if gender == 'Male':
            gen = 1
        else:
            gen = 0

    with col3:
        bmi = st.text_input('BMI')

    with col1:
        children = st.text_input('Children')

    with col2:
        smoke = st.text_input('Do You Have Habbit Of Smoking?')
        if smoke == 'yes':
            smoke = 0
        else:
            smoke = 1

    with col3:
        region = st.text_input('Region')
        if region == 'southeast':
            region = 0
        elif region == 'southwest':
            region = 1
        elif region == 'northeast':
            region = 2
        elif region == 'northwest':
            region = 3
        


    # code for Prediction
    insurance_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Insurance Test Result"):

        user_input = [age,sex,bmi,children,smoke,region]
        user_input = [float(x) for x in user_input]
        try:
            insurance_prediction = insurance_model.predict([user_input])
        except Exception as e:
            st.error(f"Error loading the model: {e}")
        try:
            st.success(insurance_prediction[0])
        except:
            st.write("Fill The Above Details")
    
            

if selected == "Data Visualization":
    files=[f for f in os.listdir(folder_path) if f.endswith(".csv")]

    selected_file=st.selectbox("Select a File", files,index=None)
    

    if selected_file:

        file_path=os.path.join(folder_path,selected_file)

        df=pd.read_csv(file_path)

        col1,col2=st.columns(2)

        columns=df.columns.tolist()

        with col1:
            st.write("")
            st.write(df.head())

        with col2:
            x_axis=st.selectbox("Select x-axis",options=columns+["none"],index=None)
            y_axis=st.selectbox("Select y-axis",options=columns+["none"],index=None)

            plot_list=["line Plot","bar Chart","scatter Plot","Distribution Plot","Count Plot"]

            selected_plot=st.selectbox("Select a Plot" ,options=plot_list)

            st.write(x_axis)
            st.write(y_axis)
            st.write(selected_plot)


        if st.button("Generate Plot"):
            fig,ax=plt.subplots(figsize=(6,4))

            if selected_plot=="line Plot":
                sns.lineplot(x=df[x_axis],y=df[y_axis],ax=ax)

            elif selected_plot == "bar Chart":
                sns.barplot(x=df[x_axis], y=df[y_axis],ax=ax)

            elif selected_plot == "scatter Plot":
                sns.scatterplot(x=df[x_axis], y=df[y_axis],ax=ax)

            elif selected_plot == "Distribution Plot":
                sns.histplot(df[x_axis], kde=True,ax=ax)

            elif selected_plot == "count Plot":
                sns.countplot(df[x_axis],kde=True,ax=ax)


            ax.tick_params(axis="x",labelsize=10)
            ax.tick_params(axis="y",labelsize=10)

            plt.title(f"{selected_plot} of {y_axis} vs {x_axis}",fontsize=12)

            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            st.pyplot(fig)

