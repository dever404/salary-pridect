import pandas as pd
import streamlit as st 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def main(): 
    html_temp = """
    <div style="padding:20px; margin-bottom:20px;text-align:center;">
        <img style="text-align:center;margin-bottom:20px;" src="https://e-polytechnique.ma/landing/img/logo-polytechnique.png"/>
        <h1 style="color:white;text-align:center;">Predictive jobs salary </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    age = st.text_input("Age","0") 
    gender = st.selectbox("Gender",["Female","Male"])
    education = st.selectbox("Education",["Bachelor's","Master's","PhD"]) 
    job = st.selectbox("Job Title",["Software Engineer","Accountant","Administrative Assistant","Business Analyst","Business Development Manager","Business Intelligence Analyst","Chief Data Officer","Chief Technology Officer","Data Analyst","IT Manager"]) 
    experience = st.text_input("Years Of Experience","0") 

    if st.button("Predict"):
        data = {'age': int(age), 'gender': gender, 'education': education, 'job': job, 'experience':int(experience)}
        input_data=pd.DataFrame([list(data.values())], columns=['age','gender','education','job','experience'])

        salary = pd.read_csv('salary.csv')

        X = salary.drop(['salary'], axis=1)
        y = salary['salary']

        lm = LinearRegression()

        encod_salary = X.select_dtypes(include=['object']).columns.tolist()

        label_encoder = LabelEncoder()
        for col in encod_salary:
            X[col] = label_encoder.fit_transform(X[col])

        encode_input = input_data.select_dtypes(include=['object']).columns.tolist()

        label_encoder = LabelEncoder()
        for col in encode_input:
            input_data[col] = label_encoder.fit_transform(input_data[col])
        
        lm.fit(X, y)
        
        predicted_salary = lm.predict(input_data)
        st.success('Predicted Salary for Input Data : {}'.format( predicted_salary[0]))

    footer = """
        <footer style="text-align:center;margin-top:20px"> 
            <p style="margin-bottom: 5px"> Developped by :</p>
            <p> ER-RAFAIY , BELKACIM , ABOURAGBA</p>
            Source code <a href="https://github.com/dever404/salary-pridect" target="_blank">GitHub</a> @ 2024
        </footer>
    """
    st.markdown(footer, unsafe_allow_html = True)

if __name__=='__main__': 
    main() 
