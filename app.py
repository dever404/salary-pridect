import pandas as pd
import streamlit as st 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder

def main(): 
    html_temp = """
    <div style="background:#025246 ;padding:10px; margin-bottom:20px">
    <h2 style="color:white;text-align:center;">Predictive jobs salary </h2>
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

        ordinal_inco=OrdinalEncoder()
        x_train=ordinal_inco.fit_transform(X)
        x_test=ordinal_inco.fit_transform(input_data)
        
        lm.fit(x_train, y)
        
        predicted_salary = lm.predict(x_test)

        st.success('Predicted Salary for Input Data : {}'.format( predicted_salary[0]))

    footer = """
    <footer style="text-align:center;margin-top:20px"> Source code <a href="https://github.com/dever404/salary-pridect" target="_blank">GitHub</a></footer>
    """
    st.markdown(footer, unsafe_allow_html = True)

if __name__=='__main__': 
    main() 
