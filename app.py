import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

def main(): 
    #st.title("Jobs slary pridect")
    html_temp = """
    <div style="background:#025246 ;padding:10px; margin-bottom:20px">
    <h2 style="color:white;text-align:center;">Jobs slary pridect App </h2>
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        lm = LinearRegression()

        ordinal_inco=OrdinalEncoder()
        x_train=ordinal_inco.fit_transform(X_train)
        x_test=ordinal_inco.fit_transform(input_data)
        
        lm.fit(x_train, y_train)
        
        predicted_salary = lm.predict(x_test)

        st.success('Predicted Salary for Input Data : {}'.format( predicted_salary[0]))

if __name__=='__main__': 
    main() 