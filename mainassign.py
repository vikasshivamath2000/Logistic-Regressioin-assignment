import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
st.title('Model Deployment : LogisticRegression')
st.sidebar.header('User Input Parameters')
def user_input_features():
    # # Input variables from the user

    Pclass = st.sidebar.selectbox("Passenger_class",('1','2','3'))
    Age = st.sidebar.number_input('Insert the age')
    SibSp = st.sidebar.selectbox("Number of Siblings/Spouses (SibSp)",('0','1','2','3','4','5','8'))
    Parch = st.sidebar.selectbox("Number of Parents/Children (Parch)",('0','1','2','3','4','5','6'))
    Fare = st.sidebar.number_input(" Insert the Fare value")
    Sex = st.sidebar.selectbox("Gender",('0','1'))
    Cabin = st.sidebar.number_input('Insert the cabin number ')
    Embarked = st.sidebar.selectbox('Embarked',('0','1','2'))
    # Combine inputs into a single array
    data = {'Pclass':Pclass,
            'Age':Age,
            'SibSp':SibSp,
            'Parch':Parch,
            'Fare':Fare,
            'Sex':Sex,
            'Cabin':Cabin,
            'Embarked':Embarked}
    features = pd.DataFrame(data,index = [0])
    return features
 
input_data = user_input_features()
st.subheader('User Input Parameters')
st.write(input_data)
Titanic_data = pd.read_csv('Titanic_data.csv')
#Titanic_data.drop(['Survived'],inplace = True, axis = 1)
Titanic_data = Titanic_data.dropna()

 # Make prediction

x = Titanic_data.iloc[:,2:]
y = Titanic_data.iloc[:,1]
clf = LogisticRegression()
clf.fit(x,y)
prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data)
st.subheader("Predicted Result")
st.write( "Survived" if prediction[0] == 1 else "Did Not Survive")
st.subheader('Prediction Probability')
st.write(prediction_proba)