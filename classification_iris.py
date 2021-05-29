#Iris Flower Prediction App

#Importing the libraries
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#Print heading
st.write("""# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!""") # ** is used for bold

#Creating a sidebar heading for user input parameters
st.sidebar.header('User Input Parameters')

#Creating a function which gets called when the input values are changed
def user_input_features():
    sepal_length=st.sidebar.slider('Sepal Length',4.3,7.9,5.4)
    sepal_width=st.sidebar.slider('Sepal Width',2.0,4.4,3.4)
    petal_length=st.sidebar.slider('Petal Length',1.0,6.9,1.3)
    petal_width=st.sidebar.slider('Petal Width',0.1,2.5,0.2)
    data={'sepal_length':sepal_length,
          'sepal_width':sepal_width,
          'petal_length':petal_length,
          'petal_width':petal_width}
    features=pd.DataFrame(data,index=[0])
    return features

#Calling the function
df=user_input_features()

#Creating the subheading
st.subheader('User Input Parameters')

#Printing the current values on the screen
st.write(df)

#Loading the data in model
iris=datasets.load_iris()
X=iris.data
Y=iris.target

#Build training model
clf=RandomForestClassifier()
clf.fit(X,Y)

#Predict the value from model built
prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)

#Write names of the class labels
st.subheader('Class Labels and their corresponding index number')
st.write(iris.target_names)

#Print predicted class label
st.subheader('Prediction')
st.write(iris.target_names[prediction])

#Print predicted probability
st.subheader('Prediction Probability')
st.write(prediction_proba)