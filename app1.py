import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
# Text data for my web page
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Will they buy the car or not?🚗")
    st.sidebar.markdown("Will they buy the car or not?🚗")

# Diplay data using checkbox
    @st.cache(persist = True)
    def data():
        data = pd.read_csv('Social_Network_Ads.csv')
        return data

        
# Split Dataframe into x and y dataframes
    @st.cache(persist = True)
    def split(df):
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)
        st.write("Running for the first time")
        return x_train, x_test, y_train, y_test
# Scale the features
    

# Calling the data, split and scale functions
    df = data()
    x_train, x_test, y_train, y_test = split(df)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    if st.sidebar.checkbox('Show Data',False):
        x_train, x_test, y_train, y_test
        st.write(type(x_train),type(x_test))

# Confusion Matrix
    def plot_metrics():
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(lr, x_test, y_test)
        st.pyplot()
        

# Training the models
    st.sidebar.subheader("Run the model")
    classifier = st.sidebar.selectbox("Choose your classifier",options= ('Logistic Regression', 'Random Forest'))
    if classifier == 'Logistic Regression':
        lr = LogisticRegression()
        lr.fit(x_train,y_train)
        y_pred = lr.predict(x_test)
        if st.sidebar.button("Classify",key = 'Classify'):
            plot_metrics()
            st.write("Now you can predict the value for one person")
        
    st.sidebar.subheader("One person prediction")
    Age = st.sidebar.number_input("Age",18,step = 1,key= 'Age')
    Salary = st.sidebar.number_input("Salary",10000,step = 1000,key= 'Salary')
    if st.sidebar.button("Predict", key = 'Predict'):
        st.write(lr.predict(sc.transform([[Age, Salary]])))

if __name__ == '__main__':
    main()