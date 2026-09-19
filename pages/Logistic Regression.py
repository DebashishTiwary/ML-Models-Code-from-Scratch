import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Real Linear Regression App", layout="wide")

st.title("Logistic Regression Model")
@st.cache_data
def generate_data():
    np.random.seed(42)
    X=np.random.rand(100,1)
    Y = (X > 0.5).astype(int)
    return X,Y
X_train, Y_train = generate_data()
@st.cache_resource
def train_model(X,Y):
    model = LogisticRegression()
    model.fit(X,Y)
    return model
model = train_model(X_train, Y_train)

st.sidebar.header("Input Features for Prediction")
x1_input = st.sidebar.slider("Select Feature X1:", -100.0, 100.0, 10.0, step=0.1)
#x2_input = st.sidebar.slider("Select Feature X2:", 0.0, 10.0, 5.0, step=0.1)


col1, col2,col3 = st.columns([1,2,2])

with col1:
    st.subheader("Model Prediction")
    
    
    user_data = np.array([[x1_input]])
    y_pred = model.predict(user_data)[0]
        
    st.metric(label="Predicted Output (Y)", value=f"{y_pred:.2f}")
with col2:
    st.subheader("Explanation")
    df = pd.DataFrame({
        'Feature X1': X_train[:, 0],
        #'Feature X2': X_train[:, 1],
        'Target Y': Y_train.flatten()

    })
    st.write("Uploading the explanation of code ASAP in simple Lang" )
with col3:
    st.title("Code From Scratch")

    python_code = """
    X=np.random.rand(100,1)
    X_b = np.c_[np.ones((100, 1)), X]
    Y = (X > 0.5).astype(int)
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    def gradientdes(X,Y):
    theta=np.random.randn(2,1)
    learning_rate=0.1
    iterations=150
    for interation in range(iterations):
        z=np.dot(X,theta)
        y_pred= sigmoid(z)
        dcost=1/100*(np.dot(X.T,(y_pred - Y)))
        theta=theta-learning_rate*dcost
    return theta
    theta_f=gradientdes(X_b,Y)
    probabilty=np.dot(X_b,sigmoid(theta_f))
    final_pred=(probabilty>0.5).astype(int)
    #programmed by D20
    """
    st.code(python_code, language="python")
st.dataframe(df.head(10))