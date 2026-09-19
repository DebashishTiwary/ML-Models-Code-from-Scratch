import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Real Linear Regression App", layout="wide")

st.title("Streamlit App with Trained Linear Regression Model")

# 1. Generate Synthetic Training Data
@st.cache_data
def generate_data():
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10
    # True relationship with some random noise added
    noise = np.random.randn(100, 1)
    Y = 4 + 3 * X[:, 0:1] + 2 * X[:, 1:2] + noise
    return X, Y

X_train, Y_train = generate_data()

# 2. Train and Cache the Scikit-Learn Model
# We use @st.cache_resource because 'model' is an ML object, not a dataset!
@st.cache_resource
def train_model(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model

model = train_model(X_train, Y_train)

# 3. Sidebar Inputs for Prediction
st.sidebar.header("Input Features for Prediction")
x1_input = st.sidebar.slider("Select Feature X1:", 0.0, 10.0, 2.0, step=0.1)
x2_input = st.sidebar.slider("Select Feature X2:", 0.0, 10.0, 5.0, step=0.1)

# 4. Predict using the Trained Model
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Prediction")
    
    # Format inputs into a 2D array: shape (1, 2)
    user_data = np.array([[x1_input, x2_input]])
    
    # Predict using sklearn's predict method
    y_pred = model.predict(user_data)[0][0]
    
    st.metric(label="Predicted Output (Y)", value=f"{y_pred:.2f}")
    
    st.markdown("---")
    st.subheader("Learned Parameters")
    st.write(f"**Intercept ($\theta_0$):** {model.intercept_[0]:.2f}")
    st.write(f"**Weight for $X_1$ ($\theta_1$):** {model.coef_[0][0]:.2f}")
    st.write(f"**Weight for $X_2$ ($\theta_2$):** {model.coef_[0][1]:.2f}")

with col2:
    st.subheader("Training Data View")
    df = pd.DataFrame({
        'Feature X1': X_train[:, 0],
        'Feature X2': X_train[:, 1],
        'Target Y': Y_train.flatten()
    })
    st.dataframe(df.head(10))