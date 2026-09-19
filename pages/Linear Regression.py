import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Real Linear Regression App", layout="wide")

st.title("Linear Regression Model")

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
col1, col2,col3 = st.columns([1,2,2])

with col1:
    st.subheader("Model Prediction")
    
    # Format inputs into a 2D array: shape (1, 2)
    user_data = np.array([[x1_input, x2_input]])
    
    # Predict using sklearn's predict method
    y_pred = model.predict(user_data)[0][0]
    
    st.metric(label="Predicted Output (Y)", value=f"{y_pred:.2f}")
    
    st.markdown("---")
    st.subheader("Learned Parameters")
    st.write(f"**Intercept ($\ theta_0$):** {model.intercept_[0]:.2f}")
    st.write(f"**Weight for $X_1$ ($\ theta_1$):** {model.coef_[0][0]:.2f}")
    st.write(f"**Weight for $X_2$ ($\ theta_2$):** {model.coef_[0][1]:.2f}")

with col2:
    st.subheader("Explanation")
    df = pd.DataFrame({
        'Feature X1': X_train[:, 0],
        'Feature X2': X_train[:, 1],
        'Target Y': Y_train.flatten()

    })
    st.write("Uploading the explanation of code ASAP in simple Lang" )
with col3:
    st.title("Code From Scratch")

    python_code = """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    X=np.random.rand(100,1)
    X_b = np.c_[np.ones((100, 1)), X]
    Y= 4+3*X
    #The above equation is taken as an example
    # Train model
    def gradientdes(X,Y):
        theta=np.random.randn(2,1)
        learning_rate=0.1
        iterations=150
        for interation in range(iterations):
            y_pred= np.dot(X,theta)
            dcost=2/100*(np.dot(X.T,(y_pred - Y)))
            theta=theta-learning_rate*dcost
    return theta
    #programmed by D20
    """
    st.code(python_code, language="python")

   
   
st.dataframe(df.head(10))