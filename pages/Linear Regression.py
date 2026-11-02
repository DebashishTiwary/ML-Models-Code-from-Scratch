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


# 4. Predict using the Trained Model
col1, col2= st.columns([2,2])

with col1:
    st.header("Input Features for Prediction")
    x1_input = st.slider("Select Feature X1:", 0.0, 10.0, 2.0, step=0.1)
    x2_input = st.slider("Select Feature X2:", 0.0, 10.0, 5.0, step=0.1)
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

    df = pd.DataFrame({
        'Feature X1': X_train[:, 0],
        'Feature X2': X_train[:, 1],
        'Target Y': Y_train.flatten()

    })

explanation,code,data = st.tabs(["Explanation","Code","Data"])
with explanation:
    st.markdown("""## 1. Main Idea and Approach

Imagine you are blindfolded on a bumpy hillside and want to walk down to the lowest valley. Since you can't see, you feel the ground with your feet, figure out which way slopes downward, and take a small step in that direction. You repeat this over and over until you reach the bottom.

This Python code uses the exact same idea—called **Gradient Descent**—to teach a computer how to discover a hidden rule behind some data.

* **The Goal:** Find the best straight-line equation ($Y = aX + b$) that fits a set of data points.
* **The Logic:** The computer starts with a completely random guess. It checks how wrong its guess is, calculates how to fix it, and adjusts its guess step-by-step until it gets as close to the correct answer as possible.

---

## 2. Brief Explanation of the Code

Here is what each part of the code does in simple terms:

### Generating the Data

* **`X = np.random.rand(100, 1)`**: Creates 100 random numbers (think of these as input values, like hours spent studying).
* **`Y = 4 + 3 * X`**: Generates corresponding output values (like exam scores) using a secret rule: multiply the input by 3 and add 4.
* **`X_b = ...`**: Prepares the data so the computer can handle the starting point (the "+ 4" part of the equation).

### Training the Model (`gradientdes` function)

* **`theta = np.random.randn(2, 1)`**: The computer starts with two completely random guesses for our secret rule.
* **`learning_rate = 0.1`**: Controls the size of the steps the computer takes when adjusting its guesses. If the step is too big, it overshoots; if it's too small, it takes forever.
* **`iterations = 150`**: The computer repeats the learning process 150 times to fine-tune its guesses.
* **Inside the loop:**
* **`y_pred = ...`**: The computer makes a prediction using its current guesses.
* **`dcost = ...`**: It measures how far off its prediction is from the real answers.
* **`theta = ...`**: It updates its guesses by taking a small step in the right direction to reduce future mistakes.
""",unsafe_allow_html=True)
with code:
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
with data:
    st.dataframe(df.head(10))