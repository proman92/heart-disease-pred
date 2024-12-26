import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# Load and preprocess the dataset
data = pd.read_csv("heart.csv")
X = data.drop(columns=['target'])
y = data['target']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
knn = KNeighborsClassifier(n_neighbors=3).fit(X_scaled, y)
nb = GaussianNB().fit(X_scaled, y)
dt = DecisionTreeClassifier(random_state=42).fit(X_scaled, y)
lr = LogisticRegression(random_state=42).fit(X_scaled, y)
svm = SVC(kernel='linear', probability=True, random_state=42).fit(X_scaled, y)

# ANN Model
y_categorical = to_categorical(y)
ann = Sequential([
    Input(shape=(X_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_scaled, y_categorical, epochs=10, batch_size=16, verbose=0)

# Streamlit app
st.title("Heart Disease Prediction")

st.sidebar.header("User Input Features")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the uploaded data
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(new_data)

    # Ensure the new data has the correct format
    try:
        new_data_scaled = scaler.transform(new_data)

        # Display predictions
        st.subheader("Prediction Results")

        st.write("**KNN Prediction:**", knn.predict(new_data_scaled))
        st.write("**Naive Bayes Prediction:**", nb.predict(new_data_scaled))
        st.write("**Decision Tree Prediction:**", dt.predict(new_data_scaled))
        st.write("**Logistic Regression Prediction:**", lr.predict(new_data_scaled))
        st.write("**SVM Prediction:**", svm.predict(new_data_scaled))
        y_pred_probs_ann = ann.predict(new_data_scaled)
        st.write("**ANN Prediction:**", np.argmax(y_pred_probs_ann, axis=1))

    except Exception as e:
        st.error(f"Error in processing data: {e}")
else:
    st.info("Please upload a CSV file containing your input data.")

st.sidebar.markdown("""
