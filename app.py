import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

data = load_data()
st.title("Heart Disease Prediction and Analysis")
st.sidebar.title("Input New Patient Data")

# Data Preprocessing
X = data.drop(columns=['target'])
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train models
def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn

def train_nb(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

def train_dt(X_train, y_train):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    return dt

def train_lr(X_train, y_train):
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    return lr

def train_svm(X_train, y_train):
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm

def train_ann(X_train, y_train):
    y_train_ann = to_categorical(y_train)
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_ann, epochs=50, batch_size=16, verbose=0)
    return model

# Train models
knn_model = train_knn(X_train, y_train)
nb_model = train_nb(X_train, y_train)
dt_model = train_dt(X_train, y_train)
lr_model = train_lr(X_train, y_train)
svm_model = train_svm(X_train, y_train)
ann_model = train_ann(X_train, y_train)

# Input new patient data
st.sidebar.header("Enter Patient Details")
def user_input_features():
    features = {}
    for col in X.columns:
        features[col] = st.sidebar.number_input(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    return pd.DataFrame([features])

new_data = user_input_features()

if st.sidebar.button("Analyze New Data"):
    new_data_scaled = scaler.transform(new_data)
    new_data_pca = pca.transform(new_data_scaled)

    # Predictions
    st.subheader("Prediction Results")
    predictions = {
        "KNN": knn_model.predict(new_data_scaled),
        "Naive Bayes": nb_model.predict(new_data_scaled),
        "Decision Tree": dt_model.predict(new_data_scaled),
        "Logistic Regression": lr_model.predict(new_data_scaled),
        "SVM": svm_model.predict(new_data_scaled),
        "ANN": np.argmax(ann_model.predict(new_data_scaled), axis=1)
    }

    for model, prediction in predictions.items():
        st.write(f"{model}: {'Disease' if prediction[0] == 1 else 'No Disease'}")

    # Detailed Metrics
    st.subheader("Model Performance on Test Data")
    metrics = {}
    for model_name, model in zip(
        ["KNN", "Naive Bayes", "Decision Tree", "Logistic Regression", "SVM"],
        [knn_model, nb_model, dt_model, lr_model, svm_model]
    ):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        metrics[model_name] = [acc, prec, recall, specificity, f1]

    metrics_df = pd.DataFrame(metrics, index=["Accuracy", "Precision", "Recall", "F1-Score"])
    st.dataframe(metrics_df)

st.subheader("Dataset Preview")
st.write(data)
