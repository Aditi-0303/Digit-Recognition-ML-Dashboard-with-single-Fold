import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="ML Dashboard",
    layout="wide"
)

st.title("Digit Recognition ML Dashboard")

st.write("Machine Learning Dashboard using Random Forest")

st.markdown("---")

# -----------------------------------
# LOAD DATASET
# -----------------------------------

dataset = load_digits()

X = dataset.data
y = dataset.target

# -----------------------------------
# DATASET STATS
# -----------------------------------

st.header("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Images", len(X))
col2.metric("Total Classes", len(np.unique(y)))
col3.metric("Features", X.shape[1])

st.markdown("---")

# -----------------------------------
# MODEL
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=50,
    random_state=0
)

# -----------------------------------
# CROSS VALIDATION
# -----------------------------------

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

st.header("Cross Validation Scores")

for i, score in enumerate(scores):
    st.write(f"Fold {i+1} Accuracy : {score:.3f}")

st.success(f"Average Accuracy : {np.mean(scores):.3f}")

st.markdown("---")

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# IMAGE DISPLAY
# -----------------------------------

st.header("Image Display")

index = st.slider(
    "Select Image",
    0,
    len(X_test)-1,
    0
)

image = X_test[index]

actual = y_test[index]

fig, ax = plt.subplots()

ax.imshow(image.reshape(8,8), cmap='gray')

ax.axis('off')

st.pyplot(fig)

st.markdown("---")

# -----------------------------------
# PREDICTION OUTPUT
# -----------------------------------

prediction = model.predict([image])[0]

st.header("Prediction Output")

st.write(f"Predicted Digit : {prediction}")

st.write(f"Actual Digit : {actual}")

if prediction == actual:
    st.success("Prediction Correct")
else:
    st.error("Prediction Incorrect")

st.markdown("---")

# -----------------------------------
# ACCURACY GRAPH
# -----------------------------------

st.header("Accuracy Comparison Chart")

fig2, ax2 = plt.subplots()

labels = [
    "Fold1",
    "Fold2",
    "Fold3",
    "Fold4",
    "Fold5"
]

ax2.bar(labels, scores)

ax2.set_ylabel("Accuracy")

ax2.set_title("Cross Validation Accuracy")

st.pyplot(fig2)