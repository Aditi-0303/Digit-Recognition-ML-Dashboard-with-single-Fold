# Digit Recognition ML Dashboard

An interactive Machine Learning dashboard built using Random Forest and Streamlit for handwritten digit recognition. The project demonstrates dataset visualization, fold-wise testing, prediction analysis, cross-validation, and decision tree visualization using the Digits dataset from Scikit-learn.

---

## Features

* Dataset statistics display
* Handwritten digit dataset visualization
* Random Forest classification
* 5-Fold Cross Validation
* Fold-wise iteration visualization
* Test image selection from each fold
* Prediction output display
* Accuracy comparison graph
* Decision tree visualization
* Interactive Streamlit dashboard

---

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* NumPy
* Matplotlib

---

## Dataset Used

This project uses the Digits Dataset from Scikit-learn.

Dataset Details:

* 1797 handwritten digit images
* Digits from 0–9
* 8x8 grayscale images
* 64 pixel-based features per image

---

## How the Project Works

1. The dataset is loaded using Scikit-learn.
2. Images are converted into pixel feature values.
3. A Random Forest model is trained on the dataset.
4. 5-Fold Cross Validation is performed.
5. During each iteration, one fold is used as testing data.
6. The model predicts unseen handwritten digit images.
7. Results are displayed through an interactive dashboard.

---

## Cross Validation Process

The dataset is divided into 5 folds.

For every iteration:

* 4 folds are used for training
* 1 fold is used for testing

This process repeats until all folds are tested once.

---

## How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/Aditi-0303/Digit-Recognition-Model.git
```

---

### Step 2: Open Project Folder

```bash
cd Digit-Recognition-Model
```

---

### Step 3: Create Virtual Environment

```bash
py -3.11 -m venv venv
```

---

### Step 4: Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

---

### Step 5: Install Required Libraries

```bash
python -m pip install numpy matplotlib
```

```bash
python -m pip install scikit-learn==1.4.2 --no-deps
```

```bash
python -m pip install scipy joblib threadpoolctl
```

```bash
python -m pip install streamlit
```

---

### Step 6: Run the Dashboard

```bash
python -m streamlit run app.py
```

---

### Step 7: Open Browser

If browser does not open automatically:

```text
http://localhost:8501
```

---

## Project Structure

```text
Digit-Recognition-Model/
│
├── app.py
├── README.md
├── .gitignore
```

---

## Future Improvements

* Real-time handwritten digit recognition
* Upload custom handwritten digit images
* Deep learning model integration
* Explainable AI visualization
* Enhanced dashboard UI

---

## Author

Aditi

