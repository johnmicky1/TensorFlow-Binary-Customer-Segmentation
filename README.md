# 🧠 TensorFlow Customer Classification Model Guide

**GitHub Link:** [TensorFlow-Binary-Customer-Segmentation](git@github.com:johnmicky1/TensorFlow-Binary-Customer-Segmentation.git)

---

## 📘 Overview
This professional guide walks you through the setup, development, and execution of a TensorFlow-based Customer Classification Model.  
The model uses simulated behavioral data to predict whether a customer is *new* (sign-up) or *existing* (login) based on their activity patterns.  
It is ideal for e-commerce, marketing analytics, or any customer segmentation use case.

---

## 🧰 Prerequisites

Before you begin, make sure you have the following tools installed:

- 🐍 Python 3.10+ (latest version recommended)  
- 🤖 TensorFlow library  
- 🔢 NumPy and Pandas for data handling  
- 📊 Matplotlib for visualization  
- 💻 Code editor or IDE (VS Code, PyCharm, Jupyter Notebook, etc.)  

**Install TensorFlow and NumPy using the commands below:**

```bash
pip install tensorflow
pip install numpy
```

---

## 🪜 Step 1: Project Setup

1. Create a new folder or directory named `TensorFlow-Customer-Classification-Model`  
2. Open your preferred code editor (VS Code, Notepad++, etc.)  
3. Create a new Python file named `tensorflow-customer-classification-model.py` and paste the code below.

```python
# (Full Python script as provided in the document)
# Includes data simulation, preprocessing, model training, and prediction
```

---

## ▶️ Step 2: Run the Code

1. Right-click the folder `TensorFlow-Customer-Classification-Model` → **Open in Terminal**  
2. PowerShell or terminal will open.  
3. Verify TensorFlow installation and GPU availability:

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
ls
python tensorflow-customer-classification-model.py
```

---

## 📊 Step 3: Model Training and Evaluation

During execution, the script will automatically:
- Generate and preprocess customer data  
- Train the TensorFlow Keras model using 64 and 32 neuron dense layers  
- Apply dropout regularization to reduce overfitting  
- Evaluate performance using Accuracy, Precision, and Recall metrics  

**Training history graphs will be saved as:**
- `training_loss.png`
- `training_accuracy.png`

---

## 🔮 Step 4: Prediction Examples

Two prediction simulations are included in the script:

1. **Existing Customer** — Late time (21:00), Desktop device, short session  
2. **New Customer** — Midday time (15:00), Mobile device, long session, social referral  

---

## 💾 Step 5: Model Saving and Deployment

The trained model is automatically saved as:

```
customer_type_classifier.keras
```

This file can be loaded into any TensorFlow environment for real-time deployment.

---

## 📁 Step 6: Output Files

After successful execution, you will find the following files:

- `customer_type_classifier.keras` — Trained TensorFlow model  
- `training_loss.png` — Loss curve visualization  
- `training_accuracy.png` — Accuracy curve visualization  

---

## 🧠 Step 7: Integration Hint

For real-world integration, load the saved model and scaler, then pass input features:

```
time_of_day, device_type_desktop, session_duration_sec, referral_source_social
```

to the `predict_customer_type()` function to classify users dynamically.

---

✅ **End of Guide** — TensorFlow Customer Classification Model successfully documented.

**GitHub Link:** [git@github.com:johnmicky1/TensorFlow-Binary-Customer-Segmentation.git](git@github.com:johnmicky1/TensorFlow-Binary-Customer-Segmentation.git)
