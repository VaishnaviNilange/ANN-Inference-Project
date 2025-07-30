ANN-Inference-Project
This project uses an Artificial Neural Network (ANN) to predict whether a bank customer will leave the bank (churn) or not, based on various input features.
Customer Churn Prediction using Artificial Neural Networks (ANN)

This project uses an Artificial Neural Network (ANN) to predict whether a bank customer will leave the bank (churn) or not, based on various input features.

🔍 Overview

- Built and trained an ANN using TensorFlow and Keras.
- Used StandardScaler for preprocessing.
- Saved and loaded the trained model (`ANN.h5`) and scaler (`scaler.pkl`) for inference.
- Performed inference on new data and predicted churn outcome.
-  📂 Files

- `Ann_inference.py` – Loads the scaler and ANN model, runs inference on test input.
- `ANN.h5` – Trained ANN model.
- `scaler.pkl` – Pre-fitted StandardScaler for consistent input scaling.
- `requirements.txt` – Python package dependencies.

 🧠 Sample Output
 model loaded>>>>>>>>>>>>>>>>>>>>>>>>>>>>
final output: [0]
customer won't exit the bank
⚙️ Tech Stack

- Python
- TensorFlow
- scikit-learn
- NumPy & Pandas

✅ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ann-customer-churn.git
   cd ann-customer-churn
2.Create a virtual environment:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
Install the requirements:

bash
Copy
Edit
pip install -r requirements.txt
Run the inference script:

bash
Copy
Edit
python Ann_inference.py


 # CODE
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow import keras
with open("scaler.pkl",'rb') as f:
    sc=pickle.load(f)
model=keras.models.load_model(r"ANN.h5")
print("model loaded>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
def exit_predication(credit_score,age,tenure,balance, num_products,hascard,is_active_member,estimated_salary,germany,spain,male):
    pred=model.predict(sc.transform([[credit_score,age,tenure,balance, num_products,hascard,is_active_member,estimated_salary,germany,spain,male]]))
    print(pred)
    output=[1 if y > 0.6 else 0 for y in pred]
    print("final output",output)
    if output[0] == 1:
        print("customer exit the bank")
    
    else:
        print("customer won't exit the bank")

exit_predication(120, 35, 2, 500000, 1, 1, 1, 100000, 0, 0, 1)
