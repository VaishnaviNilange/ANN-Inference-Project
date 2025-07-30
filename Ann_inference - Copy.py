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
