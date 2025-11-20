import sys
import numpy as np
import pickle

# Load saved model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Receive input from Node.js
data = list(map(float, sys.argv[1:]))

# Convert to array
data_np = np.array(data).reshape(1, -1)

# Scale input
scaled = scaler.transform(data_np)

# Predict
prediction = model.predict(scaled)[0]

print(prediction)
