import os
print("Current Directory:", os.getcwd())
print("Files here:", os.listdir())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('data/hotel_bookings.csv')

# Select features
df = df[['lead_time','adr','adults','children','previous_cancellations','is_canceled']]

# Handle missing values
df = df.fillna(0)

# Features and target
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open('model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))