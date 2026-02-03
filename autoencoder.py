import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import sys
import pickle


def load_dataset():
    """
    Loads the json data into an array of logs
    """
    with open("sample_dataset.json", "r") as f:
        data = json.load(f)

    if not data:
        raise Exception("No Dataset Found")
    
    # Example Features 
    # These can be changed based on the need
    features = []
    for e in data['data']:
        features.append([
            e["Roll_deg"],
            e["Pitch_deg"],
            e["Yaw_deg"],
            e["Heading_deg"],
            e["Vel_X_mps"],
            e["Vel_Y_mps"],
            e["Vel_Z_mps"],
            e["Speed_mps"],
            e["Steering_deg"],
            e["Throttle_pct"],
            e["Pos_X_m"],
            e["Pos_Y_m"],
            e["Pos_Z_m"],
            e["GPS_Lat"],
            e["GPS_Lon"],
            e["Vib_X"],
            e["Vib_Y"],
            e["Vib_Z"],
            e["Distance_To_Home_m"],
            e["Heading_To_Home_deg"],
            e["Heading_To_NextWP_deg"],
            e["YawRate_degps"]
        ])

    return features


def standarize_dataset(dataset):
    X = np.array(dataset, dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def create_model():

    # The first Input layer must match with how many features are being passed
    # Similarly the last layer should be of the same shape as the Input
    model = keras.Sequential([
        layers.Input(shape=(22,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(22, activation="linear")
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(model, X_train, X_val):
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        
        # Modifiable parameter based on user neeed
        epochs=100,
        batch_size=32,
        verbose=1
    )
    return history


def predict_test(model, X_val):

    X_pred = model.predict(X_val, verbose=0)
    
    
    # Calculate the reconstruction error
    # The higher the threshold, the more insensitive the model becomes
    # It is recommended to not lower the threshold too much to avoid
    # flagging turns, stops, etc., as false negatives

    errors = np.mean(np.square(X_val - X_pred), axis=1)
    threshold = np.percentile(errors, 95)
    anomaly = errors > threshold

    print(f"Threshold: {threshold:.4f}")
    print(f"Anomalies: {anomaly.sum()} / {len(anomaly)}")
    print(f"Anomaly rate: {100 * anomaly.sum() / len(anomaly):.2f}%")


if __name__ == "__main__":
    dataset = load_dataset()
    dataset, scaler = standarize_dataset(dataset)

    X_train, X_val = train_test_split(
        dataset, test_size=0.2, random_state=42
    )

    model = create_model()
    train_model(model, X_train, X_val)
    
    
    # This part is to avoid passing large dataset into testing 
    # in some cases where you don't want the model to be tested
    if len(sys.argv) >= 2 and sys.argv[1] == "1":
        predict_test(model, X_val)

    model_dir = os.path.expanduser("./model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "autoencoder.keras")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("Model saved to:", model_path)
    print("Scaler saved to:", scaler_path)

