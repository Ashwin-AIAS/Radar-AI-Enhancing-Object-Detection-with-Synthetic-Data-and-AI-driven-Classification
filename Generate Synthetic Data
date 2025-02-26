import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# **Load and Fix RADAR Dataset**
# Simulating synthetic RADAR dataset with corrected values
np.random.seed(42)
num_samples = 200

radar_data = pd.DataFrame({
    "x": np.random.uniform(0, 50, num_samples),
    "y": np.random.uniform(0, 50, num_samples),
    "doppler_velocity": np.random.uniform(-5, 5, num_samples),
    "reflectivity": np.random.uniform(0.5, 1.0, num_samples),  # Ensuring valid values
    "class": np.random.choice(["Car", "Pedestrian", "Static Object"], num_samples)
})

#  **Encode Labels**
class_mapping = {"Car": 0, "Pedestrian": 1, "Static Object": 2}
radar_data["class"] = radar_data["class"].map(class_mapping)

#  **Ensure No NaN Values**
print("Any NaN in dataset?", radar_data.isna().sum())

# **Normalize Data**
scaler = MinMaxScaler()
X = scaler.fit_transform(radar_data.drop(columns=["class"]))
y = radar_data["class"].values

#  **Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Check Shapes**
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

#  **Build CNN Model**
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Conv1D(64, kernel_size=2, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 Classes
])

#  **Compile Model**
optimizer = Adam(learning_rate=0.0001)  # Reduced LR for stability
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# **Train Model**
X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)

history = model.fit(X_train_cnn, y_train, epochs=50, batch_size=8, validation_data=(X_test_cnn, y_test))

#  **Evaluate Model**
test_loss, test_acc = model.evaluate(X_test_cnn, y_test)
print(f"Final CNN Model Accuracy: {test_acc:.2f}")

#  **Confusion Matrix**
y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="coolwarm", fmt="d", xticklabels=class_mapping.keys(), yticklabels=class_mapping.keys())
plt.title("Confusion Matrix - Final Optimized CNN Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

#  **Classification Report**
print(classification_report(y_test, y_pred, target_names=class_mapping.keys()))

#  **Save Model**
model.save("trained_radar_cnn_model.h5")

# **Simulated AI Agent for Real-time Radar Decisions**
class RadarAI_Agent:
    def __init__(self, model):
        self.model = model

    def run_agent(self, radar_stream):
        for radar_input in radar_stream:
            radar_input = np.array(radar_input).reshape(1, 5, 1)  # Adjust input shape
            prediction = self.model.predict(radar_input)
            predicted_class = np.argmax(prediction)
            
            class_labels = {0: "Car", 1: "Pedestrian", 2: "Static Object"}
            action = "Continue monitoring speed & position" if predicted_class == 0 else "Caution: Possible Pedestrian!"
            
            print(f"🚗 Detected: {class_labels[predicted_class]} | Action: {action}")

#  **Run AI Agent on Simulated Data Stream**
simulated_radar_stream = [
    [10, 15, 3.0, 0.85, 0],   # Car
    [22, 25, -1.2, 0.75, 1],  # Pedestrian
    [32, 35, 0.0, 0.90, 2],   # Static Object
    [10, 12, 2.5, 0.88, 0],   # Car
    [20, 23, -2.0, 0.70, 1]   # Pedestrian
]

ai_agent = RadarAI_Agent(model)
ai_agent.run_agent(simulated_radar_stream)
