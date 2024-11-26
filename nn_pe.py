import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from keras.api.layers import Dense, Input
from keras.api.models import Sequential
from keras.api.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np

# Step 1: Load the dataset
file_path = "./pe.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path, header=None)
# Assuming the class label is in the last column
X = data.iloc[:, :-1].values  # Features: All columns except the last
y = data.iloc[:, -1].values  # Target: The last column
indexes = np.where((y != 0) & (y != 3) & (y != 1))[0]
# Filtrar las filas de X y y usando los índices calculados
y = y[indexes]
X = X[indexes]
le = LabelEncoder()
y = le.fit_transform(y)

# relief = ReliefF(n_neighbors=11, n_features_to_select=36)

std = StandardScaler()
X = std.fit_transform(X)
##X = relief.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential(
    [
        Input(shape=(X_train.shape[1],)),  # Input layer
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(len(set(y)), activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # Use for integer-encoded labels
    metrics=["accuracy"],
)
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)
# Step 5: Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32,
    verbose=1,
)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred_classes)
print(accuracy)

model.save("model.keras")
