import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt # Import for plotting

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- 1. Data Simulation Function (Real-Time Scenario) ---
def simulate_customer_data(n_samples=2000):
    """
    Generates synthetic data for customer classification (New vs. Existing)
    Target (Y): 0 = New Customer (Signup), 1 = Existing Customer (Login)
    """
    # 1. Target (55% Existing, 45% New)
    y = np.random.binomial(1, 0.55, n_samples)
    df = pd.DataFrame({'is_existing': y})

    # 2. Time of Day (0-23) - New users favor peak hours (18-22)
    df['time_of_day'] = np.random.randint(0, 24, n_samples)
    
    # Using .loc with .values to avoid FutureWarnings and ensure consistent dtype
    new_user_indices = (df['is_existing'] == 0)
    time_of_day_new = df.loc[new_user_indices, 'time_of_day'].values + np.random.normal(0, 5, new_user_indices.sum())
    df.loc[new_user_indices, 'time_of_day'] = np.clip(time_of_day_new, 0, 23).astype(int)

    # 3. Device Type (0=Mobile/Tablet, 1=Desktop)
    # Existing users slightly favor desktop
    df['device_type_desktop'] = np.random.binomial(1, 0.6, n_samples)
    df.loc[df['is_existing'] == 0, 'device_type_desktop'] = np.random.binomial(1, 0.45, (df['is_existing'] == 0).sum())

    # 4. Session Duration (Seconds) - New users spend slightly longer
    df['session_duration_sec'] = np.random.normal(loc=120, scale=40, size=n_samples)
    df.loc[df['is_existing'] == 0, 'session_duration_sec'] = np.random.normal(loc=150, scale=60, size=(df['is_existing'] == 0).sum())
    df['session_duration_sec'] = np.clip(df['session_duration_sec'], 10, 600).astype(int)

    # 5. Referral Source (0=Direct/Search, 1=Social Media Campaign)
    # New users heavily influenced by social campaign
    df['referral_source_social'] = np.random.binomial(1, 0.2, n_samples)
    df.loc[df['is_existing'] == 0, 'referral_source_social'] = np.random.binomial(1, 0.7, (df['is_existing'] == 0).sum())

    # Final features and target
    X = df[['time_of_day', 'device_type_desktop', 'session_duration_sec', 'referral_source_social']]
    y = df['is_existing']

    return X, y

# Generate the data
X, y = simulate_customer_data()
print(f"Dataset generated: {X.shape[0]} samples")
print(X.head())

# --- 2. Preprocessing and Feature Engineering ---

# One-Hot Encode Time of Day (0-23 hours)
X_processed = pd.get_dummies(X, columns=['time_of_day'], prefix='hour')

# CRITICAL FIX: Ensure all columns are explicitly float32 before converting to NumPy
# This resolves the 'ValueError: Invalid dtype: object'
X_data = X_processed.astype(np.float32).values
y_data = y.values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Scale Numerical Features (Session Duration is the only un-encoded numerical feature)
# Find the index of the 'session_duration_sec' column after one-hot encoding
duration_index = X_processed.columns.get_loc('session_duration_sec')

scaler = StandardScaler()
# Note: Reshape(-1, 1) is necessary for the scaler to work on a single column
X_train[:, duration_index] = scaler.fit_transform(X_train[:, duration_index].reshape(-1, 1)).flatten()
X_test[:, duration_index] = scaler.transform(X_test[:, duration_index].reshape(-1, 1)).flatten()

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Number of features after encoding: {X_train.shape[1]}")

# --- 3. Build the TensorFlow Keras Model (Binary Classification) ---

INPUT_DIM = X_train.shape[1]

model = Sequential([
    # Input layer and first hidden layer
    Dense(64, activation='relu', input_shape=(INPUT_DIM,), name='dense_1'),
    Dropout(0.3, name='dropout_1'),

    # Second hidden layer
    Dense(32, activation='relu', name='dense_2'),
    Dropout(0.3, name='dropout_2'),

    # Output layer for Binary Classification
    # Uses 1 neuron and 'sigmoid' activation (output between 0 and 1)
    Dense(1, activation='sigmoid', name='output')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Standard loss for binary classification
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("\nModel Architecture Summary:")
model.summary()

# --- 4. Train the Model ---

# Use Early Stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("\nStarting model training...")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=0 # Run silently for clean output
)

print(f"Training finished after {len(history.history['loss'])} epochs.")

# --- 5. Plot Training History Function ---
def plot_training_history(history):
    """Plots and saves the training and validation loss and accuracy."""
    print("\nGenerating training graphs...")
    
    # 1. Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    print("-> Saved training_loss.png")

    # 2. Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_accuracy.png')
    plt.close()
    print("-> Saved training_accuracy.png")


# --- 6. Evaluation and Real-Time Application ---

# Evaluate the model on the test set
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)

print("\n--- Model Evaluation ---")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

# Generate and save the training history plots
plot_training_history(history)

# --- Real-Time Prediction Simulation ---
print("\n--- Real-Time Prediction Simulation ---")

# Example 1: High chance of being an Existing Customer (Late time, Desktop, Short session)
sample_existing = pd.DataFrame({
    'time_of_day': [21], 'device_type_desktop': [1],
    'session_duration_sec': [60], 'referral_source_social': [0]
})

# Example 2: High chance of being a New Customer (Mid-day, Mobile, Long session, Social referral)
sample_new = pd.DataFrame({
    'time_of_day': [15], 'device_type_desktop': [0],
    'session_duration_sec': [250], 'referral_source_social': [1]
})

def predict_customer_type(sample_df, scaler, model, original_columns, duration_index):
    """Preprocesses a single sample and predicts customer type."""
    # Apply One-Hot Encoding consistent with training data
    sample_encoded = pd.get_dummies(sample_df, columns=['time_of_day'], prefix='hour')

    # Reindex to ensure all 24 hour columns exist, filling missing with 0
    missing_cols = set(original_columns) - set(sample_encoded.columns)
    for c in missing_cols:
        sample_encoded[c] = 0
    
    # Ensure column order matches training data
    sample_encoded = sample_encoded[original_columns]

    # Convert to float32 before passing to NumPy array
    sample_data = sample_encoded.astype(np.float32).values 
    
    # Scale the session duration feature (at the correct index)
    sample_data[:, duration_index] = scaler.transform(sample_data[:, duration_index].reshape(-1, 1)).flatten()

    # Predict probability (0 to 1)
    prediction = model.predict(sample_data, verbose=0)[0][0]
    
    # Classify (0 = New, 1 = Existing)
    prediction_class = 1 if prediction > 0.5 else 0
    
    label = "Existing Customer (Login)" if prediction_class == 1 else "New Customer (Signup)"
    
    print(f"Features: TOD={sample_df['time_of_day'].iloc[0]}, Desktop={bool(sample_df['device_type_desktop'].iloc[0])}, Duration={sample_df['session_duration_sec'].iloc[0]}s, Social={bool(sample_df['referral_source_social'].iloc[0])}")
    print(f"Predicted Probability of being Existing: {prediction:.4f}")
    print(f"Predicted Customer Type: {label}")
    print("-" * 50)
    
    return prediction_class

# Run predictions
print("Simulating prediction for a likely Existing Customer:")
predict_customer_type(sample_existing, scaler, model, X_processed.columns, duration_index)

print("Simulating prediction for a likely New Customer:")
predict_customer_type(sample_new, scaler, model, X_processed.columns, duration_index)

# --- 7. Model Saving (for E-commerce Integration) ---

model_save_path = 'customer_type_classifier.keras'
model.save(model_save_path)
print(f"\nModel saved successfully for deployment: {model_save_path}")

print("\nTo integrate this model in a real-time environment, you would load the model, load the scaler, and pass the four input features to the 'predict_customer_type' function.")
