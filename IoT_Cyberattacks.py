# -*- coding: utf-8 -*-
"""
IoT Attack Detection with Multi-Class Handling - Final Robust Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('/home/bazzite/Downloads/RT_IOT2022.csv')

# Data Exploration
print("Original Attack Distribution:")
attack_counts = df['Attack_type'].value_counts()
print(attack_counts)

# Visualize class distribution
plt.figure(figsize=(12,6))
sns.barplot(x=attack_counts.index, y=attack_counts.values)
plt.title('Original Class Distribution')
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.show()

# Feature engineering
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Attack_type' in numeric_features:
    numeric_features.remove('Attack_type')

# Separate features and target
X = df[numeric_features]
y = df['Attack_type']

# Handle class imbalance
print("\nClass distribution before balancing:")
print(Counter(y))

# First encode ALL labels before any processing
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Automatic detection of normal traffic classes
possible_normal_terms = ['normal', 'mqtt', 'thing', 'wipro', 'legit', 'benign']
normal_classes = [
    cls for cls in y.unique() 
    if any(term in str(cls).lower() for term in possible_normal_terms)
]

if not normal_classes:
    majority_class = y.value_counts().idxmax()
    normal_classes = [majority_class]
    print(f"\nWARNING: No obvious normal classes found. Using majority class '{majority_class}' as normal traffic")

print(f"\nUsing these classes as normal traffic: {normal_classes}")

# Split into attack and normal samples using original labels
attack_mask = ~df['Attack_type'].isin(normal_classes)
normal_mask = df['Attack_type'].isin(normal_classes)

X_attack = df[attack_mask][numeric_features]
y_attack_encoded = y_encoded[attack_mask]

X_normal = df[normal_mask][numeric_features]
y_normal_encoded = y_encoded[normal_mask]

# Apply SMOTE only to attack samples
smote = SMOTE(random_state=42)
X_attack_resampled, y_attack_resampled = smote.fit_resample(X_attack, y_attack_encoded)

# Combine resampled attack with original normal samples
X_final = np.vstack((X_attack_resampled, X_normal))
y_final = np.hstack((y_attack_resampled, y_normal_encoded))

# Convert back to original labels for reporting
y_final_labels = label_encoder.inverse_transform(y_final)

print("\nClass distribution after balancing:")
print(Counter(y_final_labels))

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.3, random_state=42, stratify=y_final)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get normal samples for autoencoder training
normal_classes_encoded = label_encoder.transform(normal_classes)
normal_mask_train = np.isin(y_train, normal_classes_encoded)
X_train_normal = X_train_scaled[normal_mask_train]

if len(X_train_normal) == 0:
    X_train_normal = X_train_scaled
    print("\nWARNING: No normal samples in training set. Using all samples for autoencoder training")

print(f"\nNumber of samples for autoencoder training: {len(X_train_normal)}")

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(10,8))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], 
                hue=label_encoder.inverse_transform(y_train), 
                palette='viridis', alpha=0.6)
plt.title('PCA Visualization of IoT Traffic')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Enhanced Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced_subsample',
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(label_encoder.inverse_transform(y_test), 
                           label_encoder.inverse_transform(y_pred),
                           target_names=label_encoder.classes_))

# Confusion Matrix
plt.figure(figsize=(12,10))
cm = confusion_matrix(label_encoder.inverse_transform(y_test), 
                     label_encoder.inverse_transform(y_pred))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Important Features')
plt.show()

# Enhanced Autoencoder for Anomaly Detection
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(128, activation='selu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(64, activation='selu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(32, activation='selu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='selu')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(128, activation='selu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    return autoencoder

# Train autoencoder
autoencoder = build_autoencoder(X_train_scaled.shape[1])
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Adjust validation split based on available samples
val_split = min(0.2, (len(X_train_normal) - 100)/len(X_train_normal))
val_split = max(0.1, val_split)

print(f"\nAutoencoder training with {len(X_train_normal)} samples, validation split: {val_split:.2f}")

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=100,
    batch_size=min(256, len(X_train_normal)),
    validation_split=val_split,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Anomaly detection
def detect_anomalies(model, data, threshold):
    reconstructions = model.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    return mse > threshold

# Calculate threshold (95th percentile of training data)
reconstructions = autoencoder.predict(X_train_normal)
train_mse = np.mean(np.power(X_train_normal - reconstructions, 2), axis=1)
threshold = np.percentile(train_mse, 95)

# Test on all data
test_reconstructions = autoencoder.predict(X_test_scaled)
test_mse = np.mean(np.power(X_test_scaled - test_reconstructions, 2), axis=1)

# Visualize reconstruction errors
normal_test_mask = np.isin(y_test, normal_classes_encoded)
plt.figure(figsize=(10,6))
sns.histplot(test_mse[normal_test_mask], bins=50, kde=True, label='Normal')
sns.histplot(test_mse[~normal_test_mask], bins=50, kde=True, color='red', label='Attack')
plt.axvline(threshold, color='k', linestyle='--', label='Threshold')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.legend()
plt.show()

def hybrid_classification(rf_model, ae_model, X, threshold, label_encoder, normal_classes_encoded):
    # First use autoencoder to detect anomalies
    is_anomaly = detect_anomalies(ae_model, X, threshold)
    
    # Then use RF to classify
    predictions = rf_model.predict(X)
    
    # For anomalies detected by AE but classified as normal by RF, override
    anomaly_mask = is_anomaly & np.isin(predictions, normal_classes_encoded)
    
    # Create a special label for unknown anomalies
    # Find the maximum encoded value and add 1
    max_encoded = np.max(label_encoder.transform(label_encoder.classes_))
    new_label = max_encoded + 1
    
    predictions[anomaly_mask] = new_label
    
    return predictions, new_label

# Hybrid classification
y_pred_hybrid, unknown_label = hybrid_classification(rf, autoencoder, X_test_scaled, threshold, 
                                                   label_encoder, normal_classes_encoded)

# Prepare the class names for reporting
class_names = list(label_encoder.classes_) + ['Unknown_Anomaly']

# Create extended labels for inverse transform
y_test_labels = label_encoder.inverse_transform(y_test)

# For predictions, we need to handle the unknown label
y_pred_labels = []
for pred in y_pred_hybrid:
    if pred == unknown_label:
        y_pred_labels.append('Unknown_Anomaly')
    else:
        y_pred_labels.append(label_encoder.inverse_transform([pred])[0])

print("\nHybrid Classification Report:")
print(classification_report(y_test_labels, y_pred_labels,
                          target_names=class_names))

# Confusion Matrix for hybrid approach
plt.figure(figsize=(12,10))
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix (Hybrid Approach)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()