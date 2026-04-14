import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import joblib

# ────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("dataset_phishing.csv")

print("\nDataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nDataset Info:")
df.info()
print("\nMissing Values per Column:")
print(df.isnull().sum())
print("\nClass Distribution (Target Variable):")
print(df['status'].value_counts())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed. New shape:", df.shape)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['status'])
print("\nLabel Mapping:")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

df = df.drop(['url', 'status'], axis=1)
print("\nColumns remaining after dropping text fields:", len(df.columns))

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

missing_before = df.isnull().sum().sum()
print(f"\nTotal missing values before filling: {missing_before}")
df = df.fillna(df.median())
missing_after = df.isnull().sum().sum()
print(f"Total missing values after filling: {missing_after}")

X = df.drop('label', axis=1)
y = df['label']
print("\nFeature Matrix Shape:", X.shape)

# ─────────────────────────────────────────────
# 3. DATASET SPLITTING
# ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Testing set shape:", X_test.shape)

# ─────────────────────────────────────────────
# 4. STANDARD SCALING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
sns.set(style="whitegrid")

plt.figure(figsize=(6, 5))
sns.countplot(x=y, palette="viridis")
plt.title("Class Distribution: Legitimate (0) vs Phishing (1)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("\nClass distribution:")
print(y.value_counts())

corr = pd.DataFrame(X)
corr['label'] = y
target_corr = corr.corr()['label'].sort_values(ascending=False)

print("\nTop correlations with target:")
print(target_corr.head(20))

top_features = target_corr.head(20).index

plt.figure(figsize=(10, 8))
sns.heatmap(corr[top_features].corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap (Top 20 Features Related to Phishing)")
plt.tight_layout()
plt.show()

important_cols = target_corr.head(6).index
corr[important_cols].hist(figsize=(12, 8), bins=30, color='blue')
plt.suptitle("Distribution of Top Influencing Features", fontsize=16)
plt.tight_layout()
plt.show()

variance = X.var()
low_variance_cols = variance[variance < 0.001].index
print("\nLow-variance columns (potentially useless):")
print(list(low_variance_cols))
print("Total low-variance columns:", len(low_variance_cols))

# ─────────────────────────────────────────────
# 6. FEATURE SELECTION
# ─────────────────────────────────────────────
feature_columns = X.columns

mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=SEED)
mi_scores = pd.Series(mi_scores, index=feature_columns).sort_values(ascending=False)

print("\nTop 20 Features by Mutual Information:")
print(mi_scores.head(20))

plt.figure(figsize=(10, 6))
mi_scores.head(20).plot(kind='bar', color='teal')
plt.title("Top 20 Features Ranked by Mutual Information")
plt.ylabel("MI Score")
plt.tight_layout()
plt.show()

rf_selector = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_selector.fit(X_train_scaled, y_train)

rf_importances = pd.Series(rf_selector.feature_importances_, index=feature_columns)
rf_importances = rf_importances.sort_values(ascending=False)

print("\nTop 20 Features by Random Forest Importance:")
print(rf_importances.head(20))

plt.figure(figsize=(10, 6))
rf_importances.head(20).plot(kind='bar', color='orange')
plt.title("Top 20 Features Ranked by Random Forest Importance")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

top_mi = set(mi_scores.head(20).index)
top_rf = set(rf_importances.head(20).index)
selected_features = list(top_mi.union(top_rf))

print("\nSelected Features (combined from MI + RF):")
print(selected_features)
print("Total selected:", len(selected_features))

X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
X_val_df   = pd.DataFrame(X_val_scaled,   columns=feature_columns)
X_test_df  = pd.DataFrame(X_test_scaled,  columns=feature_columns)

X_train_selected = X_train_df[selected_features]
X_val_selected   = X_val_df[selected_features]
X_test_selected  = X_test_df[selected_features]

print("\nShapes after feature selection:")
print("Train:", X_train_selected.shape)
print("Val:  ", X_val_selected.shape)
print("Test: ", X_test_selected.shape)

# ─────────────────────────────────────────────
# 7. EVALUATION FUNCTION
# ─────────────────────────────────────────────
def evaluate_model(model, X_data, y_data, model_name="Model", is_dnn=False):
    if is_dnn:
        y_pred_proba = model.predict(X_data)
        y_pred = (y_pred_proba > 0.5).astype('int32').flatten()
    else:
        y_pred = model.predict(X_data)

    acc  = accuracy_score(y_data, y_pred)
    prec = precision_score(y_data, y_pred)
    rec  = recall_score(y_data, y_pred)
    f1   = f1_score(y_data, y_pred)

    print(f"\n{'='*40}")
    print(f"{model_name} Evaluation")
    print(f"{'='*40}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_data, y_pred))

    cm = confusion_matrix(y_data, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    return acc, prec, rec, f1

# ═══════════════════════════════════════════════════════════
# MACHINE LEARNING MODELS
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 8. LOGISTIC REGRESSION
# ─────────────────────────────────────────────
log_reg = LogisticRegression(max_iter=1000, random_state=SEED)
log_reg.fit(X_train_selected, y_train)
lr_results = evaluate_model(log_reg, X_val_selected, y_val, "Logistic Regression")

# ─────────────────────────────────────────────
# 9. DECISION TREE
# ─────────────────────────────────────────────
dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train_selected, y_train)
dt_results = evaluate_model(dt, X_val_selected, y_val, "Decision Tree")

# ─────────────────────────────────────────────
# 10. RANDOM FOREST
# ─────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(X_train_selected, y_train)
rf_results = evaluate_model(rf, X_val_selected, y_val, "Random Forest")

# ─────────────────────────────────────────────
# 11. SUPPORT VECTOR MACHINE (SVM)
# ─────────────────────────────────────────────
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED)
svm.fit(X_train_selected, y_train)
svm_results = evaluate_model(svm, X_val_selected, y_val, "Support Vector Machine")

# ═══════════════════════════════════════════════════════════
# DEEP LEARNING MODEL
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 12. DEEP NEURAL NETWORK (DNN)
# ─────────────────────────────────────────────
input_dim = X_train_selected.shape[1]

dnn_model = Sequential([
    tf.keras.Input(shape=(input_dim,)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

dnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

dnn_model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = dnn_model.fit(
    X_train_selected, y_train,
    validation_data=(X_val_selected, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

dnn_results = evaluate_model(dnn_model, X_val_selected, y_val, "Deep Neural Network", is_dnn=True)

# ─────────────────────────────────────────────
# 13. VALIDATION RESULTS SUMMARY
# ─────────────────────────────────────────────
val_results = pd.DataFrame({
    "Model":     ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Deep Neural Network"],
    "Accuracy":  [lr_results[0], dt_results[0], rf_results[0], svm_results[0], dnn_results[0]],
    "Precision": [lr_results[1], dt_results[1], rf_results[1], svm_results[1], dnn_results[1]],
    "Recall":    [lr_results[2], dt_results[2], rf_results[2], svm_results[2], dnn_results[2]],
    "F1-Score":  [lr_results[3], dt_results[3], rf_results[3], svm_results[3], dnn_results[3]]
})

print("\n" + "="*60)
print("VALIDATION RESULTS SUMMARY")
print("="*60)
print(val_results.to_string(index=False))

# ─────────────────────────────────────────────
# 14. FINAL TEST SET EVALUATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

lr_test  = evaluate_model(log_reg,   X_test_selected, y_test, "Logistic Regression - TEST")
dt_test  = evaluate_model(dt,        X_test_selected, y_test, "Decision Tree - TEST")
rf_test  = evaluate_model(rf,        X_test_selected, y_test, "Random Forest - TEST")
svm_test = evaluate_model(svm,       X_test_selected, y_test, "Support Vector Machine - TEST")
dnn_test = evaluate_model(dnn_model, X_test_selected, y_test, "Deep Neural Network - TEST", is_dnn=True)

test_results = pd.DataFrame({
    "Model":     ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Deep Neural Network"],
    "Accuracy":  [lr_test[0], dt_test[0], rf_test[0], svm_test[0], dnn_test[0]],
    "Precision": [lr_test[1], dt_test[1], rf_test[1], svm_test[1], dnn_test[1]],
    "Recall":    [lr_test[2], dt_test[2], rf_test[2], svm_test[2], dnn_test[2]],
    "F1-Score":  [lr_test[3], dt_test[3], rf_test[3], svm_test[3], dnn_test[3]]
})

print("\n" + "="*60)
print("FINAL TEST RESULTS SUMMARY")
print("="*60)
print(test_results.to_string(index=False))

# ─────────────────────────────────────────────
# 15. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
rf_feat_imp = pd.Series(rf.feature_importances_, index=selected_features)
rf_feat_imp = rf_feat_imp.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
rf_feat_imp.head(15).plot(kind='bar', color='steelblue')
plt.title("Top 15 Most Important Features (Random Forest)")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

print("\nTop 15 Most Important Features:")
print(rf_feat_imp.head(15))

# ─────────────────────────────────────────────
# 16. SAVE MODELS
# ─────────────────────────────────────────────
ml_models = {
    "Logistic Regression": (log_reg, lr_test),
    "Decision Tree":       (dt,      dt_test),
    "Random Forest":       (rf,      rf_test),
    "SVM":                 (svm,     svm_test)
}

best_name = max(ml_models, key=lambda k: ml_models[k][1][3])
best_model_obj = ml_models[best_name][0]
best_f1 = ml_models[best_name][1][3]

print(f"\nBest ML Model: {best_name}")
print(f"Best F1 Score (Test): {best_f1:.4f}")

joblib.dump(best_model_obj, "best_ml_model.pkl")
print("Best ML model saved as best_ml_model.pkl")

dnn_model.save("phishing_dnn_model.keras")
print("Deep learning model saved as phishing_dnn_model.keras")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "selected_features.pkl")
print("Scaler saved as scaler.pkl")
print("Selected features saved as selected_features.pkl")

# ═══════════════════════════════════════════════════════════
# ROC-AUC CURVES
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 17. ROC-AUC CURVES FOR ALL MODELS
# ─────────────────────────────────────────────
colors = {
    "Logistic Regression": "#9b59b6",
    "Decision Tree":       "#e6a817",
    "Random Forest":       "#3266ad",
    "SVM":                 "#e05c2e",
    "Deep Neural Network": "#2a9d5c"
}

plt.figure(figsize=(8, 6))

roc_models = {
    "Logistic Regression": log_reg,
    "Decision Tree":       dt,
    "Random Forest":       rf,
    "SVM":                 svm
}

for name, model in roc_models.items():
    y_prob = model.predict_proba(X_test_selected)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[name], lw=2,
             label=f"{name} (AUC = {roc_auc:.4f})")

dnn_prob = dnn_model.predict(X_test_selected).flatten()
fpr_dnn, tpr_dnn, _ = roc_curve(y_test, dnn_prob)
roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
plt.plot(fpr_dnn, tpr_dnn, color=colors["Deep Neural Network"], lw=2,
         label=f"Deep Neural Network (AUC = {roc_auc_dnn:.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random (AUC = 0.50)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curves — All Models")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()

print("\nROC-AUC Scores:")
for name, model in roc_models.items():
    y_prob = model.predict_proba(X_test_selected)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    print(f"{name}: AUC = {auc(fpr, tpr):.4f}")
print(f"Deep Neural Network: AUC = {roc_auc_dnn:.4f}")

# ═══════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 18. 5-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("="*60)

X_scaled_full = scaler.transform(X)
X_full_df = pd.DataFrame(X_scaled_full, columns=feature_columns)
X_full_selected = X_full_df[selected_features]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
    "SVM":                 SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED)
}

cv_results = []

for name, model in cv_models.items():
    scores = cross_validate(
        model, X_full_selected, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        n_jobs=-1
    )

    cv_results.append({
        "Model":     name,
        "Accuracy":  f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        "Precision": f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
        "Recall":    f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        "F1-Score":  f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}"
    })

    print(f"\n{name}:")
    print(f"  Accuracy : {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
    print(f"  Precision: {scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}")
    print(f"  Recall   : {scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}")
    print(f"  F1-Score : {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")

cv_df = pd.DataFrame(cv_results)

print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY TABLE")
print("="*60)
print(cv_df.to_string(index=False))

print("\nAll done!")
