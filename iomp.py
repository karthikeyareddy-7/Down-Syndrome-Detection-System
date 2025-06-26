import GEOparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Step 1: Download and Parse GSE640218 Dataset (Down Syndrome)
print("Downloading GSE6408 dataset (Down Syndrome)...")
gse6408 = GEOparse.get_GEO("GSE6408", destdir="./")
print("Parsing GSE6408 expression data...")
expression_data_ds = {
    gsm_id: gsm_obj.table["VALUE"].values
    for gsm_id, gsm_obj in gse6408.gsms.items()
}
expression_data_ds = pd.DataFrame(expression_data_ds)

# Step 2: Download and Parse GSE9321 Dataset (Healthy Controls)
print("Downloading GSE9321 dataset (Healthy Controls)...")
gse9321 = GEOparse.get_GEO("GSE9321", destdir="./")
print("Parsing GSE9321 expression data...")
expression_data_control = {
    gsm_id: gsm_obj.table["VALUE"].values
    for gsm_id, gsm_obj in gse9321.gsms.items()
}
expression_data_control = pd.DataFrame(expression_data_control)

# Step 3: Align gene counts
print("Aligning datasets...")
min_genes = min(expression_data_ds.shape[0], expression_data_control.shape[0])
expression_data_ds = expression_data_ds.iloc[:min_genes, :]
expression_data_control = expression_data_control.iloc[:min_genes, :]

# Step 4: Balance sample counts
print("Balancing number of samples...")
n_ds_samples = expression_data_ds.shape[1]
expression_data_control = expression_data_control.sample(n=n_ds_samples, random_state=42, replace=True)
expression_data_control = expression_data_control.reset_index(drop=True)
expression_data_ds = expression_data_ds.reset_index(drop=True)

# Step 5: Merge and label
print("Merging datasets and creating labels...")
merged_expression_data = pd.concat([expression_data_ds, expression_data_control], axis=0)
labels_ds = np.array([1] * expression_data_ds.shape[0])
labels_control = np.array([0] * expression_data_control.shape[0])
merged_labels = np.concatenate([labels_ds, labels_control])

# Step 6: Clean data
print("Cleaning data...")
merged_expression_data = merged_expression_data.apply(pd.to_numeric, errors='coerce')
merged_expression_data = merged_expression_data.fillna(merged_expression_data.mean())

# Step 7: Add synthetic noise
print("Adding synthetic noise to data...")
np.random.seed(42)
noise = np.random.normal(0, 0.1, merged_expression_data.shape)
synthetic_data = merged_expression_data + noise

# Step 8: Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(synthetic_data)
y = merged_labels

# Step 9: Split
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Step 10: SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ---------------------- Train Models ----------------------
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train_resampled, y_train_resampled)

print("\nTraining XGBoost...")
xgb_model = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)

print("\nTraining LightGBM...")
lgbm_model = LGBMClassifier(random_state=42, n_jobs=-1)
lgbm_model.fit(X_train_resampled, y_train_resampled)

# ---------------------- Evaluation ----------------------
print("\nEvaluating Random Forest...")
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print("Accuracy:", rf_acc)
print(classification_report(y_test, rf_preds))

print("\nEvaluating XGBoost...")
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print("Accuracy:", xgb_acc)
print(classification_report(y_test, xgb_preds))

print("\nEvaluating LightGBM...")
lgbm_preds = lgbm_model.predict(X_test)
lgbm_acc = accuracy_score(y_test, lgbm_preds)
print("Accuracy:", lgbm_acc)
print(classification_report(y_test, lgbm_preds))


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)
import pandas as pd
import os

# Evaluate each model
def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]  # For AUC
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)
    cm = confusion_matrix(y_test, preds)
    
    print(f"\n{name} Evaluation:")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC-ROC": auc
    }

# Collect all metrics
metrics = []
metrics.append(evaluate_model("Random Forest", rf_model, X_test, y_test))
metrics.append(evaluate_model("XGBoost", xgb_model, X_test, y_test))
metrics.append(evaluate_model("LightGBM", lgbm_model, X_test, y_test))

# Save to CSV
metrics_df = pd.DataFrame(metrics)
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
csv_path = os.path.join(downloads_path, "model_performance_comparison.csv")
metrics_df.to_csv(csv_path, index=False)

# Auto-open the CSV (Windows only)
os.startfile(csv_path)

# ---------------------- Confusion Matrix Images for All Models ----------------------
print("\nSaving confusion matrix images...")

# Dictionary of models and their names
model_dict = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model
}

labels = ["Control", "Down Syndrome"]

# Generate and save confusion matrix heatmaps
for name, model in model_dict.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar=False, linewidths=1.5, linecolor='gray',
                square=True, annot_kws={"size": 16})
    
    plt.title(f"{name} Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save image
    filename = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()


# ---------------------- Accuracy Comparison Bar Chart ----------------------
print("\nGenerating accuracy comparison chart...")
model_names = ['Random Forest', 'XGBoost', 'LightGBM']
accuracies = [rf_acc, xgb_acc, lgbm_acc]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies, palette=['skyblue', 'lightgreen', 'orange'])

# Highlight best model
best_index = np.argmax(accuracies)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.3f}", ha='center', va='bottom', fontweight='bold')
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1.1)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

# ---------------------- Confusion Matrix Images for All Models ----------------------
print("\nSaving confusion matrix images...")

model_dict = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model
}

labels = ["Control", "Down Syndrome"]

for name, model in model_dict.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar=False, linewidths=1.5, linecolor='gray',
                square=True, annot_kws={"size": 16})
    
    plt.title(f"{name} Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save and open image
    filename = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}") 
    plt.close()
    
    if os.name == 'nt':  # Auto-open if Windows
        os.startfile(filename)


# ---------------------- Save the Model & Scaler ----------------------
print("\nSaving XGBoost model and scaler...")
joblib.dump(rf_model, "saved_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Files saved successfully in:", os.getcwd())

