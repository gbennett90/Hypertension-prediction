# Load all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning models and tools
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans

import joblib

import warnings
warnings.filterwarnings('ignore')

# Load dataset from github
url = "https://raw.githubusercontent.com/gbennett90/Hypertension-prediction/refs/heads/master/hypertension_dataset.csv"
df = pd.read_csv(url)

# Display the first 5 rows
df.head()

# Show column names and data types
df.info()

# Check for missing values
df.isnull().sum()

# See basic statistics
df.describe()

# Replace missing values in Medication with 'No medication'
df['Medication'] = df['Medication'].fillna('No medication')

# Check if missing values are gone
df['Medication'].isnull().sum()

df.isnull().sum()

# List of categorical columns
categorical_cols = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status', 'Has_Hypertension']

# Check unique values for each categorical column
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

# Hypertension Distribution (Target Variable Count Plot)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Has_Hypertension')
plt.title('Distribution of Hypertension')
plt.xlabel('Has Hypertension')
plt.ylabel('Count')
plt.show()

# Age Distribution by Hypertension Status
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Has_Hypertension', kde=True, bins=30)
plt.title('Age Distribution by Hypertension')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Boxplot: BMI by Hypertension Status
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Has_Hypertension', y='BMI')
plt.title('BMI by Hypertension Status')
plt.xlabel('Has Hypertension')
plt.ylabel('BMI')
plt.show()

# Barplot: Smoking Status vs. Hypertension Rate
plt.figure(figsize=(6,4))
sns.barplot(
    data=df,
    x='Smoking_Status',
    y='Has_Hypertension',    # This works because you mapped it to 0/1
    estimator='mean'
)
plt.title('Proportion with Hypertension by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Proportion with Hypertension')
plt.show()

# Countplot: Exercise Level by Hypertension Status
plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Exercise_Level', hue='Has_Hypertension')
plt.title('Exercise Level vs. Hypertension Status')
plt.xlabel('Exercise Level')
plt.ylabel('Count')
plt.show()

# Pairplot (Scatterplot Matrix)
sns.pairplot(df, hue="Has_Hypertension", diag_kind="kde")
plt.show()

# Correlation
# Select only numeric columns from the DataFrame
numeric_df = df.select_dtypes(include=['number'])

# Compute the correlation matrix
corr = numeric_df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix (Numeric Features Only)")
plt.show()

# Encode binary categorical columns (Yes/No) as 1/0
df['Has_Hypertension'] = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})
df['Family_History'] = df['Family_History'].map({'Yes': 1, 'No': 0})

# One-hot encode multi-class categorical columns
df = pd.get_dummies(df, columns=['BP_History', 'Medication', 'Exercise_Level', 'Smoking_Status'], drop_first=True)

# View the first few rows to confirm encoding
print(df.head())

# Correlation with target
corr = df.corr()
print(corr['Has_Hypertension'].sort_values(ascending=False))

# --- Feature Engineering ---
# Create an Age Group feature
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])
df = pd.get_dummies(df, columns=['Age_Group'], drop_first=True)

# Combine BMI and Age into a new feature
df['BMI_Age'] = df['BMI'] * df['Age']

# Remove Highly Correlated Features
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
df = df.drop(to_drop, axis=1)

# Model-based Feature Selection
from sklearn.ensemble import RandomForestClassifier

X_full = df.drop('Has_Hypertension', axis=1)
y = df['Has_Hypertension']

model = RandomForestClassifier(random_state=42)
model.fit(X_full, y)

importances = pd.Series(model.feature_importances_, index=X_full.columns)
top_features = importances.sort_values(ascending=False).head(8).index.tolist()
print("Top features:", top_features)

selected_features = [
    'BP_History_Normal',
    'BP_History_Prehypertension',
    'Age',
    'Stress_Score',
    'Sleep_Duration',
    'BMI',
    'Salt_Intake',
    'Family_History'
]

# Plus target variable
df_selected = df[selected_features + ['Has_Hypertension']]

# Split data into train and test sets
X = df_selected.drop('Has_Hypertension', axis=1)
y = df_selected['Has_Hypertension']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Create lists to store the performance metrics
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Create an empty dictionary to store the mean cross-validation scores
cv_results = {}

# Loop through the models to train, evaluate, and get cross-validation scores
for name, model in models.items():
    # Use scaled data for models that benefit from it
    use_scaled = name in ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test

    # Perform cross-validation
    cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy')
    cv_results[name] = cv_scores.mean()

    # Fit and predict
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store the metrics
    model_names.append(name)
    accuracies.append(report['accuracy'])
    precisions.append(report['weighted avg']['precision'])
    recalls.append(report['weighted avg']['recall'])
    f1_scores.append(report['weighted avg']['f1-score'])

# Create a DataFrame from the collected metrics
metrics_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores
})

print("Collected Performance Metrics:")
print(metrics_df)

print("Cross-Validation Results (Mean Accuracy):")
for name, score in cv_results.items():
    print(f"- {name}: {score:.4f}")

# Melt the DataFrame to prepare it for plotting
metrics_melted = metrics_df.melt('Model', var_name='Metric', value_name='Score')

# Plotting the metrics
plt.figure(figsize=(14, 8))
sns.barplot(x='Score', y='Model', hue='Metric', data=metrics_melted, palette='viridis')
plt.title('Performance Metrics of Each Classification Model')
plt.xlabel('Score')
plt.ylabel('Model')
plt.xlim(0.7, 1.0) # Set a reasonable x-axis limit for better visibility
plt.legend(title='Metric')
plt.show()

# Create a DataFrame from the cross-validation results
cv_df = pd.DataFrame(list(cv_results.items()), columns=['Model', 'CV_Accuracy'])

# Sort the DataFrame by accuracy for a better-looking plot
cv_df = cv_df.sort_values(by='CV_Accuracy', ascending=False)

# Plotting the cross-validation results
plt.figure(figsize=(12, 7))
sns.barplot(x='CV_Accuracy', y='Model', data=cv_df, palette='viridis')
plt.title('5-Fold Cross-Validation Accuracy for Each Model')
plt.xlabel('Mean Cross-Validation Accuracy')
plt.ylabel('Model')
plt.xlim(0.7, 1.0)
plt.show()

# K-Means (unsupervised)
print("\n=== K-Means Clustering ===")
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
clusters = kmeans.predict(X_test_scaled)
score = silhouette_score(X_test_scaled, clusters)
print("Silhouette Score (K-Means):", score)

# K- means elbow medthod plot
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(K, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('K-Means Elbow Method')
plt.xticks(K)
plt.show()

# Hyper Parameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predict on test set
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

# Print metrics
print("Accuracy (Tuned Random Forest):", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
print("ROC-AUC (Tuned Random Forest):", roc_auc_score(y_test, y_proba_best))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'Tuned RF (AUC = {roc_auc_score(y_test, y_proba_best):.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Tuned Random Forest')
plt.legend()
plt.show()

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix - Tuned Random Forest')
plt.show()

# DEPLOYMENT PREPARATION: Save the Final Model and Scaler
# The tuned Random Forest model is the best performer. 
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nFinal model (tuned Random Forest) and scaler have been saved as 'best_model.pkl' and 'scaler.pkl'.")
print("These files are now ready for deployment in your Streamlit application.")
