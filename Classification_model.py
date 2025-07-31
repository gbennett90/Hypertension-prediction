# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
import time
import io
warnings.filterwarnings('ignore')

# Load the data CSV file 
hypertension_df = pd.read_csv(r"C:\Users\benne\OneDrive\CIS 9660- Data Mining - Summer 2025\Project 2\hypertension-risk-prediction-dataset.zip")
pd.set_option('display.max_columns', None) # Display all columns 
st.dataframe(hypertension_df.head())

# Explore the data
st.title("Hypertension Dataset Exploration")

# Dataset shape and columns
st.write(f"**Dataset Shape:** {hypertension_df.shape}")
st.write("**Columns:**", hypertension_df.columns.tolist())

# Dataset info (captured as text)
buffer = io.StringIO()
hypertension_df.info(buf=buffer)
info_str = buffer.getvalue()
st.subheader("Dataset Info")
st.text(info_str)

# First 5 rows
st.subheader("First 5 Rows")
st.dataframe(hypertension_df.head())

# Dataset description
st.subheader("Dataset Description")
st.dataframe(hypertension_df.describe())

# Check for missing values
st.subheader("Missing Values")
st.dataframe(hypertension_df.isnull().sum().to_frame(name="Missing Count"))

# Check for unique values
st.subheader("Medication Value Counts (Before Filling)")
st.dataframe(hypertension_df['Medication'].value_counts(dropna=True).to_frame(name="Count"))

# Fill missing values (799) with new category
hypertension_df['Medication'] = hypertension_df['Medication'].fillna('None')

# Verify change
st.subheader("Medication Value Counts (After Filling)")
st.dataframe(hypertension_df['Medication'].value_counts().to_frame(name="Count"))

# Target variable distribution
st.subheader("Target Variable Distribution")

# Display raw counts
st.write("Raw Counts of 'Has_Hypertension':")
st.dataframe(hypertension_df['Has_Hypertension'].value_counts().to_frame(name="Count"))

# Display percentage distribution
st.write("Percentage Distribution of 'Has_Hypertension':")
percentages = hypertension_df['Has_Hypertension'].value_counts(normalize=True) * 100
percentages_df = percentages.to_frame(name="Percentage (%)")
st.dataframe(percentages_df)

# Data Visualization
# Plotting style
plt.style.use('default')
fig = plt.figure(figsize=(12, 10))

# Distribution Visualization
plt.subplot(3, 4, 1)
hypertension_counts = hypertension_df['Has_Hypertension'].value_counts()
plt.pie(hypertension_counts.values, labels=hypertension_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Hypertension')
plt.tight_layout()
plt.show()

# Visualization 2: Correlation heatmap (need to be fixed data is messy)
plt.subplot(3, 4, 2)
# Create numeric dataset for correlation
hypertension_df_numeric = hypertension_df.copy()
le = LabelEncoder()
hypertension_df_numeric['Has_Hypertension_encoded'] = le.fit_transform(hypertension_df_numeric['Has_Hypertension'])
hypertension_df_numeric = hypertension_df_numeric.drop('Has_Hypertension', axis=1)
# Ensure all categorical variables are encoded
for col in hypertension_df_numeric.columns:
    if hypertension_df_numeric[col].dtype == 'object':
        hypertension_df_numeric[col] = le.fit_transform(hypertension_df_numeric[col])

correlation_matrix = hypertension_df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Visualization 3: Feature distributions 
features = ['Age', 'Salt_Intake', 'Stress_Score', 'BP_History']
plt.figure(figsize=(16, 10))

for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)  # 2x2 grid for 4 features
    for status in hypertension_df['Has_Hypertension'].unique():
        data = hypertension_df[hypertension_df['Has_Hypertension'] == status][feature]
        plt.hist(data, alpha=0.6, label=status, bins=20)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature} by Hypertension Status')
    plt.legend()

st.pyplot(plt)

# Visualization 4: More feature distributions
features2 = ['Sleep_Duration', 'BMI', 'Exercise_Level']
plt.figure(figsize=(16, 10))
for i, feature in enumerate(features2):
    plt.subplot(3, 4, i + 7)
    for status in hypertension_df['Has_Hypertension'].unique():
        data = hypertension_df[hypertension_df['Has_Hypertension'] == status][feature]
        if hypertension_df[feature].dtype == 'object': 
            value_counts = data.value_counts()
            plt.bar(value_counts.index, value_counts.values, alpha=0.6, label=status)
        else:
            plt.hist(data, bins=20, alpha=0.6, label=status)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature} by Hypertension Status')
    plt.legend()

st.pyplot(plt)

# Count plot for representing the number of people having certain Stress Score
plt.figure(figsize=(8, 6))
sns.countplot(x='Stress_Score', data=hypertension_df, palette='Set2')
plt.title('Stress_Score \n')
plt.xlabel('Stress_Score')
plt.ylabel('Count')
st.pyplot(plt)

# Data Processing
# Prepare features and target
X = hypertension_df.drop('Has_Hypertension', axis=1)
y = hypertension_df['Has_Hypertension']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

st.dataframe(f"Original classes: {label_encoder.classes_}")
st.dataframe(f"Encoded classes: {np.unique(y_encoded)}")
st.dataframe()

# Split the data (70:30 as required)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

st.dataframe(f"Training set size: {X_train.shape}")
st.dataframe(f"Test set size: {X_test.shape}")
