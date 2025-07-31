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

# --- Streamlit Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Hypertension Risk Prediction",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced aesthetics ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333333; /* Darker text for better contrast on light content blocks */
    }
    .stApp {
        /* Removed custom background-color to use Streamlit's default dark background */
    }
    .st-emotion-cache-cnbvjp { /* Target the main block container */
        background-color: #ffffff; /* White background for content blocks */
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50; /* Darker header color */
    }
    .stButton>button {
        background-color: #3498db; /* Blue button */
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9; /* Darker blue on hover */
    }
    .stSuccess {
        background-color: #e6ffe6;
        color: #28a745;
        border-radius: 5px;
        padding: 10px;
    }
    .stWarning {
        background-color: #fff3e6;
        color: #ffc107;
        border-radius: 5px;
        padding: 10px;
    }
    .stInfo {
        background-color: #e6f7ff;
        color: #17a2b8;
        border-radius: 5px;
        padding: 10px;
    }

    /* --- Styles for st.dataframe tables --- */
    .st-emotion-cache-1uj2z2h table { /* This targets the table element within st.dataframe */
        border-collapse: collapse;
        width: 100%;
        border-radius: 8px; /* Apply rounded corners to the table */
        overflow: hidden; /* Ensures content respects border-radius */
    }

    .st-emotion-cache-1uj2z2h th { /* Table headers */
        background-color: #3498db; /* Blue header background */
        color: white;
        padding: 12px 15px;
        text-align: left;
        border-bottom: 2px solid #2980b9; /* Darker blue border */
    }

    .st-emotion-cache-1uj2z2h td { /* Table data cells */
        background-color: #f8f8f8; /* Light grey background for cells */
        color: #333333; /* Dark text for readability */
        padding: 10px 15px;
        border-bottom: 1px solid #ddd; /* Light grey border between rows */
    }

    .st-emotion-cache-1uj2z2h tr:nth-child(even) td { /* Alternate row background for readability */
        background-color: #eef2f7; /* Slightly darker grey for even rows */
    }

    .st-emotion-cache-1uj2z2h tr:hover td { /* Hover effect for rows */
        background-color: #dbe4ed; /* Lighter blue on hover */
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hypertension Risk Prediction App")

# --- Data Loading Section ---
st.header("1. Dataset Loading")

# URL for the raw dataset on GitHub
raw_data_url = "https://raw.githubusercontent.com/gbennett90/Hypertension-prediction/refs/heads/master/hypertension_dataset.csv"

try:
    hypertension_df = pd.read_csv(raw_data_url)
    # Set display options for pandas DataFrame
    pd.set_option('display.max_columns', None)
    st.success("Dataset loaded successfully from URL!")
    st.dataframe(hypertension_df.head())
except Exception as e:
    st.error(f"Error loading dataset from URL: {e}. Please ensure the URL is correct and the file is accessible.")
    st.stop()

# --- Data Exploration ---
st.header("2. Dataset Exploration")

# Dataset shape and columns
st.subheader("Dataset Overview")
st.write(f"**Dataset Shape:** {hypertension_df.shape}")
st.write("**Columns:**", hypertension_df.columns.tolist())

# Dataset info 
st.subheader("Dataset Info")
buffer = io.StringIO()
hypertension_df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

# First 5 rows
st.subheader("Preview Data")
st.dataframe(hypertension_df.head())

# Dataset description
st.subheader("Dataset Description (Numerical Features)")
st.dataframe(hypertension_df.describe())

# Check for missing values
st.subheader("Missing Values Check")
missing_values = hypertension_df.isnull().sum().to_frame(name="Missing Count")
st.dataframe(missing_values)

# Check for unique values in 'Medication' before filling
st.subheader("Medication Value Counts (Before Filling Missing Values)")
st.dataframe(hypertension_df['Medication'].value_counts(dropna=True).to_frame(name="Count"))

# Fill missing values in 'Medication'
hypertension_df['Medication'] = hypertension_df['Medication'].fillna('None')
st.success("Missing values in 'Medication' column filled with 'None'.")

# Verify change
st.subheader("Medication Value Counts (After Filling Missing Values)")
st.dataframe(hypertension_df['Medication'].value_counts().to_frame(name="Count"))

# Target variable distribution
st.subheader("Target Variable Distribution ('Has_Hypertension')")

col1, col2 = st.columns(2)
with col1:
    # Display raw counts
    st.write("Raw Counts:")
    st.dataframe(hypertension_df['Has_Hypertension'].value_counts().to_frame(name="Count"))
with col2:
    # Display percentage distribution
    st.write("Percentage Distribution:")
    percentages = hypertension_df['Has_Hypertension'].value_counts(normalize=True) * 100
    percentages_df = percentages.to_frame(name="Percentage (%)")
    st.dataframe(percentages_df)

# --- Data Visualization ---
st.header("3. Data Visualization")

# Plotting style
plt.style.use('default')

# Visualization 1: Distribution of Hypertension
st.subheader("Distribution of Hypertension Status")
fig1, ax1 = plt.subplots(figsize=(8, 8))
hypertension_counts = hypertension_df['Has_Hypertension'].value_counts()
ax1.pie(hypertension_counts.values, labels=hypertension_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999']) # Softer colors
ax1.set_title('Distribution of Hypertension Status')
st.pyplot(fig1)
plt.close(fig1) # Close the figure to free up memory

# Visualization 2: Correlation heatmap
st.subheader("Feature Correlation Matrix")
# Create numeric dataset for correlation
hypertension_df_numeric = hypertension_df.copy()
le = LabelEncoder()

# Encode 'Has_Hypertension' for correlation matrix
hypertension_df_numeric['Has_Hypertension_encoded'] = le.fit_transform(hypertension_df_numeric['Has_Hypertension'])
hypertension_df_numeric = hypertension_df_numeric.drop('Has_Hypertension', axis=1)

# Ensure all categorical variables are encoded for the correlation matrix
for col in hypertension_df_numeric.columns:
    if hypertension_df_numeric[col].dtype == 'object':
        hypertension_df_numeric[col] = le.fit_transform(hypertension_df_numeric[col])

fig2, ax2 = plt.subplots(figsize=(12, 10))
correlation_matrix = hypertension_df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax2)
ax2.set_title('Feature Correlation Matrix')
st.pyplot(fig2)
plt.close(fig2) # Close the figure

# Visualization 3: Feature distributions (Age, Salt_Intake, Stress_Score, BP_History)
st.subheader("Distribution of Key Features by Hypertension Status")
features = ['Age', 'Salt_Intake', 'Stress_Score', 'BP_History']
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12)) # 2x2 grid for 4 features
axes3 = axes3.flatten() # Flatten for easy iteration

for i, feature in enumerate(features):
    for status in hypertension_df['Has_Hypertension'].unique():
        data = hypertension_df[hypertension_df['Has_Hypertension'] == status][feature]
        axes3[i].hist(data, alpha=0.7, label=f'Has_Hypertension: {status}', bins=20, edgecolor='black',
                     color= '#4CAF50' if status == 'No' else '#FF6347') # Green for No, Red for Yes
    axes3[i].set_xlabel(feature)
    axes3[i].set_ylabel('Frequency')
    axes3[i].set_title(f'Distribution of {feature} by Hypertension Status')
    axes3[i].legend()
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3) # Close the figure

# Visualization 4: More feature distributions (Sleep_Duration, BMI, Exercise_Level)
st.subheader("Distribution of Additional Features by Hypertension Status")
features2 = ['Sleep_Duration', 'BMI', 'Exercise_Level']
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6)) # 1x3 grid for 3 features
axes4 = axes4.flatten()

for i, feature in enumerate(features2):
    for status in hypertension_df['Has_Hypertension'].unique():
        data = hypertension_df[hypertension_df['Has_Hypertension'] == status][feature]
        if hypertension_df[feature].dtype == 'object':
            value_counts = data.value_counts()
            axes4[i].bar(value_counts.index, value_counts.values, alpha=0.7, label=f'Has_Hypertension: {status}', edgecolor='black',
                        color= '#4CAF50' if status == 'No' else '#FF6347')
        else:
            axes4[i].hist(data, bins=20, alpha=0.7, label=f'Has_Hypertension: {status}', edgecolor='black',
                        color= '#4CAF50' if status == 'No' else '#FF6347')
    axes4[i].set_xlabel(feature)
    axes4[i].set_ylabel('Frequency')
    axes4[i].set_title(f'Distribution of {feature} by Hypertension Status')
    axes4[i].legend()
plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4) # Close the figure

# Count plot for Stress Score
st.subheader("Count of Individuals by Stress Score")
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.countplot(x='Stress_Score', data=hypertension_df, palette='viridis', ax=ax5)
ax5.set_title('Count of Individuals by Stress Score')
ax5.set_xlabel('Stress Score')
ax5.set_ylabel('Count')
st.pyplot(fig5)
plt.close(fig5) # Close the figure

# --- Data Preprocessing ---
st.header("4. Data Preprocessing")

# Prepare features and target
X = hypertension_df.drop('Has_Hypertension', axis=1)
y = hypertension_df['Has_Hypertension']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

st.write(f"**Original 'Has_Hypertension' classes:** {label_encoder.classes_}")
st.write(f"**Encoded 'Has_Hypertension' classes (0 and 1):** {np.unique(y_encoded)}")

# Display a sample of encoded target variable
st.write("Sample of Encoded Target Variable (y_encoded):")
st.dataframe(pd.DataFrame(y_encoded, columns=['Encoded_Has_Hypertension']).head())

# Identify categorical columns for one-hot encoding
categorical_cols = X.select_dtypes(include='object').columns
st.write(f"**Categorical columns identified for One-Hot Encoding:** {categorical_cols.tolist()}")

# Apply One-Hot Encoding to categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
st.write("Features after One-Hot Encoding:")
st.dataframe(X_encoded.head())
st.write(f"Shape after One-Hot Encoding: {X_encoded.shape}")

# Split the data (70:30 as required)
st.subheader("Splitting Data into Training and Testing Sets (70:30 Ratio)")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

st.write(f"**Training set size (X_train):** {X_train.shape}")
st.write(f"**Test set size (X_test):** {X_test.shape}")
st.write(f"**Training target set size (y_train):** {y_train.shape}")
st.write(f"**Test target set size (y_test):** {y_test.shape}")

# Scale numerical features
st.subheader("Scaling Numerical Features (StandardScaler)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Numerical features scaled successfully.")
st.write("Sample of scaled training data (first 5 rows):")
st.dataframe(pd.DataFrame(X_train_scaled, columns=X_encoded.columns).head())

# --- Model Training and Evaluation ---
st.header("5. Model Training and Evaluation")

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42, probability=True), # probability=True for cross_val_score
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    st.subheader(f"Training and Evaluating: {name}")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    end_time = time.time()
    training_time = end_time - start_time

    results[name] = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "training_time": training_time
    }

    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Training Time:** {training_time:.4f} seconds")

    st.write("**Classification Report:**")
    st.json(report) # Display classification report as JSON for better readability

    st.write("**Confusion Matrix:**")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title(f'Confusion Matrix for {name}')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

# --- Model Comparison ---
st.header("6. Model Comparison")
st.write("Comparing the performance of different models:")

comparison_data = {
    "Model": [],
    "Accuracy": [],
    "Training Time (s)": []
}

for name, res in results.items():
    comparison_data["Model"].append(name)
    comparison_data["Accuracy"].append(res["accuracy"])
    comparison_data["Training Time (s)"].append(res["training_time"])

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df.sort_values(by="Accuracy", ascending=False))

# Bar chart for Accuracy Comparison
st.subheader("Accuracy Comparison Across Models")
fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=comparison_df.sort_values(by="Accuracy", ascending=False), palette="viridis", ax=ax_acc)
ax_acc.set_ylim(0, 1)
ax_acc.set_ylabel("Accuracy Score")
ax_acc.set_title("Model Accuracy Comparison")
ax_acc.tick_params(axis='x', rotation=45)
st.pyplot(fig_acc)
plt.close(fig_acc)

# Bar chart for Training Time Comparison
st.subheader("Training Time Comparison Across Models")
fig_time, ax_time = plt.subplots(figsize=(10, 6))
sns.barplot(x="Model", y="Training Time (s)", data=comparison_df.sort_values(by="Training Time (s)"), palette="magma", ax=ax_time)
ax_time.set_ylabel("Training Time (seconds)")
ax_time.set_title("Model Training Time Comparison")
ax_time.tick_params(axis='x', rotation=45)
st.pyplot(fig_time)
plt.close(fig_time)

# --- Cross-Validation ---
st.header("7. Cross-Validation Performance")
st.write("Performing 5-Fold Stratified Cross-Validation for each model.")

cv_results = {}
for name, model in models.items():
    st.subheader(f"Cross-Validation for: {name}")
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=strat_k_fold, scoring='accuracy')
    cv_results[name] = cv_scores
    st.write(f"**Mean CV Accuracy:** {cv_scores.mean():.4f}")
    st.write(f"**CV Scores:** {cv_scores}")

# Box plot for Cross-Validation Scores
st.subheader("Cross-Validation Accuracy Distribution")
cv_df = pd.DataFrame(cv_results)
fig_cv, ax_cv = plt.subplots(figsize=(12, 7))
sns.boxplot(data=cv_df, ax=ax_cv, palette="pastel")
ax_cv.set_title("Cross-Validation Accuracy Scores for Different Models")
ax_cv.set_ylabel("Accuracy Score")
ax_cv.set_xlabel("Model")
ax_cv.tick_params(axis='x', rotation=45)
st.pyplot(fig_cv)
plt.close(fig_cv)

# --- Prediction Interface (Simple Example) ---
st.header("8. Make a Prediction (Example with Random Forest)")
st.write("Enter values to predict hypertension risk.")

# Select a trained model for prediction
selected_model_name = st.selectbox("Select a model for prediction:", list(models.keys()))
selected_model = models[selected_model_name]

# Collect user input for features
st.write("Please enter the following details:")
input_age = st.slider("Age", 18, 90, 40)
input_salt_intake = st.slider("Salt Intake (mg/day)", 1000, 5000, 2500)
input_stress_score = st.slider("Stress Score (1-10)", 1, 10, 5)
input_bp_history = st.selectbox("BP History", hypertension_df['BP_History'].unique())
input_sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.5)
input_bmi = st.slider("BMI", 15.0, 40.0, 25.0, 0.1)
input_exercise_level = st.selectbox("Exercise Level", hypertension_df['Exercise_Level'].unique())
input_medication = st.selectbox("Medication", hypertension_df['Medication'].unique())
# Removed 'Gender' as it's not in your dataset
input_family_history = st.selectbox("Family History", hypertension_df['Family_History'].unique()) 
input_smoking_status = st.selectbox("Smoking Status", hypertension_df['Smoking_Status'].unique()) 

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Age': [input_age],
    'Salt_Intake': [input_salt_intake],
    'Stress_Score': [input_stress_score],
    'BP_History': [input_bp_history],
    'Sleep_Duration': [input_sleep_duration],
    'BMI': [input_bmi],
    'Exercise_Level': [input_exercise_level],
    'Medication': [input_medication],
    'Family_History': [input_family_history], 
    'Smoking_Status': [input_smoking_status], 
   
})


# Apply one-hot encoding to user input, ensuring all columns are present
input_encoded = pd.get_dummies(input_data, columns=input_data.select_dtypes(include='object').columns, drop_first=True)

# Align columns - crucial for consistent prediction
# This ensures that the columns in input_aligned match the columns in X_encoded (training data)
# Any columns in X_encoded not in input_encoded will be added with 0, and vice-versa.
input_aligned = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Scale the input data
input_scaled = scaler.transform(input_aligned)

if st.button("Predict Hypertension Risk"):
    with st.spinner("Making prediction..."):
        prediction_proba = selected_model.predict_proba(input_scaled)[0]
        prediction = selected_model.predict(input_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]

        st.subheader("Prediction Result:")
        st.write(f"The model predicts: **{predicted_class}**")
        st.write(f"Probability of 'No Hypertension': **{prediction_proba[0]:.2f}**")
        st.write(f"Probability of 'Has Hypertension': **{prediction_proba[1]:.2f}**")

        if predicted_class == 'Yes':
            st.warning("Based on the input, there is a predicted risk of hypertension. Please consult a healthcare professional.")
        else:
            st.success("Based on the input, there is no predicted risk of hypertension. Keep up the good work!")
