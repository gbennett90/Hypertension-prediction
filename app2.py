import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Load the saved model and scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure 'best_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Set up the Streamlit page configuration
st.set_page_config(page_title="Hypertension Risk Predictor", layout="wide")

# --- UI Layout and Title ---
st.title('Hypertension Risk Prediction App')
st.write('This application uses a pre-trained machine learning model to predict the likelihood of hypertension based on various health metrics and displays key data visualizations.')

st.markdown("""
<style>
.st-emotion-cache-1f8u9f {
    background-color: #f0f8ff; /* Lighter background */
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.st-emotion-cache-13sdm68 {
    background-color: #f0f8ff; /* Consistent background */
    border: 1px solid #4CAF50; /* Health-focused green border */
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
}
.st-emotion-cache-1215r6k {
    font-size: 1.5rem;
    font-weight: bold;
    color: #004d40; /* Dark green for headings */
}
.stSuccess {
    background-color: #e8f5e9; /* Light green for success */
    color: #1b5e20; /* Darker green for text */
}
.stError {
    background-color: #ffebee; /* Light red for error */
    color: #b71c1c; /* Darker red for text */
}
</style>
""", unsafe_allow_html=True)

# Create tabs for navigation
tab1, tab2, tab3 = st.tabs(["Prediction", "Visualizations", "Summary & Recommendations"])

# Initialize session state for prediction result
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""

with tab1:
    # --- User Input Form ---
    st.header('Enter Patient Information')

    # Add a field for the patient's name
    patient_name = st.text_input('Patient Name', help="Enter the patient's name for a personalized result.")
    st.session_state.patient_name = patient_name

    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        # Family History (binary)
        family_history_input = st.selectbox(
            'Does the patient have a family history of hypertension?',
            ('No', 'Yes'),
            help="A family history of hypertension is a significant risk factor."
        )
        family_history = 1 if family_history_input == 'Yes' else 0

        # Age (numerical)
        age = st.number_input('Age', min_value=18, max_value=120, value=40, step=1,
                              help="The patient's age in years. Age is a major factor in hypertension risk.")

        # BMI (numerical)
        bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                              help="BMI is a measure of body fat based on height and weight that applies to adult men and women. A high BMI is a known risk factor.")

        # Stress Score (numerical)
        stress_score = st.number_input('Stress Score (1-10)', min_value=1.0, max_value=10.0, value=5.0, step=0.1,
                                       help="A self-reported score of the patient's stress level, on a scale of 1 to 10.")

    with col2:
        # BP_History (categorical, needs one-hot encoding logic)
        bp_history_input = st.selectbox(
            'Blood Pressure History',
            ('Normal', 'Prehypertension', 'High'),
            help="The patient's known history of blood pressure measurements."
        )
        bp_history_normal = 1 if bp_history_input == 'Normal' else 0
        bp_history_prehypertension = 1 if bp_history_input == 'Prehypertension' else 0

        # Sleep Duration (numerical)
        sleep_duration = st.number_input('Sleep Duration (hours)', min_value=1.0, max_value=12.0, value=7.0, step=0.1,
                                         help="The average number of hours the patient sleeps per night.")

        # Salt Intake (numerical)
        salt_intake = st.number_input('Salt Intake (grams/day)', min_value=0.0, max_value=20.0, value=5.0, step=0.1,
                                      help="The estimated daily salt intake in grams. High salt intake can contribute to high blood pressure.")

    # A button to trigger the prediction
    if st.button('Predict Hypertension Risk'):
        # Create a DataFrame from the user's input
        # NOTE: The column order MUST match the order used during model training
        input_data = pd.DataFrame([[
            bp_history_normal,
            bp_history_prehypertension,
            age,
            stress_score,
            sleep_duration,
            bmi,
            salt_intake,
            family_history
        ]], columns=[
            'BP_History_Normal',
            'BP_History_Prehypertension',
            'Age',
            'Stress_Score',
            'Sleep_Duration',
            'BMI',
            'Salt_Intake',
            'Family_History'
        ])

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Store prediction in session state
        st.session_state.prediction = prediction[0]
        st.session_state.prediction_proba = prediction_proba[0]

        # Display the results
        st.subheader('Prediction Result')
        if st.session_state.prediction == 1:
            st.error('Based on the provided information, the model predicts the patient **has hypertension**.')
        else:
            st.success('Based on the provided information, the model predicts the patient **does not have hypertension**.')

        # Show more details about the prediction
        with st.expander("Show Prediction Confidence"):
            hypertension_proba = st.session_state.prediction_proba[1] * 100
            non_hypertension_proba = st.session_state.prediction_proba[0] * 100
            st.write(f"Confidence of Hypertension: **{hypertension_proba:.2f}%**")
            st.write(f"Confidence of No Hypertension: **{non_hypertension_proba:.2f}%**")
            st.info("The model's confidence is based on the probabilities calculated by the classifier.")


with tab2:
    st.header('Model Visualizations')
    st.write('Explore key insights from the dataset and model evaluation.')

    # Load dataset from github
    url = "https://raw.githubusercontent.com/gbennett90/Hypertension-prediction/refs/heads/master/hypertension_dataset.csv"
    df = pd.read_csv(url)

    # Pre-process the data for plotting
    df['Medication'] = df['Medication'].fillna('No medication')
    df['Has_Hypertension'] = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})
    df['Family_History'] = df['Family_History'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=['BP_History', 'Medication', 'Exercise_Level', 'Smoking_Status'], drop_first=True)

    # Create two columns for side-by-side plots
    col1_viz, col2_viz = st.columns(2)

    with col1_viz:
        # Plot 1: Hypertension Distribution
        st.subheader('Distribution of Hypertension')
        st.write("This chart shows the number of patients in the dataset who have hypertension versus those who do not. It helps us understand the balance of our target variable.")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x='Has_Hypertension', ax=ax)
        ax.set_title('Distribution of Hypertension')
        ax.set_xlabel('Has Hypertension (1=Yes, 0=No)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col2_viz:
        # Plot 2: Age Distribution by Hypertension Status
        st.subheader('Age Distribution by Hypertension Status')
        st.write("This histogram shows the frequency of different age groups, separated by whether they have hypertension. We can see how the risk of hypertension changes with age.")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df, x='Age', hue='Has_Hypertension', kde=True, bins=30, ax=ax)
        ax.set_title('Age Distribution by Hypertension')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    st.markdown("---")


    # The following code is for creating the metrics plots
    st.subheader('Model Performance Metrics')
    st.write("These charts compare the performance of different machine learning models on the prediction task.")

    # Recreate the metrics DataFrame
    selected_features = [
        'BP_History_Normal', 'BP_History_Prehypertension', 'Age', 'Stress_Score',
        'Sleep_Duration', 'BMI', 'Salt_Intake', 'Family_History'
    ]
    df_selected = df[selected_features + ['Has_Hypertension']]

    X = df_selected.drop('Has_Hypertension', axis=1)
    y = df_selected['Has_Hypertension']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the data for models that need it
    scaler_temp = StandardScaler()
    X_train_scaled = scaler_temp.fit_transform(X_train)
    X_test_scaled = scaler_temp.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    cv_results = {}

    for name, m in models.items():
        use_scaled = name in ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test
        m.fit(Xtr, y_train)
        y_pred = m.predict(Xte)
        report = classification_report(y_test, y_pred, output_dict=True)
        model_names.append(name)
        accuracies.append(report['accuracy'])
        precisions.append(report['weighted avg']['precision'])
        recalls.append(report['weighted avg']['recall'])
        f1_scores.append(report['weighted avg']['f1-score'])
        cv_scores = cross_val_score(m, Xtr, y_train, cv=5, scoring='accuracy')
        cv_results[name] = cv_scores.mean()

    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })

    # Create another two columns for the model performance plots
    col3_viz, col4_viz = st.columns(2)

    with col3_viz:
        # Plot 3: Performance Metrics of Each Classification Model
        metrics_melted = metrics_df.melt('Model', var_name='Metric', value_name='Score')
        st.write("This bar chart compares the Accuracy, Precision, Recall, and F1-Score for each model. It helps in identifying the best-performing model for the task.")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Score', y='Model', hue='Metric', data=metrics_melted, palette='viridis', ax=ax)
        ax.set_title('Performance Metrics of Each Classification Model')
        ax.set_xlabel('Score')
        ax.set_ylabel('Model')
        ax.set_xlim(0.7, 1.0)
        ax.legend(title='Metric')
        st.pyplot(fig)

    with col4_viz:
        # Plot 4: Cross-Validation Results
        st.subheader('5-Fold Cross-Validation Accuracy')
        st.write("Cross-validation is a technique to assess a model's performance on new data. This chart shows the average accuracy of each model across 5 different splits of the training data.")
        cv_df = pd.DataFrame(list(cv_results.items()), columns=['Model', 'CV_Accuracy'])
        cv_df = cv_df.sort_values(by='CV_Accuracy', ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='CV_Accuracy', y='Model', data=cv_df, palette='viridis', ax=ax)
        ax.set_title('5-Fold Cross-Validation Accuracy for Each Model')
        ax.set_xlabel('Mean Cross-Validation Accuracy')
        ax.set_ylabel('Model')
        ax.set_xlim(0.7, 1.0)
        st.pyplot(fig)
    st.markdown("---")

    # The confusion matrix will remain on its own line
    # Plot 5: Confusion Matrix for the Tuned Model
    st.subheader('Confusion Matrix of Tuned Random Forest')
    st.write("The confusion matrix provides a detailed view of the model's predictions. The cells show the number of correct and incorrect predictions for each class (hypertension vs. no hypertension).")
    # Use the best model from the hyperparameter tuning section
    best_model = joblib.load('best_model.pkl')
    # Get the test set for plotting
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Hypertension', 'Hypertension'])
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    st.pyplot(fig)
    
    st.markdown("---")
    # K-Means Elbow Method Plot
    st.subheader('K-Means Elbow Method')
    st.write("The elbow method helps determine the optimal number of clusters for K-Means by plotting the sum of squared distances for a range of cluster numbers. The 'elbow' in the plot suggests the point of diminishing returns.")

    # The following block is needed to run K-Means on the data
    # K-Means (unsupervised)
    inertias = []
    # To keep the app snappy, we'll only test up to 6 clusters
    K = range(1, 7)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_train_scaled)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K, inertias, 'bo-')
    ax.set_title('K-Means Elbow Method')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_xticks(K)
    st.pyplot(fig)


with tab3:
    st.header('Prediction Summary')

    # Get the patient name from session state
    patient_name = st.session_state.patient_name
    
    # Check if a name was entered for a personalized message
    if patient_name:
        st.write(f"Hello, **{patient_name}**!")

    if st.session_state.prediction is None:
        st.info("Please go to the **Prediction** tab and run a prediction first to see a summary.")
    else:
        if st.session_state.prediction == 1:
            st.error("Based on the information provided, the model predicts a high likelihood of hypertension. This is an opportunity to make positive changes. Let's look at the recommendations. 📝")
        else:
            st.success("Based on the information provided, the model predicts a low likelihood of hypertension. Great news! Keep up the good work. 👍")

        with st.expander("General Health Tips"):
            st.markdown("""
            * **Maintain a healthy diet:** Reduce salt intake and increase fruits, vegetables, and whole grains.
            * **Regular physical activity:** Aim for at least 30 minutes of moderate exercise most days of the week.
            * **Manage stress:** Practice relaxation techniques like meditation or deep breathing.
            * **Limit alcohol and avoid smoking:** These habits are significant risk factors for high blood pressure.
            * **Monitor your sleep:** Aim for 7-9 hours of quality sleep per night.
            * **Consult a professional:** These recommendations are for informational purposes only. It is crucial to consult with a healthcare professional for personalized medical advice.
            """)
