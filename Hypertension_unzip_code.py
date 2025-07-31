import zipfile

with zipfile.ZipFile(r"C:\Users\benne\Downloads\hypertension-risk-prediction-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall(r"C:\Users\benne\Downloads\hypertension_data")

print("Unzipping complete!")
