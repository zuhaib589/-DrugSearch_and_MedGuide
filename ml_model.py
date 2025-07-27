import pandas as pd

def predict_common_diseases(filepath):
    try:
        df = pd.read_csv(filepath)
        disease_counts = df['disease'].value_counts().head(5)
        return list(disease_counts.index)
    except Exception as e:
        return [f"Error reading data: {e}"]
