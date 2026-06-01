import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import os

def main():
    mlflow.autolog()
    
    print("Memulai pelatihan model di GitHub Actions...")
    
    # Load data
    df = pd.read_csv('loanDataset_preprocessing.csv')
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model berhasil dilatih! Akurasi: {score:.4f}")

if __name__ == "__main__":
    main()
