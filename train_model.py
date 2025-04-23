
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # Load and clean data
    df = pd.read_csv("water_potability.csv")
    df.fillna(df.median(), inplace=True)  # Using median for better handling of outliers

    # Feature engineering
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training with hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    model = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Best Parameters:", model.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    joblib.dump(model.best_estimator_, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model training complete!")

if __name__ == "__main__":
    train_model()
