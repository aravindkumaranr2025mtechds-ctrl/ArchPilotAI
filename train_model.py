import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model():
    df = pd.read_csv("dataset.csv")

    feature_columns = [
        "users_count",
        "traffic_level",
        "team_size",
        "update_frequency",
        "integrations",
        "security_level",
        "scalability_need"
    ]

    target_columns = [
        "architecture",
        "deployment_method",
        "api_style"
    ]

    feature_encoders = {}
    target_encoders = {}

    categorical_features = [
        "traffic_level",
        "update_frequency",
        "security_level",
        "scalability_need"
    ]

    for col in categorical_features:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        feature_encoders[col] = encoder

    for col in target_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        target_encoders[col] = encoder

    X = df[feature_columns]

    model_bundle = {
        "models": {},
        "feature_encoders": feature_encoders,
        "target_encoders": target_encoders,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "model_algorithm": "RandomForestClassifier"
    }

    print("======================================")
    print("ARCHPILOT AI - RANDOM FOREST TRAINING")
    print("======================================")

    for target in target_columns:
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\nTarget: {target}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        model_bundle["models"][target] = model

    joblib.dump(model_bundle, "archpilot_model.pkl")

    print("\nModel saved as archpilot_model.pkl")


if __name__ == "__main__":
    train_model()