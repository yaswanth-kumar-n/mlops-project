from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from .utils.logger import get_logger
from .utils.config_loader import load_config

logger = get_logger(__name__)

def train():
    config = load_config()
    processed_path = config["data"]["processed_path"]
    model_path = config["model"]["path"]
    target_col = config["model"]["target_column"]

    df = pd.read_csv(processed_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    logger.info(f"Saved model to {model_path}")

if __name__ == "__main__":
    train()