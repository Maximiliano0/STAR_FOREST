"""
Model Training Pipeline
Separates model training logic from the GUI application.
Can be run standalone or imported by other modules.
"""

import os
import pickle
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_recall_fscore_support
)

from eda_pipeline import prepare_dataset_for_model
from config import STAR_TYPES


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(MODELS_DIR, "star_classifier.pkl")
LE_COLOR_PATH = os.path.join(MODELS_DIR, "le_color.pkl")
LE_SPECTRAL_PATH = os.path.join(MODELS_DIR, "le_spectral.pkl")
X_FEATURES_PATH = os.path.join(MODELS_DIR, "X_features.pkl")


def save_confusion_matrix_plot(y_test, y_pred) -> str:
    """
    Generate and save confusion matrix visualization.

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    str: Path to saved plot
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    conf_matrix = confusion_matrix(y_test, y_pred)
    star_names = [STAR_TYPES[i] for i in sorted(STAR_TYPES.keys())]

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=star_names, yticklabels=star_names, cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Star Type Classification', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(REPORTS_DIR, "model_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def save_metrics_plot(y_test, y_pred) -> str:
    """
    Generate and save classification metrics visualization.

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    str: Path to saved plot
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Calculate metrics for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

    star_names = [STAR_TYPES[i] for i in sorted(STAR_TYPES.keys())]
    x = np.arange(len(star_names))
    width = 0.25

    plt.figure(figsize=(14, 6))

    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    plt.xlabel('Star Type', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Metrics by Star Type', fontsize=14, fontweight='bold')
    plt.xticks(x, star_names, rotation=45, ha='right')
    plt.ylim([0, 1.1])
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(REPORTS_DIR, "model_metrics_by_class.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def save_accuracy_summary_plot(accuracy, conf_matrix) -> str:
    """
    Generate and save overall accuracy and per-class summary.

    Parameters:
    -----------
    accuracy : float
        Global accuracy score
    conf_matrix : array-like
        Confusion matrix

    Returns:
    --------
    str: Path to saved plot
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall accuracy gauge
    ax1.barh(['Accuracy'], [accuracy], color='#2ecc71', height=0.5)
    ax1.set_xlim([0, 1])
    ax1.set_title('Overall Model Accuracy', fontsize=12, fontweight='bold')
    ax1.text(accuracy/2, 0, f'{accuracy:.2%}', ha='center', va='center',
             fontsize=20, fontweight='bold', color='white')
    ax1.set_xlabel('Accuracy Score')
    ax1.grid(axis='x', alpha=0.3)

    # Per-class accuracy (recall)
    per_class_recall = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    star_names = [STAR_TYPES[i] for i in sorted(STAR_TYPES.keys())]
    colors = ['#2ecc71' if acc == 1.0 else '#f39c12' for acc in per_class_recall]

    ax2.barh(star_names, per_class_recall, color=colors)
    ax2.set_xlim([0, 1.1])
    ax2.set_title('Per-Class Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Recall Score')

    for i, v in enumerate(per_class_recall):
        ax2.text(v + 0.02, i, f'{v:.2%}', va='center', fontweight='bold')

    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(REPORTS_DIR, "model_accuracy_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def train_and_save_model(use_balanced: bool = False) -> Dict[str, str]:
    """
    Trains the DecisionTreeClassifier model and saves all artifacts.

    Uses the CLEANED dataset by default. The cleaned dataset has wording issues
    fixed and is ready for model training without additional preprocessing.

    Parameters:
    -----------
    use_balanced : bool
        If True, uses oversampled dataset. If False, uses clean dataset (default).
        Oversampling is applied during EDA pipeline only if needed for class balance.

    Returns:
    --------
    dict: Paths to saved artifacts (model, encoders, feature columns)
    """

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load cleaned data from EDA pipeline
    # Note: prepare_dataset_for_model(use_balanced=False) returns the CLEAN dataset
    # with wording issues fixed, ready for training
    df = prepare_dataset_for_model(use_balanced=use_balanced)

    # Feature engineering
    feature_cols = [
        'Temperature (K)',
        'Luminosity (L/Lo)',
        'Radius (R/Ro)',
        'Absolute magnitude (Mv)',
        'Star color',
        'Spectral Class'
    ]
    x = df[feature_cols].copy()  # pylint: disable=invalid-name
    y = df['Star type']  # pylint: disable=invalid-name

    # Encode categorical features
    le_color = LabelEncoder()
    le_spectral = LabelEncoder()
    x['Star color'] = le_color.fit_transform(x['Star color'].astype(str))
    x['Spectral Class'] = le_spectral.fit_transform(
        x['Spectral Class'].astype(str))

    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
        x, y, test_size=0.2, random_state=42
    )

    # Train DecisionTreeClassifier
    model = DecisionTreeClassifier(
        random_state=42,
        min_samples_split=10,
        ccp_alpha=0.01,
        max_depth=5
    )
    model.fit(x_train, y_train)

    # Evaluate model
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print metrics
    print("-" * 55)
    print("MODEL TRAINING COMPLETED")
    print("-" * 55)
    print(f"Accuracy Global: {acc:.2%}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("-" * 55)
    print(classification_report(y_test, y_pred))

    # Generate visualizations
    print("\nGenerating visualizations...")
    cm_plot = save_confusion_matrix_plot(y_test, y_pred)
    metrics_plot = save_metrics_plot(y_test, y_pred)
    accuracy_plot = save_accuracy_summary_plot(acc, conf_matrix)

    print(f"✅ Confusion matrix plot: {cm_plot}")
    print(f"✅ Metrics by class plot: {metrics_plot}")
    print(f"✅ Accuracy summary plot: {accuracy_plot}")

    # Save artifacts
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(LE_COLOR_PATH, 'wb') as f:
        pickle.dump(le_color, f)
    with open(LE_SPECTRAL_PATH, 'wb') as f:
        pickle.dump(le_spectral, f)
    with open(X_FEATURES_PATH, 'wb') as f:
        x_features = x.columns  # pylint: disable=invalid-name
        pickle.dump(x_features, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Color encoder saved to: {LE_COLOR_PATH}")
    print(f"Spectral encoder saved to: {LE_SPECTRAL_PATH}")
    print(f"Feature columns saved to: {X_FEATURES_PATH}")

    return {
        "model": MODEL_PATH,
        "le_color": LE_COLOR_PATH,
        "le_spectral": LE_SPECTRAL_PATH,
        "features": X_FEATURES_PATH
    }


def load_model() -> Tuple:
    """
    Loads pre-trained model and encoders from disk.
    If model doesn't exist, trains a new one.

    Returns:
    --------
    tuple: (model, X_features_columns, le_color, le_spectral)
    """

    paths_to_check = [MODEL_PATH, LE_COLOR_PATH, LE_SPECTRAL_PATH,
                      X_FEATURES_PATH]
    if all(os.path.exists(p) for p in paths_to_check):
        print("Loading pre-trained model...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(LE_COLOR_PATH, 'rb') as f:
            le_color = pickle.load(f)
        with open(LE_SPECTRAL_PATH, 'rb') as f:
            le_spectral = pickle.load(f)
        with open(X_FEATURES_PATH, 'rb') as f:
            x_features = pickle.load(f)  # pylint: disable=invalid-name
        print("Model loaded successfully!")
        return model, x_features, le_color, le_spectral

    print("Model not found. Training new model...")
    train_and_save_model()
    return load_model()


if __name__ == "__main__":
    train_and_save_model(use_balanced=False)
