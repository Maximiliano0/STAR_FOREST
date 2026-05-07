"""EDA Pipeline: Data Preparation and Preprocessing

Handles:
- Local dataset download and caching
- Wording normalization for categorical features
- Domain-based validation of physical stellar parameters
- IQR-based outlier removal (per class to preserve inter-class variability)
- Class imbalance analysis and visualization
- Conditional oversampling (only when dataset is imbalanced after cleaning)
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample


DATA_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Stars.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
RAW_DATA_PATH = os.path.join(DATA_DIR, "Stars.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "Stars_clean.csv")
BALANCED_DATA_PATH = os.path.join(DATA_DIR, "Stars_balanced.csv")


# Numeric features used for outlier detection and domain validation.
NUMERIC_FEATURES: List[str] = [
    "Temperature (K)",
    "Luminosity (L/Lo)",
    "Radius (R/Ro)",
    "Absolute magnitude (Mv)",
]

# Physically-plausible ranges for stellar parameters.
# References:
#   - Stellar effective temperatures span ~1,700 K (brown dwarfs / late M)
#     up to ~50,000 K (hot O-type / Wolf-Rayet stars).
#   - Bolometric luminosities for cataloged stars span ~1e-4 Lsun
#     (faint brown dwarfs) up to ~1e6 Lsun (hypergiants).
#   - Stellar radii span ~0.08 Rsun (brown dwarfs) up to ~2,000 Rsun
#     (red hypergiants).
#   - Absolute visual magnitudes for stars roughly in [-12, +20].
#   - Star type labels in this dataset are integers 0..5.
DOMAIN_RANGES: Dict[str, Tuple[float, float]] = {
    "Temperature (K)": (1000.0, 60000.0),
    "Luminosity (L/Lo)": (1e-5, 1e7),
    "Radius (R/Ro)": (0.01, 2500.0),
    "Absolute magnitude (Mv)": (-15.0, 25.0),
}
VALID_STAR_TYPES = {0, 1, 2, 3, 4, 5}


# Canonical values inferred from the dataset and common wording variants.
COLOR_NORMALIZATION_MAP = {
    "blue": "Blue",
    "bluewhite": "Blue White",
    "blue white": "Blue White",
    "blue-white": "Blue White",
    "blue white ": "Blue White",
    "blue whitewhite": "Blue White",
    "whitish": "White",
    "white": "White",
    "whity": "White",
    "yellowish": "Yellowish",
    "yellowish white": "Yellowish White",
    "yellowwhite": "Yellowish White",
    "yellow-white": "Yellowish White",
    "orange": "Orange",
    "orange-red": "Orange-Red",
    "orangered": "Orange-Red",
    "pale yellow orange": "Pale yellow orange",
    "pale yellow orange ": "Pale yellow orange",
    "red": "Red",
}


def _normalize_key(text: str) -> str:
    """Normalize string key for comparison.

    Converts to lowercase, strips whitespace, replaces underscores with spaces.

    Parameters:
    -----------
    text : str
        Text to normalize

    Returns:
    --------
    str: Normalized text
    """
    compact = " ".join(str(text).strip().lower().replace("_", " ").split())
    return compact


def normalize_star_color(value: str) -> str:
    """Normalize star color values to standard names.

    Handles variations in spacing, case, and punctuation.

    Parameters:
    -----------
    value : str
        Color value to normalize

    Returns:
    --------
    str: Normalized color name
    """
    key = _normalize_key(value)

    if key in COLOR_NORMALIZATION_MAP:
        return COLOR_NORMALIZATION_MAP[key]

    key_no_hyphen = key.replace("-", " ")
    if key_no_hyphen in COLOR_NORMALIZATION_MAP:
        return COLOR_NORMALIZATION_MAP[key_no_hyphen]

    key_no_space = key_no_hyphen.replace(" ", "")
    if key_no_space in COLOR_NORMALIZATION_MAP:
        return COLOR_NORMALIZATION_MAP[key_no_space]

    # Fallback: preserve information with a clean title-style value.
    return " ".join(key_no_hyphen.split()).title()


def download_dataset(local_path: str = RAW_DATA_PATH, force: bool = False) -> str:
    """Download dataset from URL and cache locally.

    Parameters:
    -----------
    local_path : str
        Local file path to save dataset
    force : bool
        If True, re-download even if file exists

    Returns:
    --------
    str: Path to the downloaded dataset
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path) and not force:
            print(f"Using cached dataset: {local_path}")
            return local_path

        print(f"Downloading dataset from {DATA_URL}...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(local_path, index=False)
        print(f"Dataset saved to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise


def load_local_dataset(local_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load dataset from local path, downloading if necessary.

    Parameters:
    -----------
    local_path : str
        Path to load dataset from

    Returns:
    --------
    pd.DataFrame: Loaded dataset with cleaned column names
    """
    try:
        if not os.path.exists(local_path):
            download_dataset(local_path)

        df = pd.read_csv(local_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise


def clean_wording_issues(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect and fix wording inconsistencies in categorical features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    tuple: (cleaned_df, replacements_summary)
        - cleaned_df: DataFrame with normalized wording
        - replacements_summary: DataFrame showing what was replaced
    """
    cleaned = df.copy()

    original_colors = cleaned["Star color"].astype(str)
    cleaned_colors = original_colors.map(normalize_star_color)
    cleaned["Star color"] = cleaned_colors
    cleaned["Spectral Class"] = cleaned["Spectral Class"].astype(str).str.strip().str.upper()

    changes = pd.DataFrame({"original": original_colors, "cleaned": cleaned_colors})
    changes = changes[changes["original"] != changes["cleaned"]]

    grouped = changes.groupby(["original", "cleaned"]).size()
    replacements = grouped.to_frame(name="count").reset_index()
    replacements = replacements.sort_values("count", ascending=False)

    return cleaned, replacements


def remove_invalid_domain_values(
    df: pd.DataFrame,
    target_col: str = "Star type",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows with values outside physically-plausible stellar ranges.

    Drops rows that contain NaNs in numeric features, non-positive
    Temperature/Luminosity/Radius, out-of-range physical values, or invalid
    Star type labels.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target column with star type labels.

    Returns:
    --------
    tuple: (filtered_df, summary_df)
        - filtered_df: rows that satisfy all domain constraints.
        - summary_df: per-rule count of removed rows.
    """
    working = df.copy()
    n_before = len(working)
    summary_rows = []

    # Coerce numeric columns; non-parseable values become NaN and get dropped.
    for col in NUMERIC_FEATURES:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    nan_mask = working[NUMERIC_FEATURES].isna().any(axis=1)
    if nan_mask.any():
        summary_rows.append({"rule": "NaN in numeric features", "removed": int(nan_mask.sum())})
    working = working.loc[~nan_mask].copy()

    for col, (low, high) in DOMAIN_RANGES.items():
        if col not in working.columns:
            continue
        bad = (working[col] < low) | (working[col] > high)
        if bad.any():
            summary_rows.append(
                {"rule": f"{col} outside [{low}, {high}]", "removed": int(bad.sum())}
            )
        working = working.loc[~bad].copy()

    if target_col in working.columns:
        bad_target = ~working[target_col].isin(VALID_STAR_TYPES)
        if bad_target.any():
            summary_rows.append(
                {"rule": f"{target_col} not in {sorted(VALID_STAR_TYPES)}",
                 "removed": int(bad_target.sum())}
            )
        working = working.loc[~bad_target].copy()

    summary = pd.DataFrame(summary_rows, columns=["rule", "removed"])
    n_after = len(working)
    print(f"Domain validation: kept {n_after}/{n_before} rows "
          f"({n_before - n_after} removed).")

    return working.reset_index(drop=True), summary


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str] | None = None,
    target_col: str = "Star type",
    k: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove outliers using the interquartile-range (IQR) criterion per class.

    For each class in ``target_col`` and each numeric column, computes Q1, Q3
    and IQR, then drops rows outside ``[Q1 - k*IQR, Q3 + k*IQR]``. Filtering
    per class preserves the natural inter-class spread (e.g. brown dwarfs vs.
    hypergiants) while still discarding intra-class anomalies.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (should be domain-validated first).
    columns : list[str] | None
        Numeric columns to evaluate. Defaults to ``NUMERIC_FEATURES``.
    target_col : str
        Class column used to compute IQR per group.
    k : float
        IQR multiplier (1.5 = standard Tukey fence).

    Returns:
    --------
    tuple: (filtered_df, summary_df)
        - filtered_df: dataframe without per-class outliers.
        - summary_df: per-class/per-column count of removed rows.
    """
    cols = columns if columns is not None else NUMERIC_FEATURES
    cols = [c for c in cols if c in df.columns]

    keep_mask = pd.Series(True, index=df.index)
    summary_rows = []

    for class_id, group in df.groupby(target_col):
        for col in cols:
            q1 = group[col].quantile(0.25)
            q3 = group[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            low = q1 - k * iqr
            high = q3 + k * iqr
            outliers = (group[col] < low) | (group[col] > high)
            n_out = int(outliers.sum())
            if n_out > 0:
                summary_rows.append({
                    "Star type": int(class_id),
                    "feature": col,
                    "removed": n_out,
                })
                keep_mask.loc[group.index[outliers]] = False

    filtered = df.loc[keep_mask].copy().reset_index(drop=True)
    summary = pd.DataFrame(summary_rows, columns=["Star type", "feature", "removed"])
    print(f"IQR outlier removal: kept {len(filtered)}/{len(df)} rows "
          f"({len(df) - len(filtered)} removed).")
    return filtered, summary


def oversample_by_class(
    df: pd.DataFrame,
    target_col: str = "Star type",
    random_state: int = 42
) -> pd.DataFrame:
    """Balance dataset using random oversampling.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame: Oversampled balanced dataset
    """
    class_sizes = df[target_col].value_counts()
    target_size = class_sizes.max()

    chunks = []
    for class_id in sorted(df[target_col].unique()):
        class_df = df[df[target_col] == class_id]
        class_up = resample(
            class_df, replace=True, n_samples=target_size, random_state=random_state
        )
        chunks.append(class_up)

    balanced = pd.concat(chunks, axis=0).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    return balanced


def save_imbalance_plot(before_counts: pd.Series, after_counts: pd.Series) -> str:
    """Generate and save class imbalance before/after visualization.

    Parameters:
    -----------
    before_counts : pd.Series
        Original class distribution
    after_counts : pd.Series
        Class distribution after oversampling

    Returns:
    --------
    str: Path to saved plot
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "class_imbalance_before_after.png")

    classes = sorted(set(before_counts.index).union(set(after_counts.index)))
    before = [int(before_counts.get(c, 0) or 0) for c in classes]
    after = [int(after_counts.get(c, 0) or 0) for c in classes]

    x = list(range(len(classes)))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], before, width=width, label="Original")
    plt.bar([i + width / 2 for i in x], after, width=width, label="Oversampled")
    plt.xticks(x, [str(c) for c in classes])
    plt.xlabel("Star type")
    plt.ylabel("Samples")
    plt.title("Class Imbalance: Original vs Oversampled")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def save_boxplots(
    before: pd.DataFrame,
    after: pd.DataFrame,
    columns: List[str] | None = None,
    filename: str = "outliers_boxplots_before_after.png",
) -> str:
    """Save before/after boxplots for numeric features (outlier visualization).

    Parameters:
    -----------
    before : pd.DataFrame
        Dataset before outlier removal.
    after : pd.DataFrame
        Dataset after outlier removal.
    columns : list[str] | None
        Columns to plot. Defaults to ``NUMERIC_FEATURES``.
    filename : str
        Output file name (saved under ``REPORTS_DIR``).

    Returns:
    --------
    str: Path to saved plot.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, filename)
    cols = columns if columns is not None else NUMERIC_FEATURES
    cols = [c for c in cols if c in before.columns and c in after.columns]

    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    log_scaled = {"Luminosity (L/Lo)", "Radius (R/Ro)", "Temperature (K)"}
    for ax, col in zip(axes, cols):
        ax.boxplot(
            [before[col].dropna().values, after[col].dropna().values],
            tick_labels=["Before", "After"],
        )
        ax.set_title(col, fontsize=10)
        if col in log_scaled:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("IQR Outlier Removal: Before vs After (per feature)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_domain_removal_plot(domain_summary: pd.DataFrame) -> str:
    """Save a bar chart summarizing rows removed by each domain rule.

    Parameters:
    -----------
    domain_summary : pd.DataFrame
        Output of ``remove_invalid_domain_values`` with columns ``rule`` and
        ``removed``.

    Returns:
    --------
    str: Path to saved plot.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "domain_invalid_removed.png")

    plt.figure(figsize=(9, 4))
    if domain_summary.empty:
        plt.text(0.5, 0.5, "No domain-invalid values detected.",
                 ha="center", va="center", fontsize=12)
        plt.axis("off")
    else:
        plt.barh(domain_summary["rule"], domain_summary["removed"], color="#c0504d")
        plt.xlabel("Rows removed")
        plt.title("Domain validation: rows removed per rule")
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_feature_distributions(df: pd.DataFrame,
                               filename: str = "feature_distributions.png") -> str:
    """Save histograms of numeric features colored by Star type.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe.
    filename : str
        Output file name (saved under ``REPORTS_DIR``).

    Returns:
    --------
    str: Path to saved plot.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, filename)
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    log_scaled = {"Luminosity (L/Lo)", "Radius (R/Ro)", "Temperature (K)"}

    for ax, col in zip(axes, cols):
        for class_id in sorted(df["Star type"].unique()):
            values = df.loc[df["Star type"] == class_id, col].dropna()
            if values.empty:
                continue
            ax.hist(values, bins=20, alpha=0.5, label=f"Type {int(class_id)}")
        ax.set_title(col, fontsize=10)
        if col in log_scaled:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Feature distributions by Star type (cleaned dataset)",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_wording_issues_plot(replacements: pd.DataFrame) -> str:
    """Generate and save wording issues visualization.

    Parameters:
    -----------
    replacements : pd.DataFrame
        DataFrame of detected wording replacements

    Returns:
    --------
    str: Path to saved plot
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "wording_fixes_star_color.png")

    if replacements.empty:
        plt.figure(figsize=(7, 3))
        plt.text(0.5, 0.5, "No wording fixes detected.", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    top = replacements.copy()
    top["label"] = top["original"] + " -> " + top["cleaned"]

    plt.figure(figsize=(10, 5))
    plt.barh(top["label"], top["count"])
    plt.xlabel("Count")
    plt.ylabel("Replacement")
    plt.title("Detected wording issues in 'Star color'")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


# pylint: disable=too-many-statements
def run_eda_pipeline() -> Dict[str, str]:
    """Execute complete EDA pipeline.

    Performs conditional oversampling: only applies oversampling if the cleaned
    dataset has class imbalance (i.e., if max count > min count).

    Returns:
    --------
    dict: Paths to all generated artifacts
    """
    download_dataset(RAW_DATA_PATH)
    raw_df = load_local_dataset(RAW_DATA_PATH)

    # 1) Wording normalization for categorical features.
    wording_df, replacements = clean_wording_issues(raw_df)

    # 2) Domain validation: drop rows with physically invalid values.
    domain_df, domain_summary = remove_invalid_domain_values(wording_df)

    # 3) IQR-based outlier removal (per class to preserve inter-class spread).
    pre_outlier_df = domain_df.copy()
    clean_df, outlier_summary = remove_outliers_iqr(domain_df)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    # 4) Conditional oversampling: only if classes are still imbalanced.
    class_counts = clean_df["Star type"].value_counts()
    max_count = class_counts.max()
    min_count = class_counts.min()
    needs_oversampling = max_count > min_count

    if needs_oversampling:
        balanced_df = oversample_by_class(clean_df)
        balanced_df.to_csv(BALANCED_DATA_PATH, index=False)
        after_counts = balanced_df["Star type"].value_counts().sort_index()
        print("\nClass imbalance detected after cleaning. Applying oversampling...")
    else:
        balanced_df = clean_df.copy()
        balanced_df.to_csv(BALANCED_DATA_PATH, index=False)
        after_counts = class_counts.sort_index()
        print("\nDataset already balanced after cleaning. No oversampling needed.")

    before_counts = clean_df["Star type"].value_counts().sort_index()

    # 5) Visualizations for the report.
    imbalance_plot_path = save_imbalance_plot(before_counts, after_counts)
    wording_plot_path = save_wording_issues_plot(replacements)
    domain_plot_path = save_domain_removal_plot(domain_summary)
    boxplot_path = save_boxplots(pre_outlier_df, clean_df)
    distribution_plot_path = save_feature_distributions(clean_df)

    print("=" * 55)
    print("EDA PIPELINE COMPLETED")
    print("=" * 55)
    print(f"Raw dataset         : {RAW_DATA_PATH}")
    print(f"Clean dataset       : {CLEAN_DATA_PATH}")
    print(f"Balanced dataset    : {BALANCED_DATA_PATH}")
    print(f"Wording fixes plot  : {wording_plot_path}")
    print(f"Domain removal plot : {domain_plot_path}")
    print(f"Outlier boxplots    : {boxplot_path}")
    print(f"Distribution plot   : {distribution_plot_path}")
    print(f"Imbalance plot      : {imbalance_plot_path}")
    print("\nClass distribution (after cleaning):")
    print(before_counts.to_string())
    print("\nClass distribution (final):")
    print(after_counts.to_string())

    if domain_summary.empty:
        print("\nNo domain-invalid values detected.")
    else:
        print("\nDomain validation removals:")
        print(domain_summary.to_string(index=False))

    if outlier_summary.empty:
        print("\nNo IQR outliers detected.")
    else:
        print("\nIQR outliers removed (per class/feature):")
        print(outlier_summary.to_string(index=False))

    if replacements.empty:
        print("\nNo wording issues detected for 'Star color'.")
    else:
        print("\nDetected wording replacements:")
        print(replacements.to_string(index=False))

    return {
        "raw_csv": RAW_DATA_PATH,
        "clean_csv": CLEAN_DATA_PATH,
        "balanced_csv": BALANCED_DATA_PATH,
        "wording_plot": wording_plot_path,
        "domain_plot": domain_plot_path,
        "boxplot": boxplot_path,
        "distribution_plot": distribution_plot_path,
        "imbalance_plot": imbalance_plot_path,
    }


def prepare_dataset_for_model(use_balanced: bool = False) -> pd.DataFrame:
    """Load and prepare dataset for model training.

    Parameters:
    -----------
    use_balanced : bool
        If True, returns oversampled dataset; else returns clean dataset

    Returns:
    --------
    pd.DataFrame: Prepared dataset ready for training
    """
    download_dataset(RAW_DATA_PATH)
    raw_df = load_local_dataset(RAW_DATA_PATH)
    wording_df, _ = clean_wording_issues(raw_df)
    domain_df, _ = remove_invalid_domain_values(wording_df)
    clean_df, _ = remove_outliers_iqr(domain_df)

    if use_balanced:
        class_counts = clean_df["Star type"].value_counts()
        if class_counts.max() > class_counts.min():
            return oversample_by_class(clean_df)

    return clean_df


if __name__ == "__main__":
    run_eda_pipeline()
