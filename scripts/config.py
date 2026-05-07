"""Project Configuration and Constants

Centralized configuration to reduce hardcoding and ensure consistency
across all modules.
"""

# Feature Engineering
FEATURE_COLUMNS = [
    'Temperature (K)',
    'Luminosity (L/Lo)',
    'Radius (R/Ro)',
    'Absolute magnitude (Mv)',
    'Star color',
    'Spectral Class'
]

# Star Type Mapping
STAR_TYPES = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# Model Parameters
MODEL_PARAMS = {
    "random_state": 42,
    "min_samples_split": 10,
    "ccp_alpha": 0.01,
    "max_depth": 5,
    "test_size": 0.2
}

# Data Processing
DATA_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Stars.csv"
RANDOM_STATE = 42

# GUI Settings
GUI_WINDOW_SIZE = "500x850"
GUI_BACKGROUND_COLOR = "#f0f0f0"
GUI_PREDICT_BUTTON_COLOR = "#2ecc71"
GUI_TREE_BUTTON_COLOR = "#34495e"
GUI_TEXT_COLOR = "#2c3e50"
GUI_HIGHLIGHT_COLOR = "#e67e22"

# Image Settings
IMAGE_WIDTH = 250
IMAGE_HEIGHT = 200
