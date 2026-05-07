"""
Star Classification GUI Application
Interactive interface for star type prediction using a pre-trained decision tree classifier.
"""

# Data manipulation
import os
import warnings
import tkinter as tk
from tkinter import messagebox

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from sklearn.tree import plot_tree

from eda_pipeline import normalize_star_color
from model_train import load_model


# Suppress zero division warnings
warnings.filterwarnings("ignore", category=UserWarning)

#############################################################
#         LOAD PRE-TRAINED MODEL                            #
#############################################################

# Load model and encoders
model, x_features, le_color, le_spectral = load_model()  # pylint: disable=invalid-name

#############################################################
#          GUI APPLICATION                                 #
#############################################################


class StarApp:
    """GUI application for star type prediction."""

    def __init__(self, root):  # pylint: disable=redefined-outer-name
        """
        Initialize the Star Classification GUI.

        Parameters:
        -----------
        root : tk.Tk
            Root window of the Tkinter application
        """
        # Window properties
        self.root = root
        self.root.title("Star Classification System")
        self.root.geometry("500x850")  # Height increased for image display
        self.root.configure(padx=20, pady=20, bg="#f0f0f0")

        # Title
        tk.Label(
            root, text="Star Property Predictor",
            font=("Segoe UI", 18, "bold"), bg="#f0f0f0"
        ).pack()

        # Input fields
        self.entries = {}
        fields = [
            ("Temperature (K)", "Temperature (K)"),
            ("Luminosity (L/Lo)", "Luminosity (L/Lo)"),
            ("Radius (R/Ro)", "Radius (R/Ro)"),
            ("Absolute Magnitude (Mv)", "Absolute magnitude (Mv)"),
            ("Star Color", "Star color"),
            ("Spectral Class", "Spectral Class")
        ]
        for label_text, var_name in fields:
            frame = tk.Frame(root, bg="#f0f0f0")
            frame.pack(fill="x", pady=4)
            tk.Label(
                frame, text=label_text, width=20, anchor="w", bg="#f0f0f0"
            ).pack(side="left")
            ent = tk.Entry(frame)
            ent.pack(side="right", expand=True, fill="x")
            self.entries[var_name] = ent

        # Button for predict function
        tk.Button(
            root, text="PREDICT STAR TYPE", command=self.predict,
            bg="#2ecc71", fg="white", font=("Segoe UI", 11, "bold"),
            height=2
        ).pack(fill="x", pady=15)

        # Button for view tree logic
        tk.Button(
            root, text="View Tree Logic", command=self.show_logic,
            bg="#34495e", fg="white"
        ).pack(fill="x")

        # Result display frame
        self.result_frame = tk.LabelFrame(
            root, text=" Prediction Result ", bg="#f0f0f0",
            font=("Segoe UI", 10, "bold")
        )
        self.result_frame.pack(fill="both", expand=True, pady=15)

        # Star type result label
        self.result_label = tk.Label(
            self.result_frame, text="Waiting for input...",
            font=("Segoe UI", 14, "bold"), bg="#f0f0f0", fg="#2c3e50"
        )
        self.result_label.pack(pady=5)

        # Image display label
        self.image_display = tk.Label(self.result_frame, bg="#f0f0f0")
        self.image_display.pack(pady=10)

    def predict(self):
        """Predict star type based on user input and display result."""
        try:
            # Create DataFrame with user input
            user_input = pd.DataFrame([[
                float(self.entries['Temperature (K)'].get()),
                float(self.entries['Luminosity (L/Lo)'].get()),
                float(self.entries['Radius (R/Ro)'].get()),
                float(self.entries['Absolute magnitude (Mv)'].get()),
                le_color.transform(
                    [normalize_star_color(self.entries['Star color'].get().strip())]
                )[0],
                le_spectral.transform(
                    [self.entries['Spectral Class'].get().strip().upper()]
                )[0]
            ]], columns=x_features)

            # Predict with model
            prediction = model.predict(user_input)[0]

            star_types = {
                0: "Brown Dwarf", 1: "Red Dwarf", 2: "White Dwarf",
                3: "Main Sequence", 4: "Supergiant", 5: "Hypergiant"
            }

            result_name = star_types.get(prediction, "Unknown")

            # Display classification in window
            self.result_label.config(text=result_name.upper(), fg="#e67e22")

            # Display image
            self.update_image(result_name)

        except Exception as e:  # pylint: disable=broad-exception-caught
            messagebox.showerror("Error",
                                 f"Verifica los datos ingresados.\nDetalle: {str(e)}")

    def update_image(self, star_name):
        """
        Load and display image for the predicted star type.

        Parameters:
        -----------
        star_name : str
            Name of the star type (e.g., "White Dwarf")
        """
        # Convert name to lowercase and replace spaces with underscores
        # Example: "White Dwarf" -> "white_dwarf.png"
        filename = f"{star_name.lower().replace(' ', '_')}.png"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_dir, "pictures", filename)

        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                # Resize image to fit window
                img = img.resize((250, 200), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img)  # pylint: disable=attribute-defined-outside-init
                self.image_display.config(image=self.photo)
            except Exception:  # pylint: disable=broad-exception-caught
                self.result_label.config(
                    text=f"{star_name}\n(Error loading image)"
                )
        else:
            self.image_display.config(image='', text=f"Image not found:\n{filename}")
            print(f"Looking in: {img_path}")

    def show_logic(self):
        """Display the decision tree model logic."""
        plot_tree(
            model,
            feature_names=x_features.tolist(),
            class_names=[
                "Brown Dwarf", "Red Dwarf", "White Dwarf",
                "Main Sequence", "Supergiant", "Hypergiant"
            ],
            filled=True, rounded=True
        )
        plt.show()

# Run the Application
if __name__ == "__main__":
    # Create root window
    root = tk.Tk()
    # Initialize application
    app = StarApp(root)
    # Start event loop
    root.mainloop()
