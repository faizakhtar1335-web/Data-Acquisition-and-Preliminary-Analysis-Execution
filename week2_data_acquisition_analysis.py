"""
Week 2 Task: Data Acquisition and Preliminary Analysis Execution

Dataset:
  Palmer Penguins (public dataset)
Source:
  https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv

Purpose:
  Download a public CSV dataset, inspect and clean it, calculate descriptive
  statistics, and create preliminary visualizations.

Requirements:
  pip install pandas matplotlib seaborn requests

Run:
  python week2_data_acquisition_analysis.py

The script saves the downloaded/raw CSV, cleaned CSV, summary tables, and charts
inside a folder named "week2_data_output" beside this script.
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "main/inst/extdata/penguins.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "week2_data_output"
RAW_FILE = OUTPUT_DIR / "penguins_raw.csv"
CLEAN_FILE = OUTPUT_DIR / "penguins_clean.csv"


def acquire_data() -> pd.DataFrame:
    """Download the public CSV and load it into a pandas DataFrame."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        print("Downloading Palmer Penguins dataset...")
        urlretrieve(DATA_URL, RAW_FILE)
    else:
        print(f"Using existing downloaded file: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print an initial overview before cleaning."""
    print("\n--- First five rows ---")
    print(df.head())

    print("\n--- Column names and data types ---")
    print(df.dtypes)

    print("\n--- Dataset dimensions ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Missing values by column ---")
    print(df.isna().sum())

    print("\n--- Duplicate rows ---")
    print(df.duplicated().sum())

    print("\n--- Numeric descriptive statistics ---")
    print(df.describe(include="number").round(2))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a clean analysis copy.

    - Standardize column names.
    - Remove exact duplicate rows.
    - Trim whitespace from text fields.
    - Convert text placeholders such as empty strings to missing values.
    - Keep missing values in the raw file; use transparent, documented handling.
      For the main analysis, rows missing essential numeric measurements are
      excluded from measurement-based charts rather than inventing values.
    """
    cleaned = df.copy()

    # Standardize column names for easier Python access.
    cleaned.columns = (
        cleaned.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    # Normalize text fields and treat empty strings as missing.
    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        cleaned[column] = cleaned[column].replace("", pd.NA)

    # Remove exact duplicate records, if any.
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    removed = before - len(cleaned)

    print("\n--- Cleaning summary ---")
    print(f"Exact duplicate rows removed: {removed}")
    print("Missing values after standardization:")
    print(cleaned.isna().sum())

    # Save a cleaned copy without silently imputing measurements.
    cleaned.to_csv(CLEAN_FILE, index=False)
    print(f"Cleaned dataset saved to: {CLEAN_FILE}")
    return cleaned


def analyze_data(df: pd.DataFrame) -> None:
    """Create descriptive summaries and preliminary charts."""
    sns.set_theme()

    print("\n--- Species counts (excluding missing labels) ---")
    if "species" in df.columns:
        print(df["species"].value_counts(dropna=False))

    print("\n--- Island counts ---")
    if "island" in df.columns:
        print(df["island"].value_counts(dropna=False))

    numeric = df.select_dtypes(include="number")
    numeric.describe().T.round(2).to_csv(OUTPUT_DIR / "numeric_summary.csv")

    if "species" in df.columns:
        df["species"].value_counts().to_csv(OUTPUT_DIR / "species_counts.csv")

    # Measurement columns used by the public Palmer Penguins dataset.
    required = {"bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"}
    if required.issubset(df.columns):
        measurements = df.dropna(
            subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        )

        # Histogram: distribution of body mass.
        plt.figure(figsize=(8, 5))
        sns.histplot(data=measurements, x="body_mass_g", bins=20, kde=True)
        plt.title("Distribution of Penguin Body Mass")
        plt.xlabel("Body mass (g)")
        plt.ylabel("Number of penguins")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "body_mass_histogram.png", dpi=150)
        plt.close()

        # Scatter plot: bill length vs bill depth, colored by species.
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=measurements,
            x="bill_length_mm",
            y="bill_depth_mm",
            hue="species" if "species" in measurements.columns else None,
        )
        plt.title("Bill Length vs Bill Depth")
        plt.xlabel("Bill length (mm)")
        plt.ylabel("Bill depth (mm)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "bill_measurements_scatter.png", dpi=150)
        plt.close()

        # Box plot: body mass by species, useful for comparing groups.
        if "species" in measurements.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=measurements, x="species", y="body_mass_g")
            plt.title("Body Mass by Penguin Species")
            plt.xlabel("Species")
            plt.ylabel("Body mass (g)")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "body_mass_by_species.png", dpi=150)
            plt.close()

        print("\nPreliminary interpretation:")
        print("- The histogram shows the spread and shape of body-mass values.")
        print("- The scatter plot helps assess whether bill measurements vary by species.")
        print("- The box plot compares the median, spread, and possible extreme values of body mass.")
        print("Use the generated charts and summary tables to write evidence-based findings.")
    else:
        print("Expected measurement columns were not found; charts were skipped.")

    print(f"\nAll outputs are stored in: {OUTPUT_DIR}")


def main() -> None:
    """Run acquisition, inspection, cleaning, and preliminary analysis."""
    raw_df = acquire_data()
    inspect_data(raw_df)
    cleaned_df = clean_data(raw_df)
    analyze_data(cleaned_df)


if __name__ == "__main__":
    main()
