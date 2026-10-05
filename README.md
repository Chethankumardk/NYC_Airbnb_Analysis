# NYC Airbnb Exploratory Data Analysis

Python-based exploratory data analysis of New York City Airbnb data, focusing on data cleaning, missing-value handling, pricing patterns, neighbourhood characteristics, room types, and listing availability.

## Project Overview

This project cleans and explores a New York City Airbnb dataset to improve data quality and investigate patterns within Airbnb listings.

The workflow combines data preprocessing, exploratory analysis, and visualization using Python.

## Objectives

- Identify and remove duplicate records.
- Investigate and handle missing values.
- Clean inconsistent categorical values.
- Convert pricing and service-fee fields into usable numeric formats.
- Identify potential outliers in price and service-fee data.
- Analyze listing distribution across neighbourhood groups.
- Explore room-type and pricing patterns.
- Examine cancellation-policy distributions.
- Investigate minimum-night requirements and listing availability.
- Communicate findings using data visualizations.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Exploratory Data Analysis (EDA)

## Data Cleaning

The analysis includes:

- standardized column names
- duplicate removal
- missing-value inspection and imputation
- correction of inconsistent neighbourhood-group names
- removal of selected unused columns
- conversion of price and service-fee values to numeric data
- correction of negative minimum-night values
- standard-deviation-based outlier detection for price and service fee

## Exploratory Analysis

The project investigates questions including:

- How are listings distributed across neighbourhood groups?
- Which neighbourhood groups have higher average prices?
- What are the most common neighbourhoods?
- How are room types distributed?
- How does average price vary by room type?
- What cancellation policies are most common?
- How are price and service fee related?
- What is the typical minimum-night requirement?
- How does minimum-night requirement relate to availability?

## Repository Structure

```text
NYC_Airbnb_Analysis/
├── README.md
├── data/
│   └── README.md
├── notebooks/
│   └── nyc_airbnb_eda.ipynb
└── src/
    └── nyc_airbnb_eda.py
```

## Dataset

The original project references the **Airbnb Open Data** dataset hosted on Kaggle.

The raw dataset is not redistributed in this repository. See [`data/README.md`](data/README.md) for dataset information and reproduction instructions.

## Running the Analysis

1. Download the dataset described in `data/README.md`.
2. Extract `Airbnb_Open_Data.csv`.
3. Place the CSV in the location expected by the analysis or update the dataset path.
4. Install the required Python libraries.
5. Open `notebooks/nyc_airbnb_eda.ipynb` in Jupyter Notebook and run the cells in order.

## Portfolio Scope

This repository demonstrates practical skills in:

- data cleaning and preprocessing
- missing-value handling
- exploratory data analysis
- categorical and numerical data transformation
- outlier detection
- grouping and aggregation with Pandas
- data visualization with Matplotlib and Seaborn
