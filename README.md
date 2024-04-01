# POWER-DATA-PROJECT_1
This script investigates a dataset containing household power consumption measurements.

Data Loading

The code imports necessary libraries for data manipulation (Pandas), visualization (Matplotlib, Seaborn), numerical operations (NumPy), 3D plotting (mpl_toolkits.mplot3d), and distance calculations (scipy.spatial.distance).

Inline plotting is enabled using %matplotlib inline.

pd.read_csv is used to load the CSV file named "household_power_consumption.csv" into a Pandas DataFrame named df. The delimiter is specified as ; (semicolon).

Data Exploration

The script simply prints the first few rows of the DataFrame (df) to provide a glimpse of the data structure.

df.info() provides detailed information about the data, including data types, memory usage, and presence of null values.

Data Cleaning (Optional)

Feature Selection:

A new DataFrame (df_toclean) is created by dropping the "Date" and "Time" columns, assuming they are not relevant for the current analysis.
Cleaning Function:

The script defines a function named clean_dataset that takes a DataFrame as input.
Assertions are used to ensure the input is indeed a DataFrame.
dropna(inplace=True) removes rows containing missing values (NaN).
A more comprehensive cleaning step is implemented using boolean indexing:
~df.isin([np.nan, np.inf, -np.inf]).any(1) creates a boolean mask that identifies rows where any column contains NaN, infinity, or negative infinity.
The function then returns a new DataFrame containing only the rows that passed the mask (df[indices_to_keep]) and converts them to numeric data type (float64) using .astype(np.float64).
Note: The script defines a data cleaning function but doesn't explicitly call it to clean df. You can uncomment the line df = clean_dataset(df.copy()) to perform cleaning before further analysis.

This script provides a starting point for analyzing household power consumption data. By exploring the data structure, identifying missing values, and potentially applying cleaning steps, you can prepare the data for further analysis tasks like visualization, feature engineering, or model building.
