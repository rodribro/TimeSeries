import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# [DATASET CREATION FUNCTIONS]

def transform_bbch_data(df):

    # Get the dates from the column names, skipping the first column (CODE)
    dates = pd.to_datetime(df.columns[1:])
    
    # Reset the index to make CODE a column
    df = df.reset_index()
    
    # Melt the dataframe to convert from wide to long format
    melted_df = pd.melt(
        df,
        id_vars=['CODE'],
        value_vars=df.columns[1:],
        var_name='Date',
        value_name='BBCH'
    )
    
    # Convert the Date column to datetime
    melted_df['Date'] = pd.to_datetime(melted_df['Date'])
    
    # Sort the dataframe by CODE and Date
    melted_df = melted_df.sort_values(['CODE', 'Date'])
    
    # Reset the index
    melted_df = melted_df.reset_index(drop=True)
    
    return melted_df

def count_daily_measurements(df): 
    '''
    Count daily number of measurements
    
    Input:
    -df: Pandas dataframe
    
    Output:
    -date_dict: dictionary with keys representing a date and values the corresponding daily measurements
    '''
    
    date_dict = {}
    for date in df['Date'].unique():
        date_dict[date] = len(df[df['Date'] == date])

    return date_dict

def clean_timestamp(timestamp):
    """
    Removes BOM character, strips whitespace and andles potential parsing issues

    Input: timestamp (datetime object)

    Output: clean timestamp (datetime object)
    """
    try:
        # Remove BOM and strip
        clean_timestamp = str(timestamp).replace('\ufeff', '').strip()
        
        # Parse datetime with multiple fallback methods
        return pd.to_datetime(clean_timestamp, errors='coerce')
    
    except Exception as e:
        print(f"Parsing error: {e}")
        return None


def find_common_dates(dataframe_dictionary, check_num = 0):
    # Combine the dataframes in pairs
    combs = list(combinations(dataframe_dictionary, 2))

    # Dictionary to store the common dates
    common_dates = {}

    for combination in combs:

        # Check dictionary for the 2 dataframes to use
        first = dataframe_dictionary[combination[0]]
        second = dataframe_dictionary[combination[1]]

        # Find dates intersection between 2 dataframes
        dates = set(first['Date'].unique()) & set(second['Date'].unique())

        # Store the common dates in a dictionary with a proper key name
        key_name = f'{combination[0]} x {combination[1]}'
        common_dates[key_name] = dates

    if (check_num != 0):
        for key, value in common_dates.items():
            num_common_dates = len(value)
            print(f'{key} | No. Common Dates: {num_common_dates}\n')

    return common_dates


def calculate_cumulative_gdd(temperature_df):

    # Create copy of the original dataframe
    cumulative_gdd = temperature_df.copy()

    # Follow GDD formula
    max_temps = cumulative_gdd.groupby('Date')['temperature'].max()
    min_temps = cumulative_gdd.groupby('Date')['temperature'].min()
    gdd = (max_temps + min_temps)/2 - 4.5

    # Get GDD cumulative sum
    gdd_cumulative = gdd.cumsum()
    
    # Map the cumulative GDD back to the temperature_df
    cumulative_gdd['GDD Cumul.'] = cumulative_gdd['Date'].map(gdd_cumulative)

    # Drop unnecessary columns
    cumulative_gdd.drop(columns='temperature', inplace=True)
    
    return cumulative_gdd


def calculate_accumulated_nitrates(irrigation_df):

    # Set a copy of the original dataframe
    cumul_irr = irrigation_df.copy()

    # Set Sample column to capital letter
    cumul_irr["Sample"] = cumul_irr["Sample"].str.upper()

    # Set irrigation volume to liters (L)
    cumul_irr["Quantity (L)"] = cumul_irr["Quantity (mL)"] / 1000

    # Create temporaty column Concentration for each type of plant (a, b or c)
    cumul_irr.loc[cumul_irr["Sample"].str.contains('A'), 'Concentration'] = 6
    cumul_irr.loc[cumul_irr["Sample"].str.contains('B'), 'Concentration'] = 13
    cumul_irr.loc[cumul_irr["Sample"].str.contains('C'), 'Concentration'] = 17

    # Calculate nitrates quantity in milimoles
    cumul_irr["Nitrates (milimoles)"] = cumul_irr["Quantity (L)"] * cumul_irr["Concentration"]

    # Calculate the cumulatuve irrigation and cumulative sum of nitrates by plant sample in new columns
    cumul_irr['Cumul. Irrigation (mL)'] = cumul_irr.groupby('Sample')['Quantity (mL)'].cumsum() 
    cumul_irr['Cumul. Nitrates (milimoles)'] = cumul_irr.groupby('Sample')['Nitrates (milimoles)'].cumsum()

    # Drop unnecessary columns
    cumul_irr.drop(columns={'Quantity (mL)', 'Quantity (L)', 'Concentration','Nitrates (milimoles)'}, inplace = True)

    return cumul_irr


def calculate_cumulative_par(par_df):
    
     # Create copy of the original dataframe
    cumulative_par = par_df.copy()
    
    # First get daily sums
    daily_par = cumulative_par.groupby('Date')['μmoles'].sum().reset_index()
    
    # Sort by date to ensure correct cumulative sum
    daily_par = daily_par.sort_values('Date')
    
    # Calculate cumulative sum of daily PAR values
    daily_par['PAR Cumul.'] = daily_par['μmoles'].cumsum()
    
    # Create final dataframe with just Date and cumulative PAR
    result_df = daily_par[['Date', 'PAR Cumul.']]
    
    # Map the cumulative values back to the original dates
    cumulative_par = cumulative_par[['Date']].drop_duplicates()
    cumulative_par = cumulative_par.merge(result_df, on='Date', how='left')
    
    return cumulative_par.sort_values('Date')
    return cumulative_par


def daily_average_humidity(humidity_df):

    # Create copy of original humidity dataframe
    hum_df = humidity_df.copy()

    # Calculate daily average humidity in %
    avg_hum = hum_df.groupby('Date')['humidity'].mean()
    hum_df['Average Humidity %'] = hum_df['Date'].map(avg_hum)

    # Drop unnecessary columns and duplicates
    hum_df.drop(columns={'humidity'}, inplace=True)
    hum_df.drop_duplicates(inplace=True)

    return hum_df


def merge_variables(bbch_df, biometry_df, gdd_df, par_df, nitrates_df):
    """
    Aligns cumulative environmental variables with biometry dates, considering sample IDs
    
    Parameters:
    biometry_df: DataFrame with biometry measurements, dates and plant sample IDs
    gdd_df: DataFrame with cumulative GDD and dates
    par_df: DataFrame with cumulative PAR and dates 
    nitrates_df: DataFrame with cumulative nitrates, dates and plant sample IDs
    """
    # Sort all dataframes by date
    biometry_df = biometry_df.sort_values(['Date', 'Sample'])
    bbch_df = bbch_df.sort_values(['Date', 'Sample'])
    gdd_df = gdd_df.sort_values(['Date'])
    par_df = par_df.sort_values(['Date'])
    nitrates_df = nitrates_df.sort_values(['Date', 'Sample'])
    
    # Merge with BBCH
    merged_df = pd.merge(biometry_df, bbch_df, how='left')

    # Merge biometry with GDD
    merged_df = pd.merge_asof(
        merged_df,
        gdd_df,
        on='Date',
        direction='backward'
    )
    
    # Merge with PAR
    merged_df = pd.merge_asof(
        merged_df,
        par_df,
        on='Date',
        direction='backward'
    )
    
    # Merge with Nitrates
    final_df = pd.merge_asof(
        merged_df,
        nitrates_df,
        on='Date',
        by='Sample',
        direction='backward'
    )
    
    return final_df

# [EXPLORATORY DATA ANALISYS VISUALIZATIONS]

def explained_variance_visualizer(df):
    # Extract numeric columns only
    df_numeric = df.select_dtypes(include=[np.number])

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Apply PCA (keep all components for full variance analysis)
    pca = PCA()
    pca.fit(df_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Cumulative variance (optional)
    cumulative_variance = np.cumsum(explained_variance)

    # Create Scree Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color="b", label="Individual Explained Variance")
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="--", color="r", label="Cumulative Explained Variance")

    # Labels and formatting
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot: Explained Variance by Principal Components")
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.legend()
    plt.grid(alpha=0.3)

    # Show plot
    plt.show()

    return explained_variance

def pca(df, n_components):
    try:
        if (n_components not in [2, 3]):
            raise ValueError("n_components must be either 2 or 3.")
        
        # Extract numeric columns for PCA (excluding non-numeric ones)
        df_numeric = df.select_dtypes(include=[np.number])

        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)

        if (n_components == 2):

            # Apply PCA (2 components for visualization)
            pca = PCA(n_components=2)
            pca_2d = pca.fit_transform(df_scaled)
            loadings_2d = pca.components_

            # Create a new DataFrame with PCA results and Sample labels
            df_pca = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])
            df_pca["Family Sample"] = df["Family Sample"]  # Retain sample information

            # Plot PCA results with Sample labels
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Family Sample", palette="tab10", s=100, alpha=0.8)
            plt.title("PCA Scatter Plot (Colored by Sample)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend(title="Family Sample", bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
            plt.show()

            return pca_2d, loadings_2d
        
        elif (n_components == 3):
            
            # Apply PCA with 3 components
            pca = PCA(n_components=3)
            pca_3d = pca.fit_transform(df_scaled)
            loadings_3d = pca.components_

            # Create DataFrame for PCA results
            df_pca = pd.DataFrame(pca_3d, columns=["PC1", "PC2", "PC3"])

            # Add Family Sample column
            df_pca["Family Sample"] = df["Sample"].str[:-2]

            # Create 3D Scatter Plot
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")

            # Create color mapping based on Family Sample
            unique_samples = df_pca["Family Sample"].unique()
            colors = sns.color_palette("tab10", len(unique_samples))
            color_dict = dict(zip(unique_samples, colors))

            # Scatter plot
            for sample in unique_samples:
                subset = df_pca[df_pca["Family Sample"] == sample]
                ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"], 
                        label=sample, color=color_dict[sample], s=50, alpha=0.8)

            # Labels and title
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_zlabel("Principal Component 3")
            ax.set_title("3D PCA Scatter Plot (Colored by Family Sample)")
            ax.legend(title="Family Sample", bbox_to_anchor=(1.1, 1))

            # Show plot
            plt.show()

            return pca_3d, loadings_3d
    
    except ValueError as e:
        print(f"Error: {e}")

# [MODELS]
    
def custom_train_test_split(df, label):
    X = df.drop(columns=label)
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test



def predict_biometry(df, biometry_columns, model_type='linear'):
    """
    Trains and evaluates different regression models on specified biometry columns.

    Parameters:
    - df (DataFrame): The input dataset.
    - biometry_columns (list): List of target columns to predict.
    - model_type (str): The type of regression model ('linear', 'ridge', 'lasso', 'random_forest', 'xgboost').

    Returns:
    - DataFrame: Results including MSE, MAE, and R² for each target variable.
    """
    
    model_mapping = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.01),
        'random_forest': RandomForestRegressor(random_state=42),
        'xgboost': XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Invalid model type. Choose from {list(model_mapping.keys())}")
    
    model = model_mapping[model_type]
    
    results = []
    
    for target in biometry_columns:
        
        X_train, X_test, y_train, y_test = custom_train_test_split(df, label=target)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Target Variable': target,
            'MSE': mse,
            'MAE': mae,
            'R^2': r2
        })
    
    return model_type, results
