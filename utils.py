import pandas as pd
from itertools import combinations

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