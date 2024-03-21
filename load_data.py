import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

# Inner vortex
# dataset_location = "/Users/max/OneDrive - University of Otago/MLS datasets/InnerVortexOnly75-82S"
# Office pc 
dataset_location = "/Users/maximus/OneDrive - University of Otago/MLS datasets/InnerVortexOnly75-82S"

# 75-82 Lat
# Laptop
# dataset_location = "/Users/max/OneDrive - University of Otago/MLS datasets/Latitude75-82S"
# Office pc 
dataset_location = "/Users/maximus/OneDrive - University of Otago/MLS datasets/Latitude75-82S"


"""
Method takes dataset location
Grabs all csv files for o3, co and temp
concatenates them into one big df 
created a lag of 5 days
Everything is a feature except o3 
Splits into training and test data - training 2005 - 2022 and testing on year 2023
"""
def load_data(dataset_location):
    o3_by_day = []
    co_by_day = []
    temp_by_day = [] 
    for filename in os.listdir(dataset_location):
        file_path = os.path.join(dataset_location, filename)
        
        df = pd.read_csv(file_path)
        
        if file_path.endswith("O3.csv") or file_path.endswith("O3_v.csv"):
            year = filename.split('_')[0]
            o3_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            o3_df['Year'] = year
            o3_by_day.append(o3_df)
    
        if file_path.endswith("CO.csv") or file_path.endswith("CO_v.csv"):
            year = filename.split('_')[0]
            co_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            co_df['Year'] = year
            co_by_day.append(co_df)

        if file_path.endswith("Temperature.csv") or file_path.endswith("Temperature_v.csv"):
            year = filename.split('_')[0]
            temp_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            temp_df['Year'] = year
            temp_by_day.append(temp_df)

    o3_data = pd.concat(o3_by_day)
    co_data = pd.concat(co_by_day)
    temp_data = pd.concat(temp_by_day)

    merged_data = pd.merge(o3_data, co_data, on=['Day', 'Year'], suffixes=('_o3', '_co'))
    merged_data = pd.merge(merged_data, temp_data[['Day', 'Year', 'mean_ppmv']].rename(columns={'mean_ppmv': 'mean_ppmv_temp'}), on=['Day', 'Year'])

    merged_data = merged_data.sort_values(by = ['Year', 'Day'])

    n_lags = 5
    for i in range(1, n_lags + 1):
        merged_data[f'mean_ppmv_o3_lag_{i}'] = merged_data['mean_ppmv_o3'].shift(i)
        merged_data[f'mean_ppmv_co_lag_{i}'] = merged_data['mean_ppmv_co'].shift(i)
        merged_data[f'mean_ppmv_temp_lag_{i}'] = merged_data['mean_ppmv_temp'].shift(i)

    merged_data = merged_data.dropna()
    merged_data = merged_data[merged_data['Day'] >= 6]

    # Using 2005 - 2022 for training
    train_data = merged_data[merged_data['Year'] <= '2022']

    # Testing on final year  
    test_data = merged_data[merged_data['Year'] >= '2023']


    features = ['mean_ppmv_o3_lag_1', 'mean_ppmv_o3_lag_2', 'mean_ppmv_o3_lag_3', 'mean_ppmv_o3_lag_4', 'mean_ppmv_o3_lag_5',
                'mean_ppmv_co_lag_1', 'mean_ppmv_co_lag_2', 'mean_ppmv_co_lag_3', 'mean_ppmv_co_lag_4', 'mean_ppmv_co_lag_5',
                'mean_ppmv_temp_lag_1', 'mean_ppmv_temp_lag_2', 'mean_ppmv_temp_lag_3', 'mean_ppmv_temp_lag_4', 'mean_ppmv_temp_lag_5',
                'mean_ppmv_co', 'mean_ppmv_temp']

    target_variable = 'mean_ppmv_o3'

    X_train, y_train = train_data[features], train_data[target_variable]
    X_test, y_test = test_data[features], test_data[target_variable]

    return X_train, X_test, y_train, y_test, test_data

"""
Follows same algorithm as above except training on July and Testing on October
"""
def load_data_jul_oct(dataset_location = "/Users/max/OneDrive - University of Otago/MLS datasets/Latitude75-82S"):
    o3_by_day = []
    co_by_day = []
    temp_by_day = [] 
    for filename in os.listdir(dataset_location):
        file_path = os.path.join(dataset_location, filename)
        
        df = pd.read_csv(file_path)
        
        if file_path.endswith("O3.csv") or file_path.endswith("O3_v.csv"):
            year = filename.split('_')[0]
            o3_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            o3_df['Year'] = year
            o3_by_day.append(o3_df)
    
        if file_path.endswith("CO.csv") or file_path.endswith("CO_v.csv"):
            year = filename.split('_')[0]
            co_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            co_df['Year'] = year
            co_by_day.append(co_df)

        if file_path.endswith("Temperature.csv") or file_path.endswith("Temperature_v.csv"):
            year = filename.split('_')[0]
            temp_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            temp_df['Year'] = year
            temp_by_day.append(temp_df)

    o3_data = pd.concat(o3_by_day)
    co_data = pd.concat(co_by_day)
    temp_data = pd.concat(temp_by_day)

    merged_data = pd.merge(o3_data, co_data, on=['Day', 'Year'], suffixes=('_o3', '_co'))
    merged_data = pd.merge(merged_data, temp_data[['Day', 'Year', 'mean_ppmv']].rename(columns={'mean_ppmv': 'mean_ppmv_temp'}), on=['Day', 'Year'])

    merged_data = merged_data.sort_values(by = ['Year', 'Day'])

    n_lags = 5
    for i in range(1, n_lags + 1):
        merged_data[f'mean_ppmv_o3_lag_{i}'] = merged_data['mean_ppmv_o3'].shift(i)
        merged_data[f'mean_ppmv_co_lag_{i}'] = merged_data['mean_ppmv_co'].shift(i)
        merged_data[f'mean_ppmv_temp_lag_{i}'] = merged_data['mean_ppmv_temp'].shift(i)

    merged_data = merged_data.dropna()
    merged_data = merged_data[merged_data['Day'] <= 105]
    merged_data = merged_data[merged_data['Day'] >= 6]

    # Using July for training
    train_data = merged_data[merged_data['Day'] <= 31]

    # Testing on October ozone  
    test_data = merged_data[merged_data['Day'] >= 74]

    features = ['mean_ppmv_o3_lag_1', 'mean_ppmv_o3_lag_2', 'mean_ppmv_o3_lag_3', 'mean_ppmv_o3_lag_4', 'mean_ppmv_o3_lag_5',
                'mean_ppmv_co_lag_1', 'mean_ppmv_co_lag_2', 'mean_ppmv_co_lag_3', 'mean_ppmv_co_lag_4', 'mean_ppmv_co_lag_5',
                'mean_ppmv_temp_lag_1', 'mean_ppmv_temp_lag_2', 'mean_ppmv_temp_lag_3', 'mean_ppmv_temp_lag_4', 'mean_ppmv_temp_lag_5',
                'mean_ppmv_co', 'mean_ppmv_temp']

    target_variable = 'mean_ppmv_o3'

    X_train, y_train = train_data[features], train_data[target_variable]
    X_test, y_test = test_data[features], test_data[target_variable]

    return X_train, X_test, y_train, y_test, test_data


"""
Trains on 11 years
Tests on 12th 
If predicted values provided replaces relevant year of o3 with predicted values
"""
def load_data_11_years(dataset_location, start_year=2005, predicted_values=[]):
    o3_by_day = []
    co_by_day = []
    temp_by_day = []
    replace_data = False

    # A full solar cycle is 11 years, then test on the following year so 12 total
    end_year = start_year + 12

    # Determine if there are predicted values to replace actual data
    if len(predicted_values) > 0:
        replace_data = True

    for filename in os.listdir(dataset_location):
        file_path = os.path.join(dataset_location, filename)
        
        df = pd.read_csv(file_path)
        
        # Extracting O3 data
        if file_path.endswith("O3.csv") or file_path.endswith("O3_v.csv"):
            year = filename.split('_')[0]
            o3_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            o3_df['Year'] = year
            o3_by_day.append(o3_df)
    
        # Extracting CO data
        if file_path.endswith("CO.csv") or file_path.endswith("CO_v.csv"):
            year = filename.split('_')[0]
            co_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            co_df['Year'] = year
            co_by_day.append(co_df)

        # Extracting Temperature data
        if file_path.endswith("Temperature.csv") or file_path.endswith("Temperature_v.csv"):
            year = filename.split('_')[0]
            temp_df = df.groupby('Day')['mean_ppmv'].mean().reset_index()
            temp_df['Year'] = year
            temp_by_day.append(temp_df)

    o3_data = pd.concat(o3_by_day)
    co_data = pd.concat(co_by_day)
    temp_data = pd.concat(temp_by_day)

    # Sorting the data
    o3_data.sort_values(by=['Year', 'Day'], inplace=True)
    co_data.sort_values(by=['Year', 'Day'], inplace=True)
    temp_data.sort_values(by=['Year', 'Day'], inplace=True)

    # Merging the datasets
    merged_data = pd.merge(o3_data, co_data, on=['Day', 'Year'], suffixes=('_o3', '_co'))
    merged_data = pd.merge(merged_data, temp_data[['Day', 'Year', 'mean_ppmv']].rename(columns={'mean_ppmv': 'mean_ppmv_temp'}), on=['Day', 'Year'])
    merged_data.sort_values(by=['Year', 'Day'], inplace=True)

    # Filtering data for the selected years
    merged_data = merged_data[(merged_data['Year'] >= str(start_year)) & (merged_data['Year'] <= str(end_year))]

    # Replacing the mean_ppmv for the predicted year if predicted_values are provided
    if replace_data:
        for i in range(len(predicted_values)):
            pred_year_str = str(end_year - len(predicted_values) + i)  
            if len(predicted_values[i]) == len(merged_data[merged_data['Year'] == pred_year_str]):
                merged_data.loc[merged_data['Year'] == pred_year_str, 'mean_ppmv_o3'] = predicted_values[i]
            else:
                print("The length of predicted_values does not match the data for the replacement year.")

    # Creating lag features
    n_lags = 5
    for i in range(1, n_lags + 1):
        merged_data[f'mean_ppmv_o3_lag_{i}'] = merged_data['mean_ppmv_o3'].shift(i)
        merged_data[f'mean_ppmv_co_lag_{i}'] = merged_data['mean_ppmv_co'].shift(i)
        merged_data[f'mean_ppmv_temp_lag_{i}'] = merged_data['mean_ppmv_temp'].shift(i)

    # Dropping rows with NaN values created by shifting for lags
    merged_data.dropna(inplace=True)

    # Splitting the data into training and testing datasets
    train_data = merged_data[merged_data['Year'] < str(end_year)]
    test_data = merged_data[merged_data['Year'] == str(end_year)]

    features = ['mean_ppmv_o3_lag_1', 'mean_ppmv_o3_lag_2', 'mean_ppmv_o3_lag_3', 'mean_ppmv_o3_lag_4', 'mean_ppmv_o3_lag_5',
                'mean_ppmv_co_lag_1', 'mean_ppmv_co_lag_2', 'mean_ppmv_co_lag_3', 'mean_ppmv_co_lag_4', 'mean_ppmv_co_lag_5',
                'mean_ppmv_temp_lag_1', 'mean_ppmv_temp_lag_2', 'mean_ppmv_temp_lag_3', 'mean_ppmv_temp_lag_4', 'mean_ppmv_temp_lag_5',
                'mean_ppmv_co', 'mean_ppmv_temp']

    target_variable = 'mean_ppmv_o3'

    X_train, y_train = train_data[features], train_data[target_variable]
    X_test, y_test = test_data[features], test_data[target_variable]

    return X_train, X_test, y_train, y_test, test_data


# load_data_11_years(dataset_location)

# code for plotting the data
# insert after if file_path.ends with statement if you wanna see it: 
      # Plotting ozone data
    # plt.figure(figsize=(10, 6))
    # for o3_df in o3_by_day:
    #     year = o3_df['Year'].iloc[0]  # Extract year from the DataFrame
    #     sns.lineplot(x='Day', y='mean_ppmv', data=o3_df, marker='o', label=f'O3 - {year}')
    # plt.xlabel('Day')
    # plt.ylabel('Mean O3 ppmv')
    # plt.title('Mean O3 ppmv Across Days')
    # plt.legend()
    # plt.show()

    # # Plotting CO data
    # plt.figure(figsize=(10, 6))
    # for co_df in co_by_day:
    #     sns.lineplot(x='Day', y='mean_ppmv', data=co_df, marker='o', label=f'CO - {co_df["Year"].iloc[0]}')
    # plt.xlabel('Day')
    # plt.ylabel('Mean CO ppmv')
    # plt.title('Mean CO ppmv Across Days')
    # plt.legend()
    # plt.show()

    # # Plotting temperature data
    # plt.figure(figsize=(10, 6))
    # for temp_df in temp_by_day:
    #     sns.lineplot(x='Day', y='mean_ppmv', data=temp_df, marker='o', label=f'Temperature - {temp_df["Year"].iloc[0]}')
    # plt.xlabel('Day')
    # plt.ylabel('Mean Temperature ppmv')
    # plt.title('Mean Temperature ppmv Across Days')
    # plt.legend()
    # plt.show()