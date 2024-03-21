from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import numpy as np 
from load_data import load_data, load_data_jul_oct, load_data_11_years
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
# import dataframe_image as dfi

# Inner vortex
inner_dataset_location = "/Users/max/OneDrive - University of Otago/MLS datasets/InnerVortexOnly75-82S"
# Office pc 
# inner_dataset_location = "/Users/maximus/OneDrive - University of Otago/MLS datasets/InnerVortexOnly75-82S"

# 75-82 Lat
# Laptop
lat_dataset_location = "/Users/max/OneDrive - University of Otago/MLS datasets/Latitude75-82S"
# Office pc 
# lat_dataset_location = "/Users/maximus/OneDrive - University of Otago/MLS datasets/Latitude75-82S"

model_mse = []

def get_X_y(dataset_location, load_data = load_data, get_test_data = False):
    X_train, X_test, y_train, y_test, test_data = load_data(dataset_location)
    if get_test_data:
        return X_train, X_test, y_train, y_test, test_data
    else:
        return X_train, X_test, y_train, y_test


def make_model(X_train, y_train):
    
    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


def graph_model_performance(y_pred, test_data, title = ""):
    test_data["pred_mean_ppmv_o3"] = y_pred

    plt.figure(figsize = (12, 6))
    plt.title(title)
    sns.lineplot(x = 'Day', y = 'mean_ppmv_o3', data = test_data, label = "Actual O3")
    sns.lineplot(x = 'Day', y = 'pred_mean_ppmv_o3', data = test_data, label = "Predicted O3")
    plt.ylabel("Mean O3")
    
    plt.legend()
    
    plt.show()

# # Model trained on years up to 2022 and tested on 2023 ozone 
# X_train, X_test, y_train, y_test, test_data = get_X_y(lat_dataset_location, load_data, True)
# lat_model = make_model(X_train, y_train)

# lat_y_pred = lat_model.predict(X_test)

# lat_mse = mean_squared_error(y_test, lat_y_pred)
# # print(f"Mean squared error: {lat_mse}")
# model_mse.append(lat_mse)

# graph_model_performance(lat_y_pred, test_data, "Lat75-82S Ozone Predictions for Year 2023 vs Actual")



# # Model that makes predictions on October ozone based on July 
# X_train, X_test, y_train, y_test, test_data = get_X_y(lat_dataset_location, load_data_jul_oct, True)
# jul_oct_model = make_model(X_train, y_train)
# oct_y_pred = jul_oct_model.predict(X_test)

# oct_mse = mean_squared_error(y_test, oct_y_pred)
# # print(f"Mean squared error jul-oct: {oct_mse}")
# model_mse.append(oct_mse)

# graph_model_performance(oct_y_pred, test_data, "October Predictions for Ozone based on July data")


# # Model using Inner vortex dataset
# X_train, X_test, y_train, y_test, test_data = get_X_y(inner_dataset_location, get_test_data = True)
# inner_model = make_model(X_train, y_train)

# inner_y_pred = inner_model.predict(X_test)

# inner_mse = mean_squared_error(y_test, inner_y_pred)
# # print(f"Mean squared error: {inner_mse}")
# model_mse.append(inner_mse)

# graph_model_performance(inner_y_pred, test_data, "Inner Vortex Ozone Predictions for Year 2023 Lat75-82S")


# # Model Inner Vortex Jul_Oct 
# X_train, X_test, y_train, y_test, test_data = get_X_y(inner_dataset_location, load_data_jul_oct, True)
# oct_inner_model = make_model(X_train, y_train)

# oct_inner_y_pred = oct_inner_model.predict(X_test)

# oct_inner_mse = mean_squared_error(y_test, oct_inner_y_pred)
# # print(f"Mean squared error: {oct_inner_mse}")
# model_mse.append(oct_inner_mse)

# graph_model_performance(oct_inner_y_pred, test_data, "Inner Vortex October Ozone Predictions based On July")


# # Overall Lat model predicting Inner Ozone
# X_train, X_test, y_train, y_test, lat_test_data = get_X_y(lat_dataset_location, get_test_data = True)
# lat_model = make_model(X_train, y_train)

# # Now make model predict Inner vortex data
# X_train, X_test, y_train, y_test, inner_test_data = get_X_y(inner_dataset_location, get_test_data = True)

# inner_y_pred = lat_model.predict(X_test)

# inner_lat_mse = mean_squared_error(y_test, inner_y_pred)
# # print(f"Mean squared error: {inner_lat_mse}")
# model_mse.append(inner_lat_mse)

# graph_model_performance(inner_y_pred, inner_test_data, "Lat75-82S Model Predicting Inner Vortex Ozone levels for Year 2023")


# # Graphing Inner vortex ozone vs 75-82S Ozone
# X_train, X_test, y_train, y_test, lat_test_data = get_X_y(lat_dataset_location, get_test_data = True)
# X_train, X_test, y_train, y_test, inner_test_data = get_X_y(inner_dataset_location, get_test_data = True)

# plt.figure(figsize=(12, 6))
# sns.lineplot(y = "mean_ppmv_o3", x = "Day", data = lat_test_data, label = "Lat75-82S Ozone levels")
# sns.lineplot(y = "mean_ppmv_o3", x = "Day", data = inner_test_data, label = "Inner Vortex Ozone Levels")
# plt.title("Inner Vortex Ozone levels vs Lat75-82S Ozone Levels for Year 2023 (Test Data)")

# plt.show()


# # July model predicting Inner vortex ozone in October

# X_train, X_test, y_train, y_test, lat_test_data = get_X_y(lat_dataset_location, load_data_jul_oct, True)
# lat_jul_oct_model = make_model(X_train, y_train)

# X_train, X_test, y_train, y_test, inner_test_data = get_X_y(inner_dataset_location, load_data_jul_oct, True)
# inner_oct_y_pred = lat_jul_oct_model.predict(X_test)

# inner_oct_mse = mean_squared_error(y_test, inner_oct_y_pred)
# # print(f"Mean Squared Error: {inner_oct_mse}")
# model_mse.append(inner_oct_mse)

# graph_model_performance(inner_oct_y_pred, inner_test_data, "Model trained on Lat75-82S Ozone in July Predicting Inner Ozone Levels in October")


# # Model trained on Inner Vortex Predicting Lat75-82S data
# X_train, X_test, y_train, y_test, inner_test_data = get_X_y(inner_dataset_location, load_data, True)
# inner_model = make_model(X_train, y_train)

# X_train, X_test, y_train, y_test, lat_test_data = get_X_y(lat_dataset_location, load_data, True)
# inner_lat_y_pred = inner_model.predict(X_test)

# inner_mse = mean_squared_error(y_test, inner_lat_y_pred)
# # print(f"Mean Squared Error: {inner_mse}")
# model_mse.append(inner_mse)

# graph_model_performance(inner_lat_y_pred, lat_test_data, "Inner Vortex Model Predicting Lat75-82S Ozone for Year 2023")


# # Model trained on Inner Vortex July data predicting October Lat75-82S
# X_train, X_test, y_train, y_test = get_X_y(inner_dataset_location, load_data = load_data_jul_oct)
# inner_jul_model = make_model(X_train, y_train)

# X_train, X_test, y_train, y_test, test_data = get_X_y(lat_dataset_location, load_data_jul_oct, True)
# oct_y_pred = inner_jul_model.predict(X_test)

# oct_mse = mean_squared_error(y_test, oct_y_pred)
# # print(f"Mean Squared Error: {oct_mse}")
# model_mse.append(oct_mse)

# graph_model_performance(oct_y_pred, test_data, "Inner Vortex July Model Predicting Lat75-82S October Ozone Levels")


# # Making table with mse's for all the models
# figures = []
# for i in range(1, 9):
#     figures.append(f"Figure {i}")

# mse_df = []

# for i in range(len(figures)):
#     mse_df.append([figures[i], model_mse[i]])

# df = pd.DataFrame(mse_df, columns = ['Models / Figures', 'Mean Squared Errors'])

# df_styled = df.style.background_gradient()
# dfi.export(df, "mse_table.png")


# Now going to make some linear models trained on 11 year cycle


predicted_values = []
actual_values = []
predicted_cycle_values = []
solar_mse = []

# Doing 7 iterations so we go up to 2023, at which point 6 years of the o3 data will be completely made up of predicted values by the model
for i in range(7):
    X_train, X_test, y_train, y_test, test_data = load_data_11_years(lat_dataset_location, 2005 + i, predicted_values)
    solar_cycle_model = make_model(X_train, y_train)

    y_pred = solar_cycle_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    predicted_values.append(y_pred)
    solar_mse.append(mse)
    actual_values.append(test_data)

for i in range(7):
    X_train, X_test, y_train, y_test, test_data = load_data_11_years(lat_dataset_location, 2005 + i)
    solar_cycle_model = make_model(X_train, y_train)

    y_pred = solar_cycle_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    predicted_cycle_values.append(y_pred)
    solar_mse.append(mse)

for i in range(len(predicted_values)):
    actual_values[i]["pred_mean_ppmv_o3"] = predicted_values[i]
    actual_values[i]["solar_pred_mean_ppmv_o3"] = predicted_cycle_values[i]
    year = 2016 + i
    plt.figure(figsize = (12, 6))
    plt.title(f"Predicted vs Actual O3 on 11 Year cycle - {year}")
    sns.lineplot(x = 'Day', y = 'mean_ppmv_o3', data = actual_values[i], label = "Actual O3")
    sns.lineplot(x = 'Day', y = 'pred_mean_ppmv_o3', data = actual_values[i], label = "Predicted O3 - trained on predicted values")
    sns.lineplot(x = 'Day', y = 'solar_pred_mean_ppmv_o3', data = actual_values[i], label = "Predicted O3 - trained on actual values")
    plt.ylabel("Mean O3")

plt.legend()
plt.show()



# Making table with mse's for all the models
figures = []
for i in range(1, 7):
    figures.append(f"Figure {i}")

mse_df = []

for i in range(len(solar_mse)):
    mse_df.append([figures[i], solar_mse[i]])

df = pd.DataFrame(mse_df, columns = ['Models / Figures', 'Mean Squared Errors'])

df_styled = df.style.background_gradient()
dfi.export(df, "solar_mse_table.png")