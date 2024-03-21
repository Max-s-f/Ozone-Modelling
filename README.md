## Report of Results

Essentially two main types of models were made in this repo. 
Firstly: a model that is trained on the years 2005 to 2022 and tested on year 2023
Secondly: a model that is trained on data from July only then is made to predict ozone levels in October

These two models were made using both Inner Vortex data and Lat75-82S data

This creates four main models as follows:

Lat75-82S Predicting Ozone in 2023 (Figure 1):
![Figure 1:](/Figures/Lat75-82S-TestYear2023.png)

Inner Vortex data Predicting Ozone in 2023 (Figure 2):
![Figure 2:](/Figures/InnverVortex-TestYear2023.png)

Lat75-82S Predicting October Ozone based on July only (Figure 3):
![Figure 3:](/Figures/Oct-Jul-Lat75-82S.png)

Inner Vortex Predicting October Ozone based on July only (Figure 4):
![Figure 4:](/Figures/Oct-Jul-InnerVortex.png)


Next I looked at how well the model trained on Lat75-82S Data predicted Inner Vortex data for year 2023 (Figure 5):
![Figure 5:](/Figures/Lat75-82S-Predicting-InnerO3-2023.png)

For reference here's a graph that shows how the Inner Vortex Ozone levels differ in 2023 compared to just averaged across Lat75-82S:
![Figure 6:](/Figures/Innervs75-82SlO3Levels2023.png)

Thus we can see that even around the time that ozone levels inside the Vortex diverge from the overall upper latitude regions, the model still performs well

Along a similar vein, here's a model predicting Inner vortex Ozone in October based only on Lat75-82S July Data (Figure 6):
![Figure 7:](/Figures/Lat75-82SJuly-PredictingInnerOctoberOzone.png)


Finally here are similar models except they are trained on Inner Vortex data and made to predict across whole Lat region:
Predicting 2023 based on previous years (Figure 7):
![Figure 8:](/Figures/InnerVortexPredLat7582-2023.png)

Predicting Ozone in October based on July (Figure 8):
![Figure 9:](/Figures/InnerVortJulyPredLat75-82SOct.png)


Finally here is a table of each of the Mean Squared Errors for each model:

![Figure 10:](/Figures/mse_table.png)


### 11 year model
This model has 2 versions: 
- One where the model is trained off the first 11 years then made to predict the 12th, this is then done for all Years, i.e. 2005-2016, 2006-2017 etc.
- One where the same training occurs, however instead of using the actual Ozone values after the first prediction (i.e. after predicting for year 2017), it will use the previously predicted values and train itself off of those for the years it has predicted for. So by the time it is predicting for the year 2023, it will be trained off of 5 years worth of its own predictions

Here's it's predicitons for the year 2023:
![Figure 11:](/Figures/solar-cycle-2022.png)



### Day to Month Conversion:
Jul: 1 - 31
Aug: 32 - 62
Sept: 63 - 73
Oct: 74 - 105
Nov: 106 - 135
Dec: 136 - 167


#### To do:
- Fix solar model mse table
- Ensure that I'm replacing data correctly 