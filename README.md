
# California House Price Prediction

It serves as an excellent introduction to implementing machine learning algorithms because it requires data cleaning, has an easily understandable list of variables and sits at an optimal size.

For this project we have taken a dataset having house prices based on areas in California region. We have to predict the prices, since the price prediction is continuous it is Regression problem in machine learning.

The dataset was based on data from the 1990 California census. It may not help us with predicting current housing prices like the Zillow Zestimate dataset, it does provide an introductory dataset for teaching people about the basics of machine learning. This dataset has metrics such as :- 
* longitude
* latitude
* housing_media_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* ocean_proximity
* median_house_value
These metrics are for each block group in California. A block
group typically has a population of 600 to 3,000 people. We will just call them “districts” for short.

There are 20640 entries in dataset. 10 columns. We have to find **median_house_value**.

The libraries used in the project are below

## Libraries used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```

### Steps involved
- Load the dataset into a dataframe.
- Check the shape of the dataset, feature names and data types of the features.
- Out of 10 features 1 is categorical and 9 features are numerical.
- total_bedrooms feature has 20433 entries whereas there 20640 total entries. There are missing values for 207 districts, we replace them with mean value for total_bedrooms. The feature ocean_proximity is of object type and has 3 values on using *value_counts()*.
- *describe()* yields us count, min, max, std, 25%(percentile), 50%(percentile), 75%(percentile) values for each numerical features.
- Plotted graph to see the distribution of numerical features.
- On plotting longitude and latitude as a scatter plot we get map resembling California.
- We plot a heatmap to see the correlation between features.
- We have used pandas *get_dummies()* to convert values in ocean_proximity for encoding.
- We used *train_test_split* for training and testing data.
- Scaling is applied on the training and testing to avoid any data leakage.
- Predicted values using *Linear Regression, Decision Tree Regressor and Random Forest Regression*. *RMSE* value for *Random Forest Regression* is best out of all 3.

## Acknowledgements

 This data was initially featured in the following paper: Pace, R. Kelley, and Ronald Barry. "Sparse spatial autoregressions." Statistics & Probability Letters 33.3 (1997): 291-297.
 