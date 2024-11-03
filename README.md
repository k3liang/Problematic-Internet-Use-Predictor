# Problematic-Internet-Use-Predictor

Download of the data can be done at this [Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)

Reference for using `pandas` on the cluster: [cudf.pandas](https://github.com/rapidsai/cudf)

`EDA.ipynb`  
Exploratory Data Analysis step


Preprocessing Steps

Our data from the Child Mind Institue contains some sort of null data in every row. Therefore, we are planning to drop a column if it contains more than 50% null data.
To address the remaining null values, we will use imputing to insert random values based on the existing values in the particular column. 

Judging from our scatter plots of the data, the data is not normally distributed, therefore we are opting for data normalization instead of standardization.

Lastly, for the columns that contain seasonal data, we will use one hot encoding to transform the cateogrical data into numerical data. 
