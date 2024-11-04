# Problematic-Internet-Use-Predictor

Download of the data can be done at this [Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data).  
*Note that, per the competition rules, we cannot publish the dataset on this repo or in our notebooks, so the grader will have to join the competition to get access to the 6GB dataset.*

Reference for using `pandas` on the cluster: [cudf.pandas](https://github.com/rapidsai/cudf)

`EDA.ipynb`  
Exploratory Data Analysis step

`DataPP.ipynb` (incomplete)  
Preprocessing Steps  
*Note that we have gotten a little head start on the preprocessing for milestone 3 already, but here's our preprocessing plan/outline:*

The dataset is strange in the fact that a lot of the measured values are 0 when they shouldn't be. A child that weighs 0 does not exist, so we assume that the data recorder put a 0 in some places when they should've put NaN. In other words, for some of the columns, we'll have to replace 0 with NaN.

It's also worthy to note that the dataset contains some sort of null data in every row. In other words, if we just do df.dropna(), the entire dataset gets dropped. 
To help mitigate this issue, we are planning to drop a column if it contains more than 50% null data.

There are also a lot of outliers for each column, so we are planning to remove rows if they have an outlier.

To address the remaining missing values, we will use imputing to insert random values based on the existing values in the particular column. 

We will also scale our data.  
Judging from the histograms of the data, most of the data is not normally distributed, therefore we are opting for min-max normalization instead of standardization.

Lastly, for the columns that contain seasonal data (which is categorical), we will use one hot encoding to transform the cateogrical data into numerical data. 
