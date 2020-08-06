# Linear-Rergression
HousePrice Prediction 
## importing Libraries
import pandas as pd
import seaborn as sns
import numpy as np
%matplotlib inline

## data analysis
auto=pd.read_csv('Philadelphia_Crime_Rate_noNA.csv')
auto.head()
auto.info()
auto['County'].value_counts()
auto['PopChg'].value_counts().head()

auto.describe()

## histogram
import matplotlib.pyplot as plt
auto.hist(bins=50, figsize=(20, 15))

# Correlation 

corr_mat=auto.corr()
corr_mat['HousePrice'].sort_values(ascending=False)

# Ploting HousePrice with its most correlated feature
auto.plot(kind="scatter", x="HsPrc ($10,000)", y="HousePrice", alpha=0.8)

# This graph Looks like overfitting the data 

# Adding a new Feature 

auto["HsPrcCrime"] = auto['HsPrc ($10,000)']/auto['CrimeRate']
auto.head()

auto.plot(kind="scatter", x="HsPrcCrime", y="HousePrice", alpha=0.8)

## Model 1 testing 
feature=['CrimeRate']
x=auto[feature]
y=auto['HousePrice']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(x_train,y_train)
print(linreg)
print(linreg.intercept_)
print(linreg.coef_)
print(linreg.score(x_train,y_train))
print(linreg.score(x_test,y_test))

## Model 2 testing 
feature1=['HsPrcCrime']
X=auto[feature1]
Y=auto['HousePrice']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)

from sklearn.linear_model import LinearRegression
linreg1=LinearRegression()
linreg1.fit(X_train,Y_train)
print(linreg1)
print(linreg1.intercept_)
print(linreg1.coef_)
print(linreg1.score(X_train,Y_train))
print(linreg1.score(X_test,Y_test))

## Model 1 score is less than Model 2 

sns.jointplot(auto['HsPrcCrime'],auto['HousePrice'], kind='reg')

auto_nocc = auto[auto['HousePrice']<400000  ]

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
X_train , X_test, y_train ,y_test= train_test_split(X,y, random_state=1)

feature=['HsPrcCrime']
X=auto_nocc[feature]
y=auto_nocc['HousePrice']
linreg2=LinearRegression()
linreg2.fit(X_train,y_train)
print(linreg2)
print(linreg2.intercept_)
print(linreg2.coef_)
print(linreg2.score(X_train,y_train))
print(linreg2.score(X_test,y_test))


feature=['HsPrcCrime','MilesPhila', 'CrimeRate','PopChg']
X=auto_nocc[feature]
y=auto_nocc['HousePrice']

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
X_train , X_test, y_train ,y_test= train_test_split(X,y, random_state=1)
                                                    
linreg2=LinearRegression()
linreg2.fit(X_train,y_train)
print(linreg2)
print(linreg2.intercept_)
print(linreg2.coef_)
print(linreg2.score(X_train,y_train))
print(linreg2.score(X_test,y_test))


## function for Residual sum of Squares 

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predictions = intercept + slope * input_feature
    
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = predictions - output
    
    # square the residuals and add them up
    square_residual = residuals * residuals
    sum_square_residual = square_residual.sum()
    RSS = sum_square_residual
    
    return(RSS)
    
    
## Function for predicting price from sqft feet 
def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + (slope * input_feature)
    return predicted_values
    
## Function for predicting sqft feet from price 
def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output - intercept)/slope
    return estimated_feature
    
## Testing RSS 
Value=get_residual_sum_of_squares(auto['HsPrcCrime'],auto['HousePrice'], linreg.intercept_ , linreg.coef_)
Value

## Testing Price function 
my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, linreg.intercept_, linreg.coef_)
print(estimated_price )

## Testing sqft function
my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, linreg.intercept_, linreg.coef_)
print(estimated_squarefeet)

