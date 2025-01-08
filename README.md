# Netflix UCAN Revenue Analysis & Forecasting  

![Figure_1](https://github.com/user-attachments/assets/9da0910d-a1a7-41ac-abaf-8c56fafc6c54)

[Netflix Revenue Analysis.pdf](https://github.com/user-attachments/files/18340053/Netflix.Revenue.Analysis.pdf)

## Project Overview  
In this project, I used linear regression to analyze and forecast Netflix's streaming revenue in the US and Canada (UCAN) over time. The goal was to identify revenue trends, improve model accuracy with feature scaling, and predict future revenue growth.

## Requirements  
- **Python**  
- **Libraries**:  
  - `pandas`  
  - `matplotlib`  
  - `seaborn`  
  - `numpy`  
  - `scikit-learn`  

## Project Steps  

### 1. Load Data  
- Make sure `netflix_revenue.csv` is in the same directory as the script.  
- The dataset should have these two columns:  
  - `Date`: Time points (monthly or yearly).  
  - `UCAN Streaming Revenue`: Revenue in millions.  

### 2. Data Preprocessing  
- Convert the `Date` column into a datetime format.  
- Create a numerical representation of the `Date` column (`Date_ordinal`) analysis.  

### 3. Visualization  
- Plot `UCAN Streaming Revenue` over time to observe trends or patterns.  

### 4. Linear Regression  
- **Baseline Model**:  
  - Run a regression on the unscaled data to establish a baseline.  
- **Feature Scaling**:  
  - Apply `StandardScaler` to scale the data.  
  - Refit the regression model with the scaled data for clearer analysis.  

### 5. Forecasting  
- Extend the dataset by adding future dates.  
- Use the trained model to predict Netflix's future streaming revenue in the UCAN location.  

### 6. Plotting  
- Visualize the regression line for both the original and scaled data.  
- Plot forecasts to assess potential future trends and revenue growth.
