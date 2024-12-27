import numpy as np
import pandas as pd
import statsmodels.api as sm


# helper functions
# note that I never need the date directly in any of the methods
# I can just make prediction for all test observations, and keep track of their indices for control and treatment data
# assume all the data frames passed in here do not contain dates, but converted to integers, and maintaining same order as original data

# randomly generate control and treated data for a given test data, treatment effect and sample size
def generate_exp_data(test_df, te, n):
    test_len = len(test_df)
    treatment_idx = np.random.choice(test_len, n, replace=True)
    control_idx = np.random.choice(test_len, n, replace=True)
    
    treatment_df = test_df.iloc[treatment_idx].copy().reset_index(drop=True)
    control_df = test_df.iloc[control_idx].copy().reset_index(drop=True)
    
    treatment_df['y'] += te  # Apply treatment effect
    
    return control_df, control_idx, treatment_df, treatment_idx

# pre-processing data for the traffic data
# remove the datetime stamp, and convert to integer
# should be used with the entire test data before splitting to control and treatment
def traffic_preprocess(df):
    df['x1'] = (df['x'] - min(df['x'])).dt.days
    return df.drop(columns=['x'])

# helper for checking whether something is a scalar or a matrix, and convert a scalar to a 1x1 matrix
def to_matrix(x):
    if np.isscalar(x):
        return np.array([[x]])  # Scalar to 1x1 matrix
    if x.ndim == 1:
        return np.reshape(x, (-1, 1))  # 1D array to column vector
    return np.array(x)  # 2D array stays as it is


# calculate the average treatment effect using constant method
def calculate_ate(control_df, treatment_df):

    # calculate the average treatment effect and standard error
    ate = treatment_df['y'].mean() - control_df['y'].mean()
    se = np.sqrt(control_df['y'].var()/len(control_df) + treatment_df['y'].var()/len(treatment_df))

    # return results
    return ate, se
    
# calculate the average treatment effect using regression method
def calculate_ate_reg(control_df, treatment_df):

    # split data into outcomes and covariates 
    y_1 = to_matrix(treatment_df['y'])
    y_0 = to_matrix(control_df['y'])
    x_0 = to_matrix(control_df.drop(columns = ['y']))
    x_1 = to_matrix(treatment_df.drop(columns = ['y']))

    # centered outcome and covariates for parameter estimation
    yc_1 = y_1 - np.mean(y_1)
    yc_0 = y_0 - np.mean(y_0)
    xc_1 = x_1 - np.mean(x_1, axis=0)
    xc_0 = x_0 - np.mean(x_0, axis=0)

    # calculate the beta coefficients
    treatxvar = xc_1.T @ xc_1
    controlxvar = xc_0.T @ xc_0
    treatxcov = xc_1.T @ yc_1
    controlxcov = xc_0.T @ yc_0
    beta = np.linalg.inv(treatxvar + controlxvar) @ (treatxcov + controlxcov)
    
    # calculate the average treatment effect and standard error
    xbar_1 = to_matrix(np.mean(x_1, axis=0)) # np.mean reduces the dimension so we need to turn it back to matrix
    xbar_0 = to_matrix(np.mean(x_0, axis=0))
    alpha_1 = np.mean(y_1) - xbar_1.T @ beta
    alpha_0 = np.mean(y_0) - xbar_0.T @ beta
    reg_est = alpha_1 - alpha_0
    reg_se = (np.var(y_1 - x_1 @ beta, ddof=1)/len(treatment_df) + 
              np.var(y_0 - x_0 @ beta, ddof=1)/len(control_df))**0.5 # no need to transpose x. x is n x p, beta is p x 1
    
    # return results - reg_est is in matrix form, so need to extract the scalar
    return reg_est[0][0], reg_se

# calculate the average treatment effect using prediction method
def calculate_ate_pred(control_df, treatment_df, preds, control_idx, treatment_idx):
    y_1 = treatment_df['y']
    y_0 = control_df['y']
    # get prediction for the instance of control and treatment data
    control_pred = preds[control_idx].reset_index(drop=True)
    treatment_pred = preds[treatment_idx].reset_index(drop=True)

    # calculate the average treatment effect and standard error
    ate = np.mean(y_1-treatment_pred) - np.mean(y_0-control_pred)
    se = (np.var(y_1 - treatment_pred)/len(treatment_df) + 
          np.var(y_0 - control_pred)/len(control_df))**0.5
    
    # return results
    return ate, se

#calculate the average treatment effect using secondary adjustment method
def calculate_ate_sec(control_df, treatment_df, preds, control_idx, treatment_idx):
    
    # get the remainder outcome after removing the prediction
    control_pred = preds[control_idx].reset_index(drop=True)
    treatment_pred = preds[treatment_idx].reset_index(drop=True)
    y_1 = to_matrix(treatment_df['y'] - treatment_pred)
    y_0 = to_matrix(control_df['y'] - control_pred)

    # get covariates
    x_0 = to_matrix(control_df.drop(columns = ['y']))
    x_1 = to_matrix(treatment_df.drop(columns = ['y']))

    # centered outcome and covariates for beta estimation
    yc_1 = y_1 - np.mean(y_1)
    yc_0 = y_0 - np.mean(y_0)
    xc_1 = x_1 - np.mean(x_1, axis=0)
    xc_0 = x_0 - np.mean(x_0, axis=0)

    # calculate the beta coefficients
    treatxvar = xc_1.T @ xc_1
    controlxvar = xc_0.T @ xc_0
    treatxcov = xc_1.T @ yc_1
    controlxcov = xc_0.T @ yc_0
    beta = np.linalg.inv(treatxvar + controlxvar) @ (treatxcov + controlxcov)
    
    # calculate the average treatment effect and standard error
    xbar_1 = to_matrix(np.mean(x_1, axis=0)) # np.mean reduces the dimension so we need to turn it back to matrix
    xbar_0 = to_matrix(np.mean(x_0, axis=0))
    alpha_1 = np.mean(y_1) - xbar_1.T @ beta
    alpha_0 = np.mean(y_0) - xbar_0.T @ beta
    reg_est = alpha_1 - alpha_0
    reg_se = (np.var(y_1 - x_1 @ beta, ddof=1)/len(treatment_df) + 
              np.var(y_0 - x_0 @ beta, ddof=1)/len(control_df))**0.5 # no need to transpose x. x is n x p, beta is p x 1
    
    # return results - reg_est is in matrix form, so need to extract the scalar
    return reg_est[0][0], reg_se


    
    
# tests for the helper functions----------------------------------------------------------------------------------------------


'''# read data
cal_data = pd.read_excel('../data/CalTransit_Dataset/pems_output.xlsx')
cal_data.rename({"Time": "x", "# Incidents": "y"}, axis=1, inplace=True)
# split to training and test data 
train_df = cal_data[cal_data['x'] < '2020-03-19'].reset_index(drop=True)
test_df = cal_data[cal_data['x']>='2020-03-19'].reset_index(drop=True)
test_df['x1'] = (test_df['x'] - min(test_df['x'])).dt.days

# fit a predictive model
opt_model = sm.tsa.statespace.SARIMAX(train_df['y'], order = (1, 0, 1), seasonal_order = (0, 1, 1, 7))
preds = opt_model.fit(disp=False).get_prediction(start = len(train_df), end = len(train_df) + len(test_df) - 1).predicted_mean.reset_index(drop=True)   

# generate control and treatment data
te = 10
n = 100
control_df, control_idx, treatment_df, treatment_idx = generate_exp_data(test_df, te, n)
# drop dates when inputting to calculate functions
control_df = control_df.drop(columns=['x'])
treatment_df = treatment_df.drop(columns=['x'])

# test calculate_ate
ate, se = calculate_ate(control_df, treatment_df)
true_ate = treatment_df['y'].mean() - control_df['y'].mean()
true_se = np.sqrt(control_df['y'].var()/len(control_df) + treatment_df['y'].var()/len(treatment_df))
print(ate, se)

# test calculate_ate_reg
ate, se = calculate_ate_reg(control_df, treatment_df)
print(ate, se)

# test calculate_ate_pred
ate, se = calculate_ate_pred(control_df, treatment_df, preds, control_idx, treatment_idx)
print(ate, se)

# test calculate_ate_sec
ate, se = calculate_ate_sec(control_df, treatment_df, preds, control_idx, treatment_idx)
print(ate, se)'''
