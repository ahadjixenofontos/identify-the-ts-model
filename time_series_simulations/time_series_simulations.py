import tslearn

# has warping path visualization functionality
import dtaidistance # only calculates euclidean distance to speed up C-code underneath by avoiding extra function calls
from dtaidistance import dtw_visualisation as dtwvis

import fastdtw
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
import numpy as np
import pandas as pd
import scipy as sp
import random
import matplotlib.pyplot as plt

# simulate ar, ma, arma or arima process
# accepting defaults for ar_params or ma_params will produce an ma or ar model respectively
# set arima=True for an arima model with d=1
def sim_arima(age, ar_params = [1, 0.], ma_params = [1, 0.], arima = False, d = 1):

    np.random.seed(98765)
    
    burn = int(age/20)
    alphas = np.array(ar_params)
    betas = np.array(ma_params)
    
    if sum(ar_params) != 1:
        ar = np.r_[1, -alphas]
    else:
        ar = np.array([1, 0.])
    
    if sum(ma_params) != 1:
        ma = np.r_[1, betas]
    else:
        ma = np.array([1, 0.])
    
    y = arma_generate_sample(ar=ar, ma=ma, nsample=age)#, burnin=burn)
    
    # if arima process requested,"integrate" by taking the cumulative sum
    if arima:
        for i in range(0,d):
            y = np.cumsum(y)
    
    return y

# nlags represents the order of the model for the p or q
# nsamples represents the number of time series to be simulated
def get_coefficients(nlags, nsamples):
    coefficients = []
    for sample in range(0, nsamples):
        # get nlags random numbers between 0 and 1 
        coefficients.append([random.random() for coef in range(0, nlags)])    
    return coefficients

# populate the simulated series lists by looping through 
def populate_sim_lists(age_dist, ar_parameters = [], ma_parameters = [], arima = False, d = 1):
    
    # get the range to loop through
    if len(ar_parameters) > 0:
        nsamples = len(ar_parameters)
    elif len(ma_parameters) > 0:
        nsamples = len(ma_parameters)
        
    # initiatlize list to hold simulated time series
    simulated_series_list = [] 
    
    # loop through the coefficients and generate time series
    for index in range(0, nsamples):
        age = random.sample(age_dist, 1)[0]
        
        # no alphas 
        if len(ar_parameters) == 0:
            simulated_series = sim_arima(age, 
                                         ma_params=ma_parameters[index],
                                         arima=arima,
                                         d=d)    

        # no betas
        elif len(ma_parameters) == 0:
            simulated_series = sim_arima(age, 
                                         ar_params=ar_parameters[index],
                                         arima=arima,
                                         d=d)   
        # both alphas and betas
        else:
            simulated_series = sim_arima(age, 
                                         ar_params=ar_parameters[index],
                                         ma_params=ma_parameters[index],
                                         arima=arima,
                                         d=d)   
            
        simulated_series_list.append(simulated_series) 

    return simulated_series_list

# this function transforms a single list of simulated series into a dataframe with identifiers and timepoints
# timepoints are not dates, but integers representing the order of the amounts in time 
def append_process(list_of_time_series, process_name):  
    # first put the data for each series in a dictionary
    # collect the dictionaries in a list
    dictionaries = []
    
    for index in range(0, len(list_of_time_series)):
        # put the next series in a dictionary and add id and timepoint keys 
        series_dict = {'id': [index]*len(list_of_time_series[index]), 
                       'timepoint': list(range(0, len(list_of_time_series[index]))), 
                       'amount': list(list_of_time_series[index]),
                       'process': [process_name]*len(list_of_time_series[index])}
        # append that dictionary to the dataframe
        new_df = pd.DataFrame(series_dict)
        
        dictionaries.append(series_dict)
        
        
    # combine dictionaries into a dataframe
    frames = []

    for series_dict in dictionaries:
        series_df = pd.DataFrame(series_dict)
        frames.append(series_df)
        
    return  pd.concat(frames)

# this function loops through a list of lists of simulated series (one list for each of the processes we are simulating)
# and combines the dataframe form returned from append_process() defined above
def combine_all_series(dict_of_series):
    list_of_frames = []
    
    for key, value in dict_of_series.items():
        frame = append_process(value, str(key))
        list_of_frames.append(frame)
        
    return pd.concat(list_of_frames)

def simulate_ts_population(standardize = True):
    
    # account age distribution simulated with random integers between 35 and 190 months
    age_dist = [random.randint(35, 190) for i in range(0, 100)]
    
    # Generate AR and MA coefficients for the models with a single lag
    # ar
    ar1 = get_coefficients(1, 100)
    # ma
    ma1 = get_coefficients(1, 100)
    # arma
    ar1_for_arma11 = get_coefficients(1, 100)
    ma1_for_arma11 = get_coefficients(1, 100)
    ar1_for_arma12 = get_coefficients(1, 100)
    ma1_for_arma21 = get_coefficients(1, 100)
    # arima
    ar1_for_arima110 = get_coefficients(1, 100)
    ar1_for_arima111 = get_coefficients(1, 100)
    ar1_for_arima112 = get_coefficients(1, 100)
    ar1_for_arima211 = get_coefficients(1, 100)
    ar1_for_arima222 = get_coefficients(1, 100)
    ma1_for_arima011 = get_coefficients(1, 100)
    ma1_for_arima111 = get_coefficients(1, 100)
    ma1_for_arima112 = get_coefficients(1, 100)
    ma1_for_arima211 = get_coefficients(1, 100)
    ma1_for_arima222 = get_coefficients(1, 100)

    # Generate AR and MA coefficients for the models with two lags
    # ar
    ar2 = get_coefficients(2, 100)
    # ma
    ma2 = get_coefficients(2, 100)
    # arma
    ar2_for_arma21 = get_coefficients(2, 100)
    ma2_for_arma12 = get_coefficients(2, 100)
    ar2_for_arma22 = get_coefficients(2, 100)
    ma2_for_arma22 = get_coefficients(2, 100)
    # arima
    ar2_for_arima211 = get_coefficients(2, 100)
    ar2_for_arima222 = get_coefficients(2, 100)
    ma2_for_arima112 = get_coefficients(2, 100)
    ma2_for_arima222 = get_coefficients(2, 100)
    
    # keep simulated series separately so that they are labeled by virtue of their list membership

    # ar series
    sim_ar1 = populate_sim_lists(age_dist=age_dist, ar_parameters=ar1)
    sim_ar2 = populate_sim_lists(age_dist=age_dist, ar_parameters=ar2)

    # ma series
    sim_ma1 = populate_sim_lists(age_dist=age_dist, ma_parameters=ma1)
    sim_ma2 = populate_sim_lists(age_dist=age_dist, ma_parameters=ma2)

    # arma series
    sim_arma11 = populate_sim_lists(age_dist=age_dist, ar_parameters=ar1_for_arma11, ma_parameters=ma1_for_arma11)
    sim_arma12 = populate_sim_lists(age_dist=age_dist, ar_parameters=ar1_for_arma12, ma_parameters=ma2_for_arma12)
    sim_arma21 = populate_sim_lists(age_dist=age_dist, ar_parameters=ar2_for_arma21, ma_parameters=ma1_for_arma21)
    sim_arma22 = populate_sim_lists(age_dist=age_dist, ar_parameters=ar2_for_arma22, ma_parameters=ma2_for_arma22)

    # arima series
    sim_arima110 = populate_sim_lists(age_dist=age_dist, 
                                      ar_parameters=ar1_for_arima110,  
                                      arima=True)
    sim_arima111 = populate_sim_lists(age_dist=age_dist, 
                                      ar_parameters=ar1_for_arima111, 
                                      ma_parameters=ma1_for_arima111,  
                                      arima=True)
    sim_arima112 = populate_sim_lists(age_dist=age_dist, 
                                      ar_parameters=ar1_for_arima112, 
                                      ma_parameters=ma2_for_arima112,  
                                      arima=True)
    sim_arima211 = populate_sim_lists(age_dist=age_dist, 
                                      ar_parameters=ar2_for_arima211, 
                                      ma_parameters=ma1_for_arima211,  
                                      arima=True)
    sim_arima222 = populate_sim_lists(age_dist=age_dist, 
                                      ar_parameters=ar2_for_arima222, 
                                      ma_parameters=ma2_for_arima222, 
                                      arima=True, 
                                      d=2)
    sim_arima011 = populate_sim_lists(age_dist=age_dist, 
                                      ma_parameters=ma1_for_arima011, 
                                      arima=True)
    
    # prepare input to combine_all_series()
    dict_of_sims = {"ar1": sim_ar1, "ar2": sim_ar2,
                   "ma1": sim_ma1, "ma2": sim_ma2,
                   "arma11": sim_arma11, "arma12": sim_arma12, 
                   "arma21": sim_arma21, "arma22": sim_arma22, 
                   "arima011": sim_arima011, "arima110": sim_arima110, 
                   "arima111": sim_arima111, "arima112": sim_arima112,
                   "arima211": sim_arima211, "arima222": sim_arima222}
    
    all_series = combine_all_series(dict_of_sims)
    
    # create unique ids
    all_series['id'] = all_series['id'].map(str) + all_series['process'].map(str)
    
    # standardize
    if standardize:
        series_groups = all_series.groupby('id')['amount', 'timepoint']
        standardized_series = pd.DataFrame()

        for key, group in series_groups:
            # get mean and standard deviation within each id
            group_mean = np.mean(group['amount']) 
            group_sd = np.std(group['amount'])

            this_sample = pd.DataFrame()

            # populated standardized df
            this_sample['id'] = group['id']
            this_sample['timepoint'] = group['timepoint']
            this_sample['amount'] = group['amount']
            this_sample['stand_amount'] = (group['amount'] - group_mean) / group_sd
            this_sample['process'] = group['process']

            standardized_series = pd.concat([standardized_series, this_sample])
         
        all_series = standardized_series
    
    return all_series