import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import utilities as ut
from scipy.stats import normaltest
import datetime as dt
import seaborn as sns
import networkx as nx

def clean_local_extremes(array):
    ''' This function is made to eliminate consecutive local extremes'''
    if any([a == 1 for a in np.diff(array)]):
        indexer = list(np.diff(array)).index(1)
        indexer = [i for i,j in enumerate(list(np.diff(array))) if j == 1]
        return np.delete(array, indexer)
    else:
        return array

def compute_local_extremes(cds_ts):
    ''' Computes the indeces of local minima and local maxima
    Arguments: cds data in an array (one dimension)

    Returns: Two 1-D arrays, with minima and local maxima
    '''
    local_min = (np.diff(np.sign(np.diff(cds_ts))) > 0).nonzero()[0] + 1
    local_max = (np.diff(np.sign(np.diff(cds_ts))) < 0).nonzero()[0] + 1

    new_min = clean_local_extremes(local_min)
    new_max = clean_local_extremes(local_max)

    return new_min, new_max

def compute_std_deviation(cds_ts, local_ext, time_param):
    ''' Computes the standard deviation of the last "time_param" days at the 
    local_ext points. If it is in the first time_param days it computes
    the standard deviation of the first time_param days.

    Arguments:
    cds_ts: 1-D np array of CDS spread time series
    local_ext: 1-D array of local minima or maxima
    time_param:integer for variance parameter corresponding to epsilon parameter

    Returns:
    List of standard deviations corresponding to local minima and maxima
    '''

    #time_param = np.round(time_param)
    std_dev = []
    for index in local_ext:
        if index < time_param:
            std_dev.append(np.std(cds_ts[:time_param], ddof = 1))
        else:
            std_dev.append(np.std(cds_ts[index - time_param + 1: index + 1], ddof = 1))

    return std_dev

def compute_epsilon_drawup(cds_ts, time_param, percent):
    ''' Calculates epsilon draw-ups for the time series of an entity
    based on the standard deviation.

    Arguments:
    cds_ts: 1-D np array of CDS spread time series
    time_param:integer for variance parameter corresponding to epsilon parameter
    arg: list of two for the up and down scale.
    '''

    local_min, local_max = compute_local_extremes(cds_ts)
    epsilon_stddev_up_relative = compute_std_deviation(\
                                cds_ts, local_min, time_param)
    epsilon_stddev_down_relative = compute_std_deviation(\
                                cds_ts, local_max, time_param)
    # Scaling
    # if arg:
    #     up_scale = arg[0]
    #     down_scale = arg[1]
    #     epsilon_stddev_up_relative = [up_scale * stddev_up \
    #                             for stddev_up in epsilon_stddev_up_relative]
    #     epsilon_stddev_down_relative = [down_scale * stddev_down \
    #                             for stddev_down in epsilon_stddev_down_relative]

    # Computation of the epsilon drawups based on the standard deviation
    # if arg:
    epsilon_drawups_relative = calibrate_epsilon_drawups(\
                        cds_ts, local_min, local_max,\
                        epsilon_stddev_up_relative,\
                        epsilon_stddev_down_relative, percent)
    # else:
    #     epsilon_drawups_relative = calibrate_epsilon_drawups(\
    #                         cds_ts, local_min, local_max,\
    #                         epsilon_stddev_up_relative,\
    #                         epsilon_stddev_down_relative)           

    return epsilon_drawups_relative

def calibrate_epsilon_drawups(cds_ts, local_min, local_max, \
    epsilon_stddev_up, epsilon_stddev_down, percent):
    ''' Calibrates the epsilon draw-ups based on input parameters

    Arguments:
    cds_ts: 1d numpy array of time series of spreads
    epsilon_stddev_up: list of epsilon draw-ups at local minima
    epsilon_stddev_down: list of epsilon draw-downs at local maxima
    local_min: list of indices of local minima
    local_max: list of indices of local maxima
    min_max_order_diff: difference parameter for ordering local minima and maxima

    Returns:
    epsilon_drawups: 1d numpy array of epsilon draw-ups

    '''
    # Needs to start with a minima
    if local_min[0] < local_max[0]:
        # Needs to end with a max
        if local_min[-1] > local_max[-1]:
            local_min = local_min[0:-1]
            epsilon_stddev_up = epsilon_stddev_up[0:-1]
            #local_max = local_max[0:-2]
            #epsilon_stddev_down = epsilon_stddev_down[0:-2]
    else:
        local_max = local_max[1:]
        epsilon_stddev_down = epsilon_stddev_down[1:]
        # Needs to end with a max
        if local_min[-1] > local_max[-1]:
            local_min = local_min[0:-1]
            epsilon_stddev_up = epsilon_stddev_up[0:-1]
            #local_max_index = local_max_index[0:-2]
            #epsilon_stddev_down = epsilon_stddev_down[0:-2]

    epsilon_drawups = []
    index = 0
    index_correction = 0
    # Correction for the quantity of local minima and maxima
    if(np.abs(len(local_min)-len(local_max)))>2:
        index_correction = np.abs(len(local_min)-len(local_max))

    # if arg:
    #     condition = lambda x,y,z,r,s: check_old(x,y,z,r,s)
    # else:
    #     condition = lambda x,y,z,r,s: check_new(x,y,z,r,s)


    while index < len(local_min) - index_correction - 2:
        if check_ep(cds_ts,local_min,local_max,epsilon_stddev_up,index,percent):
            epsilon_drawups.append(local_min[index])
        index += 1

    return epsilon_drawups

def check_ep(cds_ts,local_min,local_max,epsilon_stddev_up,index, percent):
    if cds_ts[local_max[index]]-cds_ts[local_min[index]] > epsilon_stddev_up[index] and \
        cds_ts[local_max[index]]-cds_ts[local_min[index]] > percent * cds_ts[local_min[index]]:
        return True
    else:
        return False

def check_old(cds_ts,local_min,local_max,epsilon_stddev_up,index):
    if cds_ts[local_max[index]]-cds_ts[local_min[index]] > epsilon_stddev_up[index]:
        return True
    else:
        return False

def check_new(cds_ts,local_min,local_max,epsilon_stddev_up,index, percent):
    if cds_ts[local_max[index]]-cds_ts[local_min[index]] > epsilon_stddev_up[index] and \
        cds_ts[local_max[index]]-cds_ts[local_min[index]] > percent * cds_ts[local_min[index]]:
        return True
    else:
        return False

def get_ep_new(cds_ts, time_param, percent):
    ''' Computing the epsilon drawups with the new way

    Arguments:
    cds_ts: SERIES, 1-DIM
    time_param: int
    '''
    try:
        first_index = [x[0] for x in enumerate(cds_ts) if type(x[1]) is str][-1]
        df_static = cds_ts[:first_index+1]
        df_dynamic = cds_ts[first_index+1:]
    except IndexError:
        df_dynamic = cds_ts

    return compute_epsilon_drawup(df_dynamic, time_param, percent)

def get_ep_old(cds_ts, time_param):
    ''' Computing the epsilon drawups with the new way

    Arguments:
    cds_ts: SERIES, 1-DIM
    time_param: int
    '''
    percent = 0
    try:
        first_index = [x[0] for x in enumerate(cds_ts) if type(x[1]) is str][-1]
        df_static = cds_ts[:first_index+1]
        df_dynamic = cds_ts[first_index+1:]
    except IndexError:
        df_dynamic = cds_ts

    return compute_epsilon_drawup(df_dynamic, time_param, percent)


def compute_table_epsilon(cds_ts, date_axis, time_param):
    ''' Computes the whole table of epsilon drawups

    Arguments:
    cds_ts: dataframe with the time series
    date_axis: array with the dates
    time_param: time paramter to compute the epsilon drawups
    '''
    first_index = []
    # for i in cds_ts.index:
    #     first_index.append(next(x[0] for x in enumerate(cds_ts.loc[i,:])\
    #                              if type(x[1]) is str))

    table_eps = pd.DataFrame(0.0, index = cds_ts.loc[0], columns = date_axis)

    for i in cds_ts.index:
        epsilon_drawups = compute_epsilon_drawup(cds_ts.loc[i,1:], time_param, percent)
        table_eps.iloc[i, epsilon_drawups] = 1.0

    return table_eps

def compute_table_epsilon_v2(cds_ts, time_param, percent):
    ''' Computes the whole table of epsilon drawups for the 
        second set of data.

        Improved version

    Arguments:
    cds_ts: dataframe with the time series
    time_param: time paramter to compute the epsilon drawups
    '''
    # Get the date where we have the last string
    first_index = [x[0] for x in enumerate(cds_ts.iloc[0,:]) if type(x[1]) is str][-1]
    df_static = cds_ts[cds_ts.columns[:first_index+1]]
    df_dynamic = cds_ts[cds_ts.columns[first_index + 1:]]
    # Convert the columns names to datetime objects
    # df = columns_to_dt(df_dynamic)
    df = df_dynamic.copy()

    table_eps = pd.DataFrame(0.0, index = cds_ts['Ticker'], columns = df.columns)

    for i in range(cds_ts.shape[0]):
        epsilon_drawups = compute_epsilon_drawup(cds_ts.iloc[i,first_index+1:], time_param, percent/100)
        table_eps.iloc[i, epsilon_drawups] = 1.0

    return table_eps            

def compute_log_returns_cds(cds_ts):
    ''' Computes the log returns of the cds spreads'''
    return np.log(cds_ts.iloc[:,1:]).diff(axis = 1).drop(1,axis = 1)

def normal_test(df):
    ''' Computes whether the rows of the dataframe are normally distributed'''
    k2, p = normaltest(df,1)
    return k2, p

def compute_delayed_ep_drawups(ep_draw_ts, delay):
    ''' Computes the table for the lag in the contagion of the epsilon drawups.

    Arguments:
    ep_draw_ts: DataFrame with the epsilon drawups, rows are companies and columns are dates
    delay: integer number 
    
    '''
    aux = ep_draw_ts.copy()
    if type(delay) is int:
        for c in range(len(aux.columns)-delay):
            lag = []
            if any(aux.iloc[:,c]):
                for i in range(delay):
                    lag.extend([i for i,j in enumerate(aux.iloc[:,c+i+1]) if j==1])

            aux.iloc[list(set(lag)),c] = [max(i,0.5) for i in aux.iloc[list(set(lag)),c]]
        return aux
    else:
        print('Lag introduced is not an integer.')
        return aux

def compute_save_delays_epsilon(ep, time_param, delay, percent):
    ''' Computes the delayed epsilon drawups for
        different delays and save the file.

    Arguments:
    ep: table of raw epsilon drawups
    time_param: time used to compute the epsilon drawups
    delay: Either list or integer containing 
    '''
    if percent > 0:
        path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups_new'
        directory = 'std_' + str(time_param) + '_long_' + str(percent)
    else:
        path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups_new'
        directory = 'stddev_' + str(time_param)

    ut.epsilon_to_csv(ep, 'raw_ep_drawups' + str(time_param) + '.csv' , directory)
    if type(delay) == list:
        for d in delay:
            if type(d) == int:
                ep_delayed = compute_delayed_ep_drawups(ep, d)
                print('Computed the delay with time: ' + str(time_param) + ' and delay: ' + str(d))
                ut.epsilon_to_csv(ep_delayed, 'ep_drawups_' + str(time_param) + '_delay_' + str(d) + '.csv' , directory)
            else:
                print('ValueError: Element in delay is not an integer. Please introduce accepted values.')
    elif type(delay) == int:
        ep_delayed = compute_delayed_ep_drawups(ep, delay)
        ut.epsilon_to_csv(ep_delayed, 'ep_drawups_' + str(time_param) + '_delay_' + str(delay) + '.csv', directory)
    else:
        print('ValueError: Delay is not a list nor an integer. Please introduce accepted values.')

def compute_save_all_epsilon(cds, time_param, delay, percent):
    ''' Computes the epsilon drawups for all the time parameters introduced,
        the delayed epsilon drawups for each of them and then save them in the
        correspondent file and directory.

    Arguments:
    cds: dataframe with the cds time series
    time_param: list or integer with the time parameter
    delay: list or integer for the delay
    '''
    if type(time_param) == list:
        for t in time_param:
            ep = compute_table_epsilon_v2(cds, t, percent)
            compute_save_delays_epsilon(ep, t, delay, percent)
    elif type(time_param) == int:
        ep = compute_table_epsilon_v2(cds, time_param, percent)
        compute_save_delays_epsilon(ep, time_param, delay, percent)
    else:
        print('ValueError: Delay is not a list nor a integer. Please introduce accepted values.')

def compute_save_delays_epsilon_filtered(ep, time_param, delay, percent, absolute, filter_market):
    ''' Computes the delayed epsilon drawups for
        different delays and save the file.

    Arguments:
    ep: table of raw epsilon drawups
    time_param: time used to compute the epsilon drawups
    delay: Either list or integer containing 
    '''
    if percent > 0:
        path = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\epsilon'
        if filter_market:
            directory = 'std_' + str(time_param) + '_abs_' + str(absolute) + '_long_' + str(percent)
        else:
            directory = 'std_' + str(time_param) + '_abs_' + str(absolute) + '_long_' + str(percent) + '_notfiltered'

        if not os.path.isdir(os.path.join(path, directory)):
            os.makedirs(os.path.join(path, directory))
    else:
        path = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\epsilon'
        if filter_market:
            directory = 'std_' + str(time_param) + '_abs_' + str(absolute)
        else:
            directory = 'std_' + str(time_param) + '_abs_' + str(absolute) + '_notfiltered'
        if not os.path.isdir(os.path.join(path, directory)):
            os.makedirs(os.path.join(path, directory))

    ep.to_csv(os.path.join(path,directory,'raw_ep_drawups' + str(time_param) + '.csv' ))
    if type(delay) == list:
        for d in delay:
            if type(d) == int:
                ep_delayed = compute_delayed_ep_drawups(ep, d)
                print('Computed the delay with time: ' + str(time_param) + ' and delay: ' + str(d))
                ep_delayed.to_csv(os.path.join(path,directory,'ep_drawups_' + str(time_param) + '_delay_' + str(d) + '.csv'))
            else:
                print('ValueError: Element in delay is not an integer. Please introduce accepted values.')
    elif type(delay) == int:
        ep_delayed = compute_delayed_ep_drawups(ep, delay)
        print('Computed the delay with time: ' + str(time_param) + ' and delay: ' + str(delay))
        ep_delayed.to_csv(os.path.join(path,directory,'ep_drawups_' + str(time_param) + '_delay_' + str(delay) + '.csv'))
    else:
        print('ValueError: Delay is not a list nor an integer. Please introduce accepted values.')


def compute_save_all_epsilon_filtered(cds, time_param, delay, percent, absolute, filter_market):
    ''' Computes the epsilon drawups for all the time parameters introduced,
        the delayed epsilon drawups for each of them and then save them in the
        correspondent file and directory.

    Arguments:
    cds: dataframe with the cds time series
    time_param: list or integer with the time parameter
    delay: list or integer for the delay
    '''
    if type(delay) == int:
        delay = [delay]
    data_sumit = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered_data_original'
    market_filename = "market_data.npy"
    market_ts = np.load(os.path.join(data_sumit, market_filename))
    market_ts[-8] = (market_ts[-9] + market_ts[-7])/2
    if type(time_param) == list:
        for t in time_param:
            if filter_market:
                market_drawup = compute_epsilon_drawup_filtered(market_ts, t, percent,absolute)
                print(market_drawup)
                for d in delay:
                    ep = compute_table_epsilon_filtered(cds, t, percent, market_drawup, d, absolute)
                    compute_save_delays_epsilon_filtered(ep, t, d, percent, absolute, filter_market)
            else:
                ep = compute_table_epsilon_v2(cds, t, percent)
                for d in delay:
                    compute_save_delays_epsilon_filtered(ep, t, d, percent, absolute, filter_market)
    elif type(time_param) == int:
        if filter_market:
            market_drawup = compute_epsilon_drawup_filtered(market_ts, time_param, percent,absolute)
            for d in delay:
                ep = compute_table_epsilon_filtered(cds, time_param, percent, market_drawup, d, absolute)
                compute_save_delays_epsilon_filtered(ep, time_param, d, percent, absolute, filter_market)
        else:
            ep = compute_table_epsilon_v2(cds, time_param, percent)
            for d in delay:
                compute_save_delays_epsilon_filtered(ep, time_param, d, percent, absolute, filter_market)
    else:
        print('ValueError: Delay is not a list nor a integer. Please introduce accepted values.')
 
def compute_table_epsilon_filtered(cds_ts, time_param, percent, market_drawup, d, absolute):
    ''' Computes the whole table of epsilon drawups for the 
        second set of data.

        Improved version

    Arguments:
    cds_ts: dataframe with the time series
    time_param: time paramter to compute the epsilon drawups
    '''
    # Get the date where we have the last string
    first_index = [x[0] for x in enumerate(cds_ts.iloc[0,:]) if type(x[1]) is str][-1]
    df_static = cds_ts[cds_ts.columns[:first_index+1]]
    df_dynamic = cds_ts[cds_ts.columns[first_index + 1:]]
    # Convert the columns names to datetime objects
    # df = columns_to_dt(df_dynamic)
    df = df_dynamic.copy()

    table_eps = pd.DataFrame(0.0, index = cds_ts['Ticker'], columns = df.columns)

    for i in range(cds_ts.shape[0]):
        epsilon_drawups = compute_epsilon_drawup_filtered(cds_ts.iloc[i,first_index+1:], time_param, percent/100, absolute)
        epsilon_drawups = filter_market_data(epsilon_drawups, market_drawup, d)
        if len(epsilon_drawups) > 0:
            table_eps.iloc[i, epsilon_drawups] = 1.0

    return table_eps

def compute_epsilon_drawup_filtered(cds_ts, time_param, percent, absolute):
    ''' Calculates epsilon draw-ups for the time series of an entity
    based on the standard deviation.

    Arguments:
    cds_ts: 1-D np array of CDS spread time series
    time_param:integer for variance parameter corresponding to epsilon parameter
    arg: list of two for the up and down scale.
    '''

    local_min, local_max = compute_local_extremes(cds_ts)
    epsilon_stddev_up_relative = compute_std_deviation(\
                                cds_ts, local_min, time_param)

    # Computation of the epsilon drawups based on the standard deviation
    # if arg:
    epsilon_drawups_relative = calibrate_epsilon_drawups_absolute(\
                        cds_ts, local_min, local_max,\
                        epsilon_stddev_up_relative, percent, absolute)

    return epsilon_drawups_relative

def calibrate_epsilon_drawups_absolute(cds_ts, local_min, local_max, \
    epsilon_stddev_up, percent, absolute):
    ''' Calibrates the epsilon draw-ups based on input parameters

    Arguments:
    cds_ts: 1d numpy array of time series of spreads
    epsilon_stddev_up: list of epsilon draw-ups at local minima
    epsilon_stddev_down: list of epsilon draw-downs at local maxima
    local_min: list of indices of local minima
    local_max: list of indices of local maxima
    min_max_order_diff: difference parameter for ordering local minima and maxima

    Returns:
    epsilon_drawups: 1d numpy array of epsilon draw-ups

    '''
    # Needs to start with a minima
    if local_min[0] < local_max[0]:
        # Needs to end with a max
        if local_min[-1] > local_max[-1]:
            local_min = local_min[0:-1]
            epsilon_stddev_up = epsilon_stddev_up[0:-1]
            #local_max = local_max[0:-2]
            #epsilon_stddev_down = epsilon_stddev_down[0:-2]
    else:
        local_max = local_max[1:]
        # Needs to end with a max
        if local_min[-1] > local_max[-1]:
            local_min = local_min[0:-1]
            epsilon_stddev_up = epsilon_stddev_up[0:-1]
            #local_max_index = local_max_index[0:-2]
            #epsilon_stddev_down = epsilon_stddev_down[0:-2]

    epsilon_drawups = []
    index = 0
    index_correction = 0
    # Correction for the quantity of local minima and maxima
    if(np.abs(len(local_min)-len(local_max)))>2:
        index_correction = np.abs(len(local_min)-len(local_max))

    # if arg:
    #     condition = lambda x,y,z,r,s: check_old(x,y,z,r,s)
    # else:
    #     condition = lambda x,y,z,r,s: check_new(x,y,z,r,s)


    while index < len(local_min) - index_correction - 2:
        if check_ep_absolute(cds_ts,local_min,local_max,epsilon_stddev_up,index,percent, absolute):
            epsilon_drawups.append(local_min[index])
        index += 1

    return epsilon_drawups

def check_ep_absolute(cds_ts,local_min,local_max,epsilon_stddev_up,index, percent, absolute):
    if cds_ts[local_max[index]]-cds_ts[local_min[index]] > epsilon_stddev_up[index] and \
        cds_ts[local_max[index]]-cds_ts[local_min[index]] > percent * cds_ts[local_min[index]]\
        and cds_ts[local_max[index]]-cds_ts[local_min[index]] > absolute:
        return True
    else:
        return False

def filter_market_data(ep, market, delay):
    total = market[:]
    for d in range(1,delay+1):
        total.extend([a+d for a in ep])
    total = sorted(list(set(total)))
    filtered = [e for e in ep if e not in total]
    return filtered

def epsilon_filtered(percent, absolute, filter_market):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Russian_processed\data_5Y'
    cds = pd.read_excel(os.path.join(path,'entities_data_fixed.xlsx'))
    first_index = [x[0] for x in enumerate(cds.iloc[0,:]) if type(x[1]) is str][-1]

    df_static = cds[cds.columns[:first_index+1]].copy()
    df_dynamic = cds[cds.columns[first_index + 1:]].copy()

    initial = '01-MAY-14'
    final = '31-MAR-15'
    data = df_dynamic.iloc[:,list(df_dynamic.columns).index(initial):(list(df_dynamic.columns).index(final)+1)]

    cds_ts = pd.concat([df_static,data],axis = 1)

    compute_save_all_epsilon_filtered(cds_ts, [10, 15, 20], [1, 2, 3], percent, absolute, filter_market)

def epsilon_not_filter(percent,absolute):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Russian_processed\data_5Y'
    cds = pd.read_excel(os.path.join(path,'entities_data_fixed.xlsx'))
    first_index = [x[0] for x in enumerate(cds.iloc[0,:]) if type(x[1]) is str][-1]

    df_static = cds[cds.columns[:first_index+1]].copy()
    df_dynamic = cds[cds.columns[first_index + 1:]].copy()

    initial = '01-MAY-14'
    final = '31-MAR-15'
    data = df_dynamic.iloc[:,list(df_dynamic.columns).index(initial):(list(df_dynamic.columns).index(final)+1)]

    cds_ts = pd.concat([df_static,data],axis = 1)

    compute_save_all_epsilon_filtered(cds_ts, [10, 15, 20], [1, 2, 3], percent, absolute)



def epsilon(percent):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Russian_processed\data_5Y'
    cds = pd.read_excel(os.path.join(path,'entities_data_fixed.xlsx'))
    compute_save_all_epsilon(cds, [10, 15, 20], [1, 2, 3], percent)


def hist_jumps(array):
    jumps = np.abs(np.diff(array))
    h = list(jumps/array[:-1])
    plt.hist(h, bins = 100)
    plt.show()
    return h

def quantile_jumps(array, quant):
    h = hist_jumps(array)
    return pd.Series(h).quantile(quant)


# def generate_all_epsilon():
#     path_cds = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Sliced_cds'
#     diff_win = os.listdir(path_cds)
#     path_ep = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Sliced_ep_draw'
#     diff_ep = os.listdir(path_ep)
#     ep_to_compute = [name for name in diff_win if name not in diff_ep]

#     time_param = [10, 15, 20]
#     delay = [1, 2, 3]

#     for name in ep_to_compute:
#         directory = os.path.join(path_cds, name)
#         df = pd.read_csv(os.path.join(directory, 'cds_' + name + '.csv'))
#         compute_save_all_epsilon(df, time_param, delay, name)


#     for name in os.listdir(path_cds):
#         compute_save_all_epsilon(df, time_param, delay, name)


# Functions to split the data in different windows

def date_from_str(string):
    dic = {'JAN':1,'jan':1,'FEB':2,'feb':2,'MAR':3,'mar':3,'APR':4,'apr':4,'MAY':5,'may':5,'JUN':6,'jun':6,
            'JUL':7,'jul':7,'AUG':8,'aug':8,'SEP':9,'sep':9,'OCT':10,'oct':10,'NOV':11,'nov':11,'DEC':12,'dec':12}
    date_list = string.split('-')
    day = int(date_list[0])
    month = dic[date_list[1]]
    year = 2000 + int(date_list[2])
    return dt.date(year,month,day)

def str_from_date(date):
    dic_inv = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
            7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}
    year = str(date.year - 2000)
    month = dic_inv[date.month]
    day = str(date.day)
    return '-'.join([day, month, year])

def dic_to_days(**kwargs):
    converter = {'year':365, 'years':365, 'month':30,
                'months':30, 'day':1, 'days':1}
    # Now we get the total amount of days for the window we want
    tot = 0
    for k,v in kwargs.items():
        tot += v * converter[k]
    window = dt.timedelta(days = tot)

def columns_to_dt(df_ori):
    df = df_ori.copy()
    aux = pd.DataFrame(df.columns)
    temp = aux[0].apply(lambda x: date_from_str(x))
    df.columns = temp.values

    return df

def generate_all_epsilon(window, move, *arg):

    if type(window) is int:
        window = dt.timedelta(days = window)
    if type(move) is int:
        move = dt.timedelta(days = move)

    path_epsilon = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups_new'
    if arg:
        directories = [name for name in os.listdir(path_epsilon) if name.endswith('long_' + str(arg[0]))]
    else:
        directories = [name for name in os.listdir(path_epsilon) if 'long' in name and 'long_' not in name]
    for directory in [os.path.join(path_epsilon, d) for d in directories]:
        files = os.listdir(directory)
        for filename in files:
            df = pd.read_csv(os.path.join(directory, filename))
            if arg:
                split_data(df, window, move, filename, arg[0])
            else:
                split_data(df, window, move, filename, 0)



def split_data(df_ori, window, move, *arg):
    ''' Saves the data in different slices of length window and
        difference between them of move

    Arguments: (Need to be improved)
    df_ori: DataFrame
    Window: datetime.timedelta in DAYS
    move: datetime.timedelta in DAYS
    arg: name of the file in case we are working with epsilon draw ups
    '''
    if not arg:
        path_to_save = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Sliced_cds\Slice_0\Slice_'\
            + str(window.days) + '_' + str(move.days)
    else:
        path_to_save = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Sliced_ep_draw_new\Slice_'\
         + str(arg[1]) + r'\Slice_' + str(window.days) + '_' + str(move.days)
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        original_name = arg[0]


    # Get the date where we have the last string
    first_index = [x[0] for x in enumerate(df_ori.iloc[0,:]) if type(x[1]) is str][-1]
    
    df_static = df_ori[df_ori.columns[:first_index+1]].copy()
    df_dynamic = df_ori[df_ori.columns[first_index + 1:]].copy()

    if any(df_static.iloc[:,0]=='GAZPRU-Gneft'):
        index_neft = list(df_static.iloc[:,0]).index('GAZPRU-Gneft')
        df_static.iloc[index_neft,0] = 'GAZPRU.Gneft'

    # Convert the columns names to datetime objects
    df = columns_to_dt(df_dynamic)
    # Get the final day
    Last_Day = df.columns[-1]

    # First day with a quote
    initial_day = df.columns[first_index + 1]
    # print(type(initial_day))
    final_day = initial_day + window

    # Get the first columns with data and the days chosen
    
    while final_day < Last_Day:
        df_aux = df_static.join(df[pd.bdate_range(initial_day,final_day)])
        directory = os.path.join(path_to_save, str(initial_day) + '_to_' + str(final_day))
        if arg:
               filename = str(initial_day) + '_to_' + str(final_day) + '_' + original_name
        else:
            filename = 'cds_' + str(initial_day) + '_to_' + str(final_day) + '.csv'
        if os.path.isdir(directory):
            df_aux.to_csv(os.path.join(directory, filename), index = False)
        else:
            os.makedirs(directory)
            df_aux.to_csv(os.path.join(directory, filename), index = False)
        initial_day += move
        final_day += move

    df_aux = df_static.join(df[pd.bdate_range(initial_day,Last_Day)])
    directory = os.path.join(path_to_save, str(initial_day) + '_to_' + str(Last_Day))
    if arg:
           filename = str(initial_day) + '_to_' + str(Last_Day) + '_' + original_name
    else:
        filename = 'cds_' + str(initial_day) + '_to_' + str(Last_Day) + '.csv'
    if os.path.isdir(directory):
        df_aux.to_csv(os.path.join(directory, filename), index = False)
    else:
        os.makedirs(directory)
        df_aux.to_csv(os.path.join(directory, filename), index = False)

def get_original_dates():
    initial = '1-MAY-14'
    final = '31-MAR-15'
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups_new'
    path_to_save = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Sliced_ep_draw_new\original_dates'
    dirs = [f for f in os.listdir(path) if f.endswith('long') or f.endswith('long_5')]
    for d in dirs:
        if not os.path.isdir(os.path.join(path_to_save, d)):
            os.makedirs(os.path.join(path_to_save, d))
        files = [f for f in os.listdir(os.path.join(path,d))]
        for file in files:
            ep = pd.read_csv(os.path.join(path,d,file), index_col = 0)
            df = split_by_dates(ep, initial, final)
            df.to_csv(os.path.join(path_to_save,d,file))



def split_by_dates(df, initial, final):
    initial = date_from_str(initial)
    final = date_from_str(final)
    df_copy = columns_to_dt(df)
    dfinal = df_copy[pd.bdate_range(initial, final)].copy()
    return dfinal


def delete_end(string, ch):
    try:
        last_index = [x[0] for x in enumerate(string) if x[1] == ch][-1]
        return string[:last_index]
    except IndexError:
        return string


def difference_old_new():
    path_old = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups'
    path_new = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups_new'
    dirs = sorted([file for file in os.listdir(path_new) if 'long_' in file])
    total = []
    for directory in dirs:
        files = sorted([file for file in os.listdir(os.path.join(path_new,directory)) if 'raw_' in file])
        for filename in files:
            df_new = pd.read_csv(os.path.join(path_new,directory,filename))
            df_old = pd.read_csv(os.path.join(path_old, delete_end(directory, '_'),filename))
            numbers = [int(s) for s in directory.split('_') if s.isdigit()]
            if len(numbers) == 2:
                time_para = str(numbers[0])
                percent = str(numbers[1])
            else:
                time_para = str(numbers[0])
                percent = str(7.5)
            results_5 = pd.DataFrame(0.0, index = range(19),
                columns = ['Ticker','old_' + time_para + '_' + percent,'new_' + time_para + '_' + percent,'diff_%_' + time_para + '_' + percent])
            c = results_5.columns.tolist()
            results_5['Ticker'] = df_old['Ticker']
            results_5[c[3]] = ((df_old.sum(1)-df_new.sum(1))/df_new.sum(1)*100)
            # results_5[c[3]] = results_5[c[3]].apply(round)/100
            results_5[c[1]] = (df_old.sum(1)).astype(int)
            results_5[c[2]] = (df_new.sum(1)).astype(int)
            results_5.loc[19, 'Ticker'] = 'Total'
            results_5.loc[19,c[1]] = results_5[c[1]].sum()
            results_5.loc[19,c[2]] = results_5[c[2]].sum()
            results_5.loc[19,c[3]] = (results_5.loc[19,c[1]]-results_5.loc[19,c[2]])/results_5.loc[19,c[1]]*100
            results_5[c[1]] = results_5[c[1]].astype(int)
            results_5[c[2]] = results_5[c[2]].astype(int)
            results_5[c[3]] = (results_5[c[3]]*100).apply(round)/100
            print(directory)
            print(filename)
            print(results_5)
            total.append(results_5)
            results_5.to_excel(os.path.join(path_new, 'differences', 'diff_' + time_para + '_' + percent + '.xlsx'))
    total = pd.concat(total, axis = 1)
    total.to_excel(os.path.join(path_new,'differences', 'total.xlsx'))
    print(total)



### HEATMAP

def compute_corr_ep(ep):
    ''' Computes a heatmap of the epsilon drawups

    Arguments:
    ep: DataFrame with 0s and 1s
    '''
    first_index = [x[0] for x in enumerate(ep.iloc[0,:]) if type(x[1]) is str][-1]
    df = ep[ep.columns[first_index+1:]]
    df.set_index(ep[ep.columns[first_index]], inplace = True)
    df2 = df.transpose()

    num_count = df.shape[0]
    total = df.sum(1).astype(int)

    heat_map = pd.DataFrame(data = np.zeros((num_count,num_count)),
                        index = df.index, columns = df.index)
    # print(heat_map)
    for i in df.index:
        ep_drawups = df2[df2[i]==1.0][i].index
        for j in df.index:
            if i == j:
                continue
            else:
                heat_map.loc[i,j] = df[ep_drawups].loc[j,:].sum()/total[i]
                # Alternative:
                #heat_map[i,j] = df2[df2[j].index.isin(ep_drawups)][j].sum()/total[i]

    return heat_map

def order_heat(hm, *arg):
    if arg:
        first = [a for a in arg[0] if a in hm.index]
    else:
        first = ['RUSSIA', 'RUSRAI', 'GAZPRU', 'GAZPRU.Gneft','CITMOS', 'SBERBANK',
        'LUKOIL','AKT']

    rest = [r for r in hm.index if r not in first]
    tot = []
    tot.extend(first)
    tot.extend(rest)
    return hm[tot].transpose()[tot].transpose()



def compute_delay_heatmap(ep, delay, *arg):
    ''' Computes the heatmap of the epsilon drawups using delay.

    Arguments:
    ep: DataFrame with epsilon drawups
    delay: Integer for the delay in the epsilon drawups
    '''

    try:
        first_index = [x[0] for x in enumerate(ep.iloc[0,:]) if type(x[1]) is str][-1]
        df = ep[ep.columns[first_index+1:]].copy()
        df.set_index(ep[ep.columns[first_index]], inplace = True)
    except IndexError:
        df = ep.copy()

    df2 = df.transpose()

    num_count = df.shape[0]
    total = df.sum(1).astype(int)

    aux = np.zeros((delay,df.shape[1]))

    heat_map = pd.DataFrame(data = np.zeros((num_count,num_count)),
                        index = df.index, columns = df.index)
    if arg:
        if type(arg[0]) is list:
            heat_map = order_heat(heat_map, arg[0])
            first = [a for a in arg[0] if a in df.index]
        else:
            heat_map = order_heat(heat_map)


    for i in df.index:
        indeces_delay = [set(np.where(np.array(df2[i]) == 1)[0])]
        for k in range(1,delay+1):
            a = np.zeros(df.shape[1])
            a[k:] = np.array(df2[i])[:-k]
            indeces_delay.append(set(np.where(a==1)[0]))

        for j in df.index:
            count = 0
            if i == j:
                continue
            else:
                indeces_j = set(np.where(np.array(df2[j]) == 1)[0])
                for index in indeces_delay:
                    count += len(indeces_j.intersection(index))
                heat_map.loc[i,j] = count/total[i]
                # Alternative:
                #heat_map[i,j] = df2[df2[j].index.isin(ep_drawups)][j].sum()/total[i]

    return heat_map



''' In the next functions, time_para is an integer
    and percent is an integer or a float.'''

def heatmap_percent(time_para, percent):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups_new'
    directory = 'std_' + str(time_para) + '_long_' + str(percent)
    filename = r'raw_ep_drawups' + str(time_para) + '.csv'
    ep = pd.read_csv(os.path.join(path,directory,filename))
    heatmap = compute_corr_ep(ep)
    return heatmap

def save_heatmap(time_para, percent):
    heat = heatmap_percent(time_para, percent)
    ax = sns.heatmap(heat, cmap="YlGnBu")
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups_new\Heatmaps'
    plt.savefig(os.path.join(path, r'heatmap_' + str(time_para) + '_' + str(percent).replace('.','-')))
    plt.close('all')

def plotheat(time_para, percent):
    heat = heatmap_percent(time_para, percent)
    ax = sns.heatmap(heat, cmap="YlGnBu")
    plt.show()
    plt.close('all')

def save_all_heat(percent, *arg):
    ''' This function computes all the heatmas of all the periods.

    Arguments:
    percent: 5, 7.5 or 10 Usually 5.
    arg: 1 2 or 3 for the delay. Now we only work with 1.
    '''
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Sliced_ep_draw_new'
    directory = 'Slice_' + str(percent)
    if arg:
        path_heat = os.path.join(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Heatmaps_delay',\
                        r'Delay_' + str(arg[0]))
        if not os.path.isdir(path_heat):
            os.makedirs(path_heat)
    else:
        path_heat = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Heatmaps'
    dirs_Slice = os.listdir(os.path.join(path,directory))
    for d_slice in dirs_Slice:
        if not os.path.isdir(os.path.join(path_heat, directory, d_slice)):
            os.makedirs(os.path.join(path_heat, directory, d_slice))
        dirs_date = os.listdir(os.path.join(path,directory,d_slice))
        for d_date in dirs_date:
            if not os.path.isdir(os.path.join(path_heat, directory, d_slice,d_date)):
                os.makedirs(os.path.join(path_heat, directory, d_slice, d_date))
            files = [f for f in os.listdir(os.path.join(path,directory,d_slice, d_date)) if '_raw_' in f]
            for filename in files:
                df = pd.read_csv(os.path.join(path,directory,d_slice, d_date, filename))
                if arg:
                    if len(arg)>1:
                        heatmap = compute_delay_heatmap(df,arg[0],arg[1])
                    else:
                        heatmap = compute_delay_heatmap(df, arg[0])
                else:
                    heatmap = compute_corr_ep(df)

                heatmap.to_csv(os.path.join(path_heat,directory,d_slice, d_date,\
                         'Heatmap_' + d_date + '_' + filename.replace('.csv','')[-2:] + '.csv'))
                ax = sns.heatmap(heatmap, cmap = sns.cm.rocket_r, vmin = 0, vmax = 1)
                plt.savefig(os.path.join(path_heat,directory,d_slice, d_date,\
                         'Heatmap_' + d_date + '_' + filename.replace('.csv','')[-2:] + '.png'),
                         bbox_inches = 'tight')
                plt.close('all')


def save_heatmap_periods(time_para, delay):
    path_read = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Probabilities'
    path_heat = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Heatmap'
    folder = 'time_' + str(time_para) + '_delay_' + str(delay)
    files = [f for f in os.listdir(os.path.join(path_read,folder))\
                if '_prob_matrix.csv' in f]
    if not os.path.isdir(os.path.join(path_heat, folder)):
    	os.makedirs(os.path.join(path_heat,folder))
    for file in files:
        heat = pd.read_csv(os.path.join(path_read,folder,file),index_col = 0)
        heat = order_heat(heat)
        ax = sns.heatmap(heat, cmap = sns.cm.rocket_r, vmin = 0, vmax = 1)
        plt.savefig(os.path.join(path_heat, folder, file.replace('prob_matrix.csv','heatmap.png')),
                    bbox_inches = 'tight')
        plt.close('all')


def comparisons_heatmap(percent, date1, date2):
    ''' Computes the difference in percent between the 
        heatmaps in the different dates.

    Arguments:
    percent: integer o float: 5, 7.5 or 10
    date1: string as follows: "YYYY-MM-DD"
    which is the initial date of the period to be studied
    INCLUDE THE ZEROES
    '''
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Heatmaps\Slice_' + str(percent)
    path_comp = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Heatmaps\Slice_' + str(percent) + r'\Comparisons'
    dirs = [d for d in os.listdir(path) if d.startswith('Slice_')]
    for d_slice in dirs:
        first_dir = [d1 for d1 in os.listdir(os.path.join(path,d_slice)) if d1.startswith(date1)][0]
        second_dir = [d2 for d2 in os.listdir(os.path.join(path,d_slice)) if d2.startswith(date2)][0]

        for time_param in [10,15,20]:
            df1 = pd.read_csv(os.path.join(path, d_slice, first_dir, 'Heatmap_' + first_dir + '_' + str(time_param) + '.csv'), index_col = 0)
            df2 = pd.read_csv(os.path.join(path, d_slice, second_dir, 'Heatmap_' + second_dir  + '_' + str(time_param) + '.csv'), index_col = 0)
            result = ((df2-df1)/df1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # print(result)
            directory = os.path.join(path_comp,first_dir + '_and_' + second_dir)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            result.to_csv(os.path.join(directory, 'comparison_' + str(time_param) + '.csv'))
            ax = sns.heatmap(result, cmap = 'YlGnBu', center = 0)
            plt.savefig(os.path.join(directory, 'Heatmap_comp_' + str(time_param) + '.png'))
            plt.close('all')

def comparisons_heatmap2(percent, date1, date2, delay, window, move):
    ''' Computes the difference in percent between the 
        heatmaps in the different dates.

    Arguments:
    percent: integer o float: 5, 7.5 or 10
    date1: string as follows: "YYYY-MM-DD"
    which is the initial date of the period to be studied
    INCLUDE THE ZEROES
    '''
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Heatmaps_delay\Delay_' + str(delay) + r'\Slice_' +  str(percent)\
            + r'\Slice_' + str(window) + '_' + str(move)
    path_comp = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Heatmaps_delay\Comparisons'

    first_dir = [d1 for d1 in os.listdir(path) if d1.startswith(date1)][0]
    second_dir = [d2 for d2 in os.listdir(path) if d2.startswith(date2)][0]
    number = len([f for f in os.listdir(path_comp) if f.startswith(first_dir + '_and_' + second_dir)])
    directory = os.path.join(path_comp,first_dir + '_and_' + second_dir + '_nr' + str(number + 1))
    if not os.path.isdir(directory):
        os.makedirs(directory)

    for time_param in [10,15,20]:
        df1 = pd.read_csv(os.path.join(path, first_dir, 'Heatmap_' + first_dir + '_' + str(time_param) + '.csv'), index_col = 0)
        df2 = pd.read_csv(os.path.join(path, second_dir, 'Heatmap_' + second_dir  + '_' + str(time_param) + '.csv'), index_col = 0)
        result = ((df2-df1)/df1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # print(result)

        result.to_csv(os.path.join(directory, 'comparison_' + str(time_param) + '.csv'))

        ax = sns.heatmap(result, cmap = 'YlGnBu', center = 0, annot=True)
        plt.show()
        # plt.savefig(os.path.join(directory, 'Heatmap_comp_' + str(time_param) + '.png'))
        # plt.close('all')

def comparisons_heat_mean(percent, date1, date2):
    ''' Computes the difference in percent between the 
        heatmaps in the different dates and saves it
        in a csv file.

    Arguments:
    percent: integer o float: 5, 7.5 or 10
    date1: string as follows: "YYYY-MM-DD"
    which is the initial date of the period to be studied
    INCLUDE THE ZEROES
    '''
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Heatmaps\Slice_' + str(percent)
    path_comp = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Heatmaps\Slice_' + str(percent) + r'\Comparisons'
    dirs = [d for d in os.listdir(path) if d.startswith('Slice_')]
    for d_slice in dirs:
        first_dir = [d1 for d1 in os.listdir(os.path.join(path,d_slice)) if d1.startswith(date1)][0]
        second_dir = [d2 for d2 in os.listdir(os.path.join(path,d_slice)) if d2.startswith(date2)][0]

        for time_param in [10,15,20]:
            df1 = pd.read_csv(os.path.join(path, d_slice, first_dir, 'Heatmap_' + first_dir + '_' + str(time_param) + '.csv'), index_col = 0)
            row_means1 = df1.mean(1)
            col_means1 = df1.mean()
            df2 = pd.read_csv(os.path.join(path, d_slice, second_dir, 'Heatmap_' + second_dir  + '_' + str(time_param) + '.csv'), index_col = 0)
            row_means2 = df2.mean(1)
            col_means2 = df2.mean()
            # print(df1.index)
            result = pd.concat([row_means1, row_means2,col_means1,col_means2], axis = 1)
            result.columns = ['Row1', 'Row2', 'Col1', 'Col2']
            # print(result)
            directory = os.path.join(path_comp,first_dir + '_and_' + second_dir)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            result.to_csv(os.path.join(directory, 'mean_comparison_' + str(time_param) + '_' + str(percent) + '.csv'))



def plot_heat_map(heat_map):
    sns.set()
    ax = sns.heatmap(heat_map, cmap="YlGnBu")
    plt.show()
    plt.close('all')



def create_net(heatmap):
    G = nx.DiGraph()
    G.add_nodes_from(list(heatmap.index))
    for i in heatmap.index:
        for j in heatmap.columns:
            G.add_edge(*(i,j, {'weight': heatmap.loc[i,j]}))
