import networkx as nx
from matplotlib import pyplot as plt
import epsilon_module as em
import numpy as np
import pandas as pd
import datetime
import os
import pyperclip
import math
import matplotlib
from plotly.offline import download_plotlyjs,  iplot, plot
import plotly.plotly as py
#import network_calibration as nc

def read_process_cds(*arg):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Russian_processed\data_5Y'
    filename = r'cds_ts_cleaned.xlsx'
    df = pd.read_excel(os.path.join(path,filename))
    # df.drop(df[df['Counterparty'] == 'ALROSA'].index, axis = 0, inplace = True)
    ind = df.columns.tolist().index('17-AUG-15')
    col = df.columns[:ind]
    df = df[col]
    c1 = df.columns[5]
    for c2 in df.columns[6:]:
        if any(math.isnan(d) for d in df[c2]):
            for i in df[c2].index:
                if math.isnan(df.loc[i,c2]):
                    df.loc[i,c2] = df.loc[i,c1]
        # if any(df[c2].astype(str)=='-'):
        #     for i in df[c2].index:
        #         if str(df.loc[i,c2]) == '-':
        #             df.loc[i,c2] = df.loc[i,c1]-0.0001
        c1 = c2
    if arg:
        df.to_excel(os.path.join(path,'entities_data_fixed.xlsx'), index = False)
    else:
        return df


def get_value_at_index(array, indeces):
    """
    Returns the value of a time series at specified indices

    Parameters
    ----------
    array: 1d array of time series
    indeces: list of index to slice time series
    """
    local_val = np.empty(len(array))
    local_val.fill(np.nan)
    for i in indeces:
        local_val[i] = array[i]
    return local_val


def plot_eps_drwups_entity(cds_ts, date_axis,*arg):

    """
    Plots the time seres of an entity together with local minima, local maxima and
    epsilon draw-up list

    Parameters
    ----------
    cds_ts: DataFrame (Series) with the CDS spread of the entity
    date_axis: dates for the x-axis
    arg:
        local_min: np array of local minimas
        local_max: np array of local maximas
        epsilon_drawups: list of epsilon draw-ups

    """

    if arg:
        local_min = arg[0]
        local_max = arg[1]
        epsilon_drawups = arg[2]
    else:
        local_min, local_max = em.compute_local_extremes(cds_ts)
        epsilon_drawups = em.compute_epsilon_drawup(cds_ts,10)

    min_val = get_value_at_index(list(cds_ts), local_min)
    max_val = get_value_at_index(list(cds_ts), local_max)
    epsilon_val = get_value_at_index(list(cds_ts), epsilon_drawups)
    #start = datetime.datetime(2014, 5, 1)
    #end = datetime.datetime(2015, 3, 31)
    #date_axis = pd.bdate_range(start, end)
    #print(len(date_axis))
    #print()
    plt.plot(date_axis, cds_ts, linewidth=1)
    plt.plot(date_axis, min_val, '*', color='green')
    plt.plot(date_axis, max_val, '^', color='red')
    plt.plot(date_axis, epsilon_val, 'o', color='black')
    plt.grid('on')
    plt.title('Modified epsilon drawup')# for ' + entity_name, fontsize=20)
    plt.ylabel('CDS spread in bps', fontsize=20)
    #plt.xlabel('Date', fontsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.legend(('CDS spread', 'Local minima', 'Local maxima',
                'Modified epsilon draw-up'), loc=2, prop={'size': 20})
    plt.show()

def plot_cds_new_old_ep(cds_ts,date_axis,*arg):

    """
    Plots the time seres of an entity together with local minima, local maxima and
    epsilon draw-up list

    Parameters
    ----------
    cds_ts: DataFrame (Series) with the CDS spread of the entity
    date_axis: dates for the x-axis
    arg:
        local_min: np array of local minimas
        local_max: np array of local maximas
        epsilon_drawups: list of epsilon draw-ups

    """

    date_axis = [em.date_from_str(d) for d in date_axis]
    ep_old = em.get_ep_old(cds_ts,10)
    ep_new = em.get_ep_new(cds_ts,10)
    extra = [a for a in ep_old if a not in ep_new]

    new_ep_val = get_value_at_index(list(cds_ts), ep_new)
    extra_ep_val = get_value_at_index(list(cds_ts), extra)
    #start = datetime.datetime(2014, 5, 1)
    #end = datetime.datetime(2015, 3, 31)
    #date_axis = pd.bdate_range(start, end)
    #print(len(date_axis))
    #print()
    plt.plot(date_axis, cds_ts, linewidth=1)
    # plt.plot(date_axis, min_val, '*', color='green')
    plt.plot(date_axis, new_ep_val, 'o', color='black')
    plt.plot(date_axis, extra_ep_val, 'o', color='red')
    plt.grid('on')
    if arg:
        plt.title('Modified epsilon drawup of ' + str(arg[0]))
    else:
        plt.title('Modified epsilon drawup')# for ' + entity_name, fontsize=20)
    plt.ylabel('CDS spread in bps', fontsize=20)
    #plt.xlabel('Date', fontsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.legend(('CDS spread', 'New epsilon draw-up', 'Extra epsilon draw-up'),
        loc=2, prop={'size': 20})
    plt.show()

def groups_of(array, number):
    array = list(array)
    size = len(array)
    counter = 0
    list_of_index = []
    while counter < size-number+1:
        aux = []
        for j in range(number):
            aux.append(array[counter])
            counter += 1
        list_of_index.append(aux)

    aux = []
    while counter < size:
        aux.append(array[counter])
        counter += 1
    list_of_index.append(aux)

    return list_of_index


def save_old_new_ep(cds_ts, percent):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\CDS'
    if not os.path.isdir(os.path.join(path,str(percent) + '%')):
        os.makedirs(os.path.join(path,str(percent) + '%'))
    try:
        first_index = [x[0] for x in enumerate(cds_ts.iloc[0,:]) if type(x[1]) is str][-1]
        df_static = cds_ts.iloc[:,:first_index+1].copy()
        df = em.columns_to_dt(cds_ts.iloc[:,first_index+1:])
    except IndexError:
        df = em.columns_to_dt(cds_ts)
    groups = groups_of(cds_ts['Ticker'],4)
    date_axis = df.columns
    dic = {0:[0,0],1:[0,1],2:[1,0],3:[1,1]}


    for i in range(len(groups)):
        f, axarr = plt.subplots(2, 2)
        temp = df[cds_ts['Ticker'].isin(groups[i])]
        maximum = temp.max(1).max(0)
        minimum = temp.min(1).min(0)
        for j in range(len(groups[i])):
            position = dic[j]
            a = position[0]
            b = position[1]
            aux = df[cds_ts['Ticker'] == groups[i][j]].copy()
            aux = list(aux.values[0])
    
            ep_old = em.get_ep_old(aux, 10)
            ep_new = em.get_ep_new(aux, 10, percent/100)

            extra = [a for a in ep_old if a not in ep_new]

            new_ep_val = get_value_at_index(aux, ep_new)
            extra_ep_val = get_value_at_index(aux, extra)

            # plt.subplot(4, j+1, 1)
            axarr[a,b].plot(date_axis, aux, linewidth=1, label = 'CDS spread')

            axarr[a,b].plot(date_axis, new_ep_val, 'o', color='black', markersize = 2, label = 'New epsilon draw-up')
            axarr[a,b].plot(date_axis, extra_ep_val, 'o', color='red', markersize = 2, label = 'Deleted epsilon draw-up')
            axarr[a,b].grid('on')
            axarr[a,b].set_title(str(groups[i][j]))
            axarr[a,b].set_ylim(minimum - 30, maximum + 30)
            #plt.ylabel('CDS spread in bps', fontsize=20)
            #plt.xlabel('Date', fontsize=20)
            #plt.rc('xtick', labelsize=15)
            #plt.rc('ytick', labelsize=15)
        blue_patch = mpatches.Patch(color='blue', label='CDS spread')
        red_patch = mpatches.Patch(color='red', label='Deleted ep. drawup')
        black_patch = mpatches.Patch(color='black', label='Kept ep. drawup')

        plt.figlegend(handles=[blue_patch, black_patch, red_patch])
        plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
        plt.show()
        f.savefig(os.path.join(path,str(percent) + '%','_'.join(groups[i]) + '.png'))
        plt.close('all')

def plot_selected(cds_ts,selection,time,delay, *arg):
    """
    Plots the time series of all entities in the cds time series

    Parameters
    ----------
    cds_ts: DataFrame with the data for the CDS
    date_axis: dates for the x-axis
    arg: If it is there then the epsilon drawups are also plotted
    """

    # if 'RUSSIA' in selection:
    #     selection = [a for a in selection if a!='RUSSIA']

    aux = cds_ts[cds_ts['Ticker']==selection[0]].copy()
    aux2 = cds_ts[cds_ts['Ticker'].isin(selection[1:])].copy()
    cds_ts = pd.concat([aux,aux2], axis = 0)

    initial = '1-MAY-14'
    final = '31-MAR-15'

    try:
        first_index = [x[0] for x in enumerate(cds_ts.iloc[0,:]) if type(x[1]) is str][-1]
        df_static = cds_ts.iloc[:,:first_index+1].copy()
        df = em.split_by_dates(cds_ts.iloc[:,first_index+1:], initial, final)
    except IndexError:
        df = em.split_by_dates(cds_ts, initial, final)

    start = em.date_from_str(initial)
    end = em.date_from_str(final)

    date_axis = pd.bdate_range(start, end)
    labels = []
    ep = {}
    ep_from_sov = {}
    ep_ind = {}

    for i in range(len(df.index)):
        l = cds_ts.loc[cds_ts.index[i],'Ticker']
        labels.append(l)
        ep[l] = em.compute_epsilon_drawup(df.iloc[i,:],time,0)

    val_rus = ep[selection[0]]
    for k in ep.keys():
        if k == selection[0]:
            continue
        else:
            ep_from_sov[k], ep_ind[k] = ep_coincidence(val_rus, ep[k], delay)

    b = False
    for i in range(len(df.index)):

        plt.plot(date_axis, df.iloc[i,:],linewidth=2, label = labels[i])
        if arg:
            if labels[i] == selection[0]:
                epsilon_val = get_value_at_index(list(df.iloc[i,:]), ep[labels[i]])
                plt.plot(date_axis, epsilon_val, 'o', color='black')

            else:

                ind_val = get_value_at_index(list(df.iloc[i,:]), ep_ind[labels[i]])

                from_sov_val = get_value_at_index(list(df.iloc[i,:]), ep_from_sov[labels[i]])

                if i == len(df.index)-1:
                    b=True
                if b:
                    plt.plot(date_axis, ind_val, 'o', color='black', label = 'Epsilon drawups')
                    plt.plot(date_axis, from_sov_val, 'o', color='blue', label = 'Caused by Gazprom')
                else:
                    plt.plot(date_axis, ind_val, 'o', color='black')
                    plt.plot(date_axis, from_sov_val, 'o', color='blue')

                b = False

    plt.grid(True)
    if arg:
        plt.title('CDS spreads with epsilon drawups')
    else:
        plt.title('CDS spreads')
    plt.ylabel('CDS spread in bps', fontsize=20)
    # plt.xlabel('Date', fontsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.legend()
    plt.show()

def plot_selection(selection,time,delay, *arg):
    """
    Plots the time series of all entities in the cds time series

    Parameters
    ----------
    cds_ts: DataFrame with the data for the CDS
    date_axis: dates for the x-axis
    arg: If it is there then the epsilon drawups are also plotted
    """

    # if 'RUSSIA' in selection:
    #     selection = [a for a in selection if a!='RUSSIA']

    cds_ts = pd.read_excel(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Russian_processed\data_5Y\entities_data_fixed.xlsx')

    aux = cds_ts[cds_ts['Ticker']==selection[0]].copy()
    aux2 = cds_ts[cds_ts['Ticker'].isin(selection[1:])].copy()
    cds_ts = pd.concat([aux,aux2], axis = 0)

    initial = '1-MAY-14'
    final = '31-MAR-15'

    try:
        first_index = [x[0] for x in enumerate(cds_ts.iloc[0,:]) if type(x[1]) is str][-1]
        df_static = cds_ts.iloc[:,:first_index+1].copy()
        df = em.split_by_dates(cds_ts.iloc[:,first_index+1:], initial, final)
    except IndexError:
        df = em.split_by_dates(cds_ts, initial, final)

    start = em.date_from_str(initial)
    end = em.date_from_str(final)

    date_axis = pd.bdate_range(start, end)
    labels = []
    ep = {}
    ep_from_sov = {}
    ep_ind = {}

    for i in range(len(df.index)):
        l = cds_ts.loc[cds_ts.index[i],'Ticker']
        labels.append(l)
        ep[l] = em.compute_epsilon_drawup(df.iloc[i,:],time,0)



    val_rus = ep[selection[0]]
    for k in ep.keys():
        if k == selection[0]:
            print(str(k) + ': ' +str(len(ep[k]))+' total')
            continue
        else:
            ep_from_sov[k], ep_ind[k] = ep_coincidence(val_rus, ep[k], delay)

            print(str(k) + ': ' + str(len(ep[k]))+' total and '+str(len(ep_from_sov[k])))


    for k in ep.keys():
        if k == selection[0]:
            print(str(k) + ': ' +str(ep[k])+' total')
        else:
            print(str(k) + ': ' + str(ep[k])+' total \n '+str(ep_from_sov[k]))

    b = False
    for i in range(len(df.index)):

        plt.plot(date_axis, df.iloc[i,:],linewidth=2, label = labels[i])
        if arg:
            if labels[i] == selection[0]:
                epsilon_val = get_value_at_index(list(df.iloc[i,:]), ep[labels[i]])
                plt.plot(date_axis, epsilon_val, 'o', color='black')

            else:

                ind_val = get_value_at_index(list(df.iloc[i,:]), ep_ind[labels[i]])

                from_sov_val = get_value_at_index(list(df.iloc[i,:]), ep_from_sov[labels[i]])

                if i == len(df.index)-1:
                    b=True
                if b:
                    plt.plot(date_axis, ind_val, 'o', color='black', label = 'Epsilon drawups')
                    plt.plot(date_axis, from_sov_val, 'o', color='blue', label = 'Caused by ' + selection[0])
                else:
                    plt.plot(date_axis, ind_val, 'o', color='black')
                    plt.plot(date_axis, from_sov_val, 'o', color='blue')

                b = False

    plt.grid(True)
    if arg:
        plt.title('CDS spreads with epsilon drawups')
    else:
        plt.title('CDS spreads')
    plt.ylabel('CDS spread in bps', fontsize=20)
    # plt.xlabel('Date', fontsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.legend()
    plt.show()

def ep_coincidence(val_rus, count, delay):
    intersection = []
    for c in count:
        for d in range(delay+1):
            if c-d in val_rus:
                intersection.append(c)
                break
    rest = [c for c in count if c not in intersection]
    return intersection, rest

    # for i in range(1):#len(groups)):
    #     f, axarr = plt.subplots(2, 2)
        # temp = df[cds_ts['Ticker'].isin(groups[i])]
        # maximum = temp.max(1).max(0)
        # minimum = temp.min(1).min(0)
    #     for j in range(len(groups[i])):
    #         position = dic[j]
    #         a = position[0]
    #         b = position[1]
    #         aux = df[cds_ts['Ticker'] == groups[i][j]].copy()
    #         aux = list(aux.values[0])
    
    #         ep_old = em.get_ep_old(aux, 10)
    #         ep_new = em.get_ep_new(aux, 10)

    #         extra = [a for a in ep_old if a not in ep_new]

    #         new_ep_val = get_value_at_index(aux, ep_new)
    #         extra_ep_val = get_value_at_index(aux, extra)

    #         # plt.subplot(4, j+1, 1)
    #         axarr[a,b].set_ylim(minimum - 30, maximum + 30)
    #         axarr[a,b].plot(date_axis, aux, linewidth=1)

    #         axarr[a,b].plot(date_axis, new_ep_val, 'o', color='black', markersize = 3)
    #         axarr[a,b].plot(date_axis, extra_ep_val, 'o', color='red', markersize = 3)
    #         axarr[a,b].grid('on')
    #         axarr[a,b].set_title(str(groups[i][j]))
    #         #plt.ylabel('CDS spread in bps', fontsize=20)
    #         #plt.xlabel('Date', fontsize=20)
    #         #plt.rc('xtick', labelsize=15)
    #         #plt.rc('ytick', labelsize=15)
    #     plt.legend(('CDS spread', 'New epsilon draw-up', 'Extra epsilon draw-up'),
    #        loc=2, prop={'size': 5})
    #     plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    #     plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    #     plt.show()
    #     plt.savefig(os.path.join(path,name,'_'.join(groups[i]) + '.png'))
    #     plt.close('all')
    

def plot_cds_all(cds_ts, *arg):
    """
    Plots the time series of all entities in the cds time series

    Parameters
    ----------
    cds_ts: DataFrame with the data for the CDS
    date_axis: dates for the x-axis
    arg: If it is there then the epsilon drawups are also plotted
    """
    first_index = [x[0] for x in enumerate(cds_ts.iloc[0,:]) if type(x[1]) is str][-1]
    start = em.date_from_str(cds_ts.columns[first_index + 1])
    end = em.date_from_str(cds_ts.columns[-1])
    date_axis = pd.bdate_range(start, end)
    for i in range(len(cds_ts.index)):
        plt.plot(date_axis, cds_ts.iloc[i,first_index + 1:],linewidth=1)
        if arg:
            epsilon_drawups = em.compute_epsilon_drawup(cds_ts.iloc[i,first_index + 1:],10)
            epsilon_val = get_value_at_index(list(cds_ts.iloc[i,first_index + 1:]), epsilon_drawups)
            plt.plot(date_axis, epsilon_val, 'o', color='black')
    plt.grid(True)
    if arg:
        plt.title('CDS spreads with epsilon drawups')
    else:
        plt.title('CDS spreads')
    plt.ylabel('CDS spread in bps', fontsize=20)
    # plt.xlabel('Date', fontsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.show()


def plot_epsilon_drawup_entity(entities_np, entity_name, epsilon_choice, epsilon_down_time_parameter,
                                epsilon_down_scale, minimal_epsilon_down, absolute_down,
                                epsilon_up_time_parameter, epsilon_up_scale, minimal_epsilon_up, absolute_up):
    '''
    Plots the time seres of an entity together with local minima, local maxima and
    epsilon draw-up list

    Parameters
    ----------
    entities_np: 2d numpy matrix corresponding to time series of entities
    entity_name: string, name of the entity whose time series has to be plotted
    epsilon_choice: string parameter corresponding to standard deviation or percentage epsilon
    epsilon_time_parameter: int parameter corresponding to standard deviation calculation    

    '''

    entity_index = np.where(entities_np[:, 0] == entity_name)
    entity_ts = np.ravel(entities_np[entity_index, 1:])
    #print entity_ts
    #print np.shape(entity_ts)
    #entity_ts = np.ravel(entities_np)
    print(np.shape(entity_ts))
    local_min_index, local_max_index = nc.compute_local_minmax(entity_ts)
    epsilon_drawup_list = nc.compute_epsilon_drawup(entity_ts, epsilon_choice, epsilon_down_time_parameter, epsilon_down_scale, minimal_epsilon_down, absolute_down,
                           epsilon_up_time_parameter, epsilon_up_scale, minimal_epsilon_up, absolute_up)

    print("epsilon= ", epsilon_drawup_list)
    local_min_val = get_value_at_index(entity_ts, local_min_index)
    local_max_val = get_value_at_index(entity_ts, local_max_index)
    #print local_min_index
    get_average_drawup(entity_ts,local_max_index,epsilon_drawup_list)
    epsilon_drawup_val = get_value_at_index(entity_ts, epsilon_drawup_list)
    #start = datetime.datetime(2014, 5, 1)
    #end = datetime.datetime(2015, 4, 15)

    start = datetime.datetime(2014, 5, 1)
    end = datetime.datetime(2015, 3, 31)
    date_axis = pd.bdate_range(start, end)
    print(len(date_axis))
    print()
    plt.plot(date_axis, entity_ts, linewidth=1)
    plt.plot(date_axis, local_min_val, '*', color='green')
    plt.plot(date_axis, local_max_val, '^', color='red')
    plt.plot(date_axis, epsilon_drawup_val, 'o', color='black')
    plt.grid('on')
    plt.title('Modified epsilon drawup for ' + entity_name, fontsize=20)
    plt.ylabel('CDS spread in bps', fontsize=20)
    #plt.xlabel('Date', fontsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.legend(('CDS spread', 'Local minima', 'Local maxima',
                'Modified epsilon draw-up'), loc=2, prop={'size': 20})
    plt.show()

def comparisons_prob_sumit_bayes():
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\probabilities\Comp_old_new'
    for time_param in [10,15,20]:
        for absolute in [0,10]:
            folder = 'time_'+str(time_param)+'_abs_'+str(absolute)
            if not os.path.isdir(os.path.join(path,folder)):
                os.makedirs(os.path.join(path, folder))
            for delay in [1,2,3]:
                for filtered in [True, False]:
                    if filtered:
                        filename = 'filter_' + str(time_param) + '_delay_' + str(delay) + '_abs_' + str(absolute) + '.csv'
                    else:
                        filename = 'notfilt_' + str(time_param) + '_delay_' + str(delay) + '_abs_' + str(absolute) + '.csv'
                    df = comp_sumit_bayes_old_new(time_param, delay,absolute, filtered)
                    df.to_csv(os.path.join(path,folder, filename))



def comp_sumit_bayes_old_new(time_param, delay,absolute, filtered):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\probabilities'
    folder_old = 'CountryRank'
    folder_new = 'CountryRank_new'
    directory = 'time_'+str(time_param)+'_abs_'+str(absolute)
    if filtered:
        name = 'comp_marketfilter_'+str(time_param)+'_delay_'+str(delay)+'_abs_'+str(absolute)
    else:
        name = 'comp_notfilter_'+str(time_param)+'_delay_'+str(delay)+'_abs_'+str(absolute)

    filename_old = name+'.csv'
    filename_new = name+'_new.csv'
    df_old = pd.read_csv(os.path.join(path,folder_old,directory,filename_old),index_col = 0)
    df_new = pd.read_csv(os.path.join(path,folder_new,directory,filename_new),index_col = 0)
    result = pd.concat([df_old['Sumit Old'], df_new[['Sumit New','Bayes']]], axis = 1).sort_values(by='Sumit New',ascending = False)
    return result



def write_to_excel(table, path, name_file):
    table.to_excel(os.path.join(r'C:\Users\Javier\Documents\MEGA' \
            + '\Thesis\CDS_data\Russian_processed\data', name_file))

def write_to_csv(table, name_file):
    table.to_csv(os.path.join(r'C:\Users\Javier\Documents\MEGA' \
            + '\Thesis\CDS_data\Russian_processed\data', name_file))

def epsilon_to_excel(table, name_file, *arg):
    if arg:
        directory = os.path.join(r'C:\Users\Javier\Documents\MEGA'\
            + r'\Thesis\CDS_data\epsilon_drawups',arg[0])
        if os.path.isdir(directory):
            table.to_excel(directory, name_file)
        else:
            os.makedirs(directory)
            table.to_excel(directory, name_file)

    else:
        table.to_excel(os.path.join(r'C:\Users\Javier\Documents\MEGA' \
            + r'\Thesis\CDS_data\epsilon_drawups', name_file))

def epsilon_to_csv(table, name_file, *arg):
    if arg:
        directory = os.path.join(r'C:\Users\Javier\Documents\MEGA'\
            + r'\Thesis\CDS_data\epsilon_drawups_new',arg[0])
        if os.path.isdir(directory):
            table.to_csv(os.path.join(directory, name_file))
        else:
            os.makedirs(directory)
            table.to_csv(os.path.join(directory, name_file))

    else:
        table.to_csv(os.path.join(r'C:\Users\Javier\Documents\MEGA' \
            + r'\Thesis\CDS_data\epsilon_drawups', name_file))

def matrix_as_table(df):
    dic = {}
    for i in range(len(df.index)):
        dic[df.index[i]] = i+1
    print(r'\begin{center}')
    print(r'\begin{tabular}', end = r'{|')
    for i in range(len(df.columns) + 2):
        print(r' c ', end = '|')
    print(r'}')
    print(r'\hline')
    print(r'Names & Number', end = '')
    for c in df.columns:
        print(r' & ' + str(dic[c]), end = '')
    print(r'\\')
    print(r'\hline')
    for i in df.index:
        print(str(i) + ' & ' + str(dic[i]), end = r'')
        for c in df.columns:
            print(r' & ' + str(df.loc[i,c]), end = '')
        print(r'\\')
        print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{center}')

def add_to(string, extra, **kwargs):
    if kwargs:
        return (string + extra + str(kwargs['end']))
    else:
        return (string + extra + '\n')

def matrix_as_table_rot_names(df):
    output = ''
    dic = {}
    for i in range(len(df.index)):
        dic[df.index[i]] = i+1
    # print(r'\begin{center}')
    output = add_to(output, r'\begin{center}')
    # print(r'\begin{tabular}', end = r'{|')
    output = add_to(output, r'\begin{tabular}', end = r'{|')
    for i in range(len(df.columns) + 1):
        # print(r' c ', end = '|')
        output = add_to(output, r' c ', end = '|')
    # print(r'}')
    output = add_to(output, r'}')
    # print(r'\hline')
    output = add_to(output, r'\hline')
    # Printing the names rotated
    # print(r'\raisebox{-\height }{\rotatebox[origin=l]{90}{Names}}', end = '')
    output = add_to(output, r'\raisebox{-\height }{\rotatebox[origin=l]{90}{Names}}', end = '')
    for c in df.columns:
        # print(r' & \raisebox{-\height }{\rotatebox[origin=l]{90}{' \
         # + str(c) + r'}}', end = '')
        output = add_to(output, r' & \raisebox{-\height }{\rotatebox[origin=l]{90}{' \
         + str(c) + r'}}', end = '')
    # print(r'\\')
    output = add_to(output, r'\\')
    # print(r'\hline')
    output = add_to(output, r'\hline')
    # Printing the numbers
    # print(r'Numbers', end = '')
    output = add_to(output, r'Numbers', end = '')
    for c in df.columns:
        # print(' & ' + str(dic[c]), end = '')
        output = add_to(output, ' & ' + str(dic[c]), end = '')
    # print(r'\\')
    output = add_to(output, r'\\')
    # print(r'\hline')
    output = add_to(output, r'\hline')
    for i in df.index:
        # print(str(dic[i]), end = r'')
        output = add_to(output, str(dic[i]), end = r'')
        for c in df.columns:
            # print(r' & ' + '{:g}'.format(df.loc[i,c]), end = '')
            output = add_to(output, r' & ' + '{:g}'.format(df.loc[i,c]), end = '')
        # print(r'\\')
        output = add_to(output, r'\\')
        # print(r'\hline')
        output = add_to(output, r'\hline')
    # print(r'\end{tabular}')
    output = add_to(output, r'\end{tabular}')
    # print(r'\end{center}')
    output = add_to(output, r'\end{center}')

    pyperclip.copy(output)

def prob_as_table(df, center):
    output = ''
    dic = {}
    for i in range(len(df.index)):
        dic[df.index[i]] = i+1
    # print(r'\begin{center}')
    if center:
    	output = add_to(output, r'\begin{center}')
    # print(r'\begin{tabular}', end = r'{|')
    output = add_to(output, r'\begin{tabular}', end = r'{|')
    for i in range(len(df.columns) + 2):
        # print(r' c ', end = '|')
        output = add_to(output, r' c ', end = '|')
    # print(r'}')
    output = add_to(output, r'}')
    # print(r'\hline')
    output = add_to(output, r'\hline')
    # Printing the top part:
    output = add_to(output, r'nr', end = r' & ')
    output = add_to(output, r'Company', end = r' & ')
    output = add_to(output, r'Mean', end = r' & ')
    output = add_to(output, r'Std\\')
    output = add_to(output, r'\hline')
    for i in df.index:
        output = add_to(output, str(dic[i]), end = r' & ')
        output = add_to(output, str(i), end = r'')
        for c in df.columns:
            output = add_to(output, r' & ', end = '')
            output = add_to(output, ('%.4f' % df.loc[i,c]).rstrip('0').rstrip('.'), end = '')
        output = add_to(output, r'\\')
        output = add_to(output, r'\hline')

    output = add_to(output, r'\end{tabular}')
    # print(r'\end{center}')
    if center:
    	output = add_to(output, r'\end{center}')

    pyperclip.copy(output)

def draw_network(entities, mat):
    g = nx.from_numpy_matrix(mat)
    eig_cen = nx.eigenvector_centrality_numpy(g)
    net_stats = pd.DataFrame(columns = ['Entities', 'Centrality'])
    net_stats['Entities'] = entities
    net_stats['Centrality'] = eig_cen.values()
    pos = nx.spring_layout(g)
    label_list = dict(enumerate(net_stats['Entities'].tolist()))
    nx.draw_networkx(g, labels = label_list, node_color = 'yellow', node_size = [v*2000 for v in list(eig_cen.values())],
                edge_color = 'gray', pos = pos, font_size = 18, font_color = 'black', alpha = 0.6)
    plt.axis('off')
    plt.show()

def draw_stressed_network(entities_list, rank, edge_list):
    size = np.size(entities_list)
    mst = pd.DataFrame(np.zeros((size,size)),index = entities_list,
                                            columns = entities_list)
    for e in edge_list:
        mst.loc[e[0],e[1]] = 1

    label_list = dict(enumerate(entities_list))
    reds = plt.get_cmap('Reds')
    nodes_list = []
    red_array = []
    for i in range(len(entities_list)):
        if rank[i] == 1:
            red_array.append(reds(0.99))
        else:
            red_array.append(reds(rank[i]))

        nodes_list.append(entities_list[i] + '(' + str(rank[i])[0:4] + ')')
    label_list = dict(enumerate(nodes_list))
    h = nx.from_numpy_matrix(np.matrix(mst), create_using = nx.DiGraph())
    h_aux = nx.from_numpy_matrix(np.matrix(mst))
    pos = nx.kamada_kawai_layout(h_aux)
    eigen_cen = nx.eigenvector_centrality_numpy(h_aux)
    centrality = eigen_cen.values()
    nx.draw_networkx(h, labels = label_list, edge_color = 'black',
                    node_color = red_array, node_cmap = reds,
                    node_size = [v*2000 for v in centrality], pos = pos,
                    font_size = 8)
    plt.axis('off')
    plt.show()

def make_annotations(pos, anno_text, font_size=14, font_color='rgb(10,10,10)'):
    L=len(pos)
    if len(anno_text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(dict(text=anno_text[k], 
                                x=pos[k][0], 
                                y=pos[k][1]+0.075,#this additional value is chosen by trial and error
                                xref='x1', yref='y1',
                                font=dict(color= font_color, size=font_size),
                                showarrow=False)
                          )
    return annotations

def draw_stressed_network_v2(entities_list, rank, edge_list):
    ''' Draw a network where
        node size according to centrality
        node color according to probability of default
        
        Arguments:
        entities_list: list of strings
        rank: list of floats
        edge_list: list of tuples or lists
    '''
    size = np.size(entities_list)
    mst = pd.DataFrame(np.zeros((size,size)),index = entities_list,
                                            columns = entities_list)
    # Get adjacency matrix
    for e in edge_list:
        mst.loc[e[0],e[1]] = 1
    
    # Get colors and list of nodes
    reds = plt.get_cmap('Reds')
    nodes_list = []
    red_array = []
    for i in range(len(entities_list)):
        if rank[i] == 1:
            red_array.append(reds(0.99))
        else:
            red_array.append(reds(rank[i]))

        nodes_list.append(entities_list[i] + '(' + str(rank[i])[0:4] + ')')
        
    label_list = dict(enumerate(nodes_list))
    
    colors_nodes = ['rgb(' +','.join([str(a * 225) for a in matplotlib.colors.to_rgb(r)]) + ')' for r in red_array]
    
    G = nx.from_numpy_matrix(np.matrix(mst), create_using = nx.DiGraph())
    G_aux = nx.from_numpy_matrix(np.matrix(mst))
    
    eigen_cen = nx.eigenvector_centrality_numpy(G_aux)
    centrality = list(eigen_cen.values())
    
    pos = nx.kamada_kawai_layout(G_aux)
    
    Xn=[pos[k][0] for k in range(len(pos))]
    Yn=[pos[k][1] for k in range(len(pos))]
    # Define Plotly trace for nodes, here we can change color size, etc
    
    trace_nodes = dict(type = 'scatter',
                        x = Xn,
                        y = Yn,
                        mode = 'markers',
                        marker = dict(symbol = 'dot', size = [v*100 for v in centrality], color = colors_nodes),
                        text = nodes_list)
    
    Xe=[]
    Ye=[]
    for e in G.edges():
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])

    trace_edges=dict(type='scatter',
                     mode='lines',
                     x=Xe,
                     y=Ye,
                     line=dict(width=1, color='rgb(25,25,25)'),
                     hoverinfo='none' 
                    )
    axis = dict(showline = False,
                zeroline = False,
                showgrid = False,
                showticklabels = False,
                title = ''
                )
    layout = dict(title = 'MyGraph',
                font = dict(family = 'Balto'),
                width=600,
                height=600,
                autosize=False,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(
                            l=40,
                            r=40,
                            b=85,
                            t=100,
                            pad=0,
                            ),
                hovermode='closest',
                plot_bgcolor='#efecea', #set background color
                annotations=make_annotations(pos, nodes_list),       
                )
    data = [trace_nodes,trace_edges] 
    fig = dict(data=data, layout=layout)
    
    # fig['layout'].update(annotations=make_annotations(pos, nodes_list))
    
    plot(fig)
    py.image.save_as(fig,os.path.join(r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Networks',
                'precrisis_russia.png'))
    plt.show()