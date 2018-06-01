from __future__ import division
import datetime
import network_calibration as nc
import data_preprocess as dp
import utilities as ut
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt


start_time = datetime.datetime.now()
path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\sumit_data'
filename = 'entities_data.npy'
entities_np = dp.load_entities_data(path, filename)
#ut.plot_eur_time_series(entities_np)
print(np.shape(entities_np))
[row, col] = np.shape(entities_np)
input_entities_list = entities_np[:, 0]

[row, col ] = np.shape(entities_np)
input_entities_np = np.empty([row, col+1], dtype = object)
#input_entities_np[:,0] = input_entities_list
input_entities_np = entities_np
input_entities_np = input_entities_np[:-3,:]
#print input_entities_np
start = datetime.datetime(2014, 5, 1)
end = datetime.datetime(2015, 4, 30)

date_axis = pd.bdate_range(start, end)
print (len(date_axis))
print(np.shape(input_entities_np))
#ut.plot_rus_time_series(input_entities_np)
#input_entities_np = input_entities_np[input_entities_np[:, 1] == 'E.Eur',:]
#print input_entities_np
input_entities_list = input_entities_np[:, 0]
#input_region_list = input_entities_np[:, 1]
#print input_entities_list
input_epsilon_choice = 'std_dev'  # choices are 'std_dev' or 'percentage'
market_filename = "market_data.npy"
filter_market = True
if filter_market:
    input_market_ts = dp.load_market_data(path, market_filename)
    input_market_ts[-8] = (input_market_ts[-7]+input_market_ts[-9])/2
else:
    input_market_ts = np.zeros([np.shape(input_entities_np)[0]-1])

#ut.plot_rus_time_series(input_entities_np)

#print input_market_ts
print('Done loading data:' + str(datetime.datetime.now() - start_time))
print ('----------------------------------------------------------------')

#ing_np =input_entities_np[input_entities_np[:,0] == 'ING Groep NV', :][0]
#print "ing ", ing_np
#print np.std(ing_np[2:] - ing_np[1:-1])/ np.mean(ing_np[2:] - ing_np[1:-1])

# get the coefficient of variance of returns of iTraxx
#print 'coefficient of variance', ut.get_coefficient_of_variance(input_market_ts)
# plotting time series with epsilon_draw-ups
entity_name = u'Russian Fedn'

#ut.plot_epsilon_drawup_entity(input_entities_np, entity_name, 'std_dev', 5, 0, 0, 0, 5, 1, 0.00, 10)
#nc.test_epsilon_std_dev_params(input_entities_np, entity_name, [10, 15, 20], 1)

#entity_index = np.where(input_entities_np[:, 0] == entity_name)
#entity_ts = np.ravel(input_entities_np[entity_index, 2:])
#for param in [5, 10, 20]:
#    print nc.compute_epsilon_drawup(entity_ts, 'std_dev', 5, 0, 0, 0, param, 1, 0.00, 20)




input_time_lag = [0,1,2,3]
epsilon_down_time_parameter = 20
epsilon_down_scale = 0
minimal_epsilon_down = 0
absolute_down = 0
epsilon_up_time_parameter = 20
epsilon_up_scale = 1
minimal_epsilon_up = 0
absolute_up = 10

weight_matrix,epsilon_drawup_matrix = nc.epsilon_drawup_network(
            input_entities_np, input_time_lag, input_market_ts,
            input_epsilon_choice, epsilon_down_time_parameter,
            epsilon_down_scale, minimal_epsilon_down, absolute_down,
            epsilon_up_time_parameter, epsilon_up_scale,
            minimal_epsilon_up, absolute_up)

# ut.convert_weight_matrix_to_csv(weight_matrix)


#nc.evaluate_max_choice(epsilon_drawup_matrix,input_entities_list)
#nc.calculate_network_stats(input_entities_list, weight_matrix)
#nc.test_significance_network(input_entities_np, input_time_lag, input_market_ts,
#                                          input_epsilon_choice, 5, 0, 0, 0, 5, 1, 0.00, 10)
'''for i in range(1):
    input_time_lag = range(0,i+1)
    print input_time_lag
    weight_matrix = nc.epsilon_drawup_network(input_entities_np, input_time_lag, input_market_ts,
                                          input_epsilon_choice, 5, 0, 0, 0, 5, 1, 0.00, 20)

    print np.mean(weight_matrix[0,:])
    plt.plot(weight_matrix[0,:], label = str(i))
plt.legend(loc=2)
plt.show()'''
    #print weight_matrix[0, :]

#print weight_matrix

#plt.imshow(weight_matrix, cmap='hot', interpolation='nearest')
#plt.show()

#plotting network based on weight matrix
#ut.draw_network_alternative(weight_matrix, input_entities_np[:, 0], input_entities_np[:, 1], True)

#network_stats_np = nc.calculate_network_stats(input_entities_np[:, 0], weight_matrix)
#print (pd.DataFrame(data=network_stats_np, columns=['Entities', 'Centrality']))

#print '----------------------------------------------------------------'
print('Done calibrating network weight matrix:' + str(datetime.datetime.now() - start_time))
#print '----------------------------------------------------------------'
print(weight_matrix)
weight_df = pd.DataFrame(weight_matrix, index = input_entities_list, columns = input_entities_list)
aux = weight_df.loc['Russian Fedn',:]
print(weight_df.loc['Russian Fedn',:])
# print(pd.DataFrame(weight_matrix))

# choices are self-calibrating or name of the entity to stress
#input_stressed_list = ['self-calibrating']
input_stressed_list = ['Russian Fedn']
# float parameter between 0 and 1 corresponding to initial level of stress for stressed entity
input_initial_stress = [1]

country_rank_df = nc.compute_countryrank(weight_matrix, input_entities_list, input_stressed_list,
                                         input_initial_stress, draw_network=False)
country_rank_df.to_csv(os.path.join(path,'ctry.csv'))

aux2 = country_rank_df.set_index('Entity')

dic_count = {'RUSSIA':'Russian Fedn',
'AKT':'Oil Transporting Jt Stk Co Transneft',
'BKECON':'Vnesheconombank',
'BOM':'Bk of Moscow',
'CITMOS':'City Moscow',
'GAZPRU':'JSC GAZPROM',
'GAZPRU.Gneft':'JSC Gazprom Neft',
'LUKOIL':'Lukoil Co',
'MBT':'Mobile Telesystems',
'MDMOJC':'MDM Bk Open Jt Stk Co',
'ALROSA':'Open Jt Stk Co ALROSA',
'ROSNEF':'OJSC Oil Co Rosneft',
'RSBZAO':'Jt Stk Co Russian Standard Bk',
'RUSAGB':'Russian Agric Bk',
'RUSRAI':'JSC Russian Railways',
'SBERBANK':'SBERBANK',
'VIP':'OPEN Jt Stk Co VIMPEL Comms',
'VTB':'JSC VTB Bk'}

    

comp = pd.concat([aux2,aux, bayes], axis = 1).sort_values(by='CountryRank',ascending = False)
comp.rename(columns = {'CountryRank': 'Neuronal Net',
                        'Russian Fedn':'Connectedness to Sov',
                        'SD':'Bayes'}, inplace = True)

comp.loc['Russian Fedn','Connectedness to Sov'] = 1
comp.loc['Russian Fedn','Bayes'] = 1

if filter_market:
    path_bayes = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\probabilities\bds'
    directory = 'stddev_'+str(epsilon_up_time_parameter)+'_abs_'+str(absolute_up)
    filename = 'prob_weight_bayes_marketfilter_'\
        + str(epsilon_up_time_parameter)\
        + '_abs_'+str(absolute_up)+'.csv'

    bayes = pd.read_csv(os.path.join(path_bayes, directory, filename),index_col = 0)
    bayes.rename(index = dic_count, inplace = True)
else:
    path_bayes = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Sliced_ep_draw_new\original_dates'
    
    filename = 'prob_weight_bayes_notfilter_'\
        + str(epsilon_up_time_parameter)\
        + '_abs_'+str(absolute_up)+'.csv'

path_to_save = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\probabilities\CountryRank'

comp.to_csv(os.path.join(path_to_save, filename))

print(comp)


# print(country_rank_df)


def plot(country_rank_df):
    entities_list = list(country_rank_df['Entity'])
    ctry_rank = list(country_rank_df['CountryRank'])
    n_groups = len(entities_list)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4

    rects1 = plt.bar(index, ctry_rank, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Counterparties')

    #plt.xlabel('Counterparties', fontsize=15)
    plt.ylabel('CountryRank', fontsize =15)
    plt.title('CountryRank for European counterparties', fontsize = 20)
    plt.xticks(index + bar_width , entities_list, rotation='vertical', fontsize=10)
    plt.legend()

    plt.tight_layout()
    plt.show()



# plot(country_rank_df)

#ut.write_ctry_rank_results(country_rank_df, 'country_rank_eur_bank.csv')










#------------------------------End of File------------------------------

#alternative coutnry rank implementation
'''
weight_matrix = nc.preprocess_weight_matrix(weight_matrix)
input_stressed_node = 4
parent_list, country_rank = nc.prim_span_tree_alt(weight_matrix, input_stressed_node, input_initial_stress)
#print parent_list, country_rank
country_rank_df = pd.DataFrame(index = input_entities_list, columns=['Region', 'CountryRank'])
country_rank_df['CountryRank'] = country_rank
country_rank_df['Region'] = input_region_list
country_rank_df = country_rank_df.sort_values(by=['CountryRank'], ascending = [0])
country_rank_df.to_csv('country_rank.csv')
print country_rank_df
print country_rank_df[country_rank_df['Region'] == 'E.Eur']
print '----------------------------------------------------------------'
print ['Done evaluating CountryRank:', str(datetime.datetime.now() - start_time)]
print '----------------------------------------------------------------'
#ut.draw_countryrank_network(input_entities_list, list(country_rank), visited_list, edge_list, edge_weight)
'''