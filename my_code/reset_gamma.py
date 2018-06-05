import pandas as pd
import numpy as np
import os 

def read_data():
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis'
    df_port = pd.read_csv(os.path.join(path,'PortfolioA.csv'), index_col = 1)
    df_Omega = pd.read_csv(os.path.join(path,'Omega.csv'))
    return df_port, df_Omega

def read_probabilities(**kwargs):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Probabilities'
    score = kwargs['score']
    alg = kwargs['algorithm']
    if alg == 'tabu':
        alg = alg + str(kwargs['tabu_param'])

    directory = score + '_' + alg + '_' + str(200) + '_' + str(100)

    filename = [file for file in os.listdir(os.path.join(path,directory)) if 'bayes' in file][0]
      
    df_prob = pd.read_csv(os.path.join(path,directory,filename), index_col = 0)
    if df_prob.columns.empty:
        df_prob = pd.read_csv(os.path.join(path,directory,filename), index_col = 0, sep = ';')

    return df_prob

def read_prob_v2(time_para, delay,percent):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Probabilities'
    directory = 'time_'+str(time_para)+'_delay_'+str(delay)+'_per_'+str(percent)
    files = [f for f in os.listdir(os.path.join(path,directory)) if f.endswith('prob_matrix.csv')]
    list_prob = []
    i = 0
    for file in files:
        i += 1
        df = pd.read_csv(os.path.join(path,directory,file), index_col = 0)
        list_prob.append(pd.DataFrame(df.transpose()['RUSSIA'].values, index = df.index, columns = ['Period_' + str(i)]))
    return list_prob

def change_gamma_v2(df_portfolio, df_prob):

    dic_count = {'RUSSIA':'Government of the Russian Federation',
    'AKT':'Oil Transporting Jt Stk Co Transneft',
    'BKECON':'Vnesheconombank',
    'BOM':'Bank of Moscow OJSC',
    'CITMOS':'City Moscow',
    'GAZPRU':'GAZPROM PJSC',
    'GAZPRU.Gneft':'JSC Gazprom Neft',
    'LUKOIL':'LUKOIL PJSC',
    'MBT':'Mobile Telesystems',
    'MDMOJC':'MDM Bank JSC',
    'ALROSA':'ALROSA Company Ltd',
    'ROSNEF':'Rosneftegaz OJSC',
    'RSBZAO':'Jt Stk Co Russian Standard Bk',
    'RUSAGB':'Joint-stock company Russian Agricultural Bank',
    'RUSRAI':'JSC Russian Railways',
    'SBERBANK':'Sberbank of Russia',
    'VIP':'VimpelCom Ltd.',
    'VTB':'VTB Bank (public joint-stock company)'}

    df_prob.rename(index = dic_count, inplace = True)

    df_aux = df_portfolio.drop([i for i in df_portfolio.index if i not in df_prob.index])
    df_prob.drop([i for i in df_prob.index if i not in df_portfolio.index], inplace = True)

    for i in df_aux.index:
        df_aux.loc[i,'gamma'] = df_prob.iloc[df_prob.index.tolist().index(i),0]

    df_aux.loc['Government of the Russian Federation','gamma'] = 1.0
    df_aux.sort_values(by = 'gamma', ascending = False, inplace = True)

    return df_aux

def change_gamma(df_prob, df_port):

def get_name(k):
    
    dic_count_1 = {'RUSSIA':'Russian Fedn',
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


    dic_count_2 = {'RUSSIA':'Government of the Russian Federation',
    'AKT':'Oil Transporting Jt Stk Co Transneft',
    'BKECON':'Vnesheconombank',
    'BOM':'Bank of Moscow OJSC',
    'CITMOS':'City Moscow',
    'GAZPRU':'GAZPROM PJSC',
    'GAZPRU.Gneft':'JSC Gazprom Neft',
    'LUKOIL':'LUKOIL PJSC',
    'MBT':'Mobile Telesystems',
    'MDMOJC':'MDM Bank JSC',
    'ALROSA':'ALROSA Company Ltd',
    'ROSNEF':'Rosneftegaz OJSC',
    'RSBZAO':'Jt Stk Co Russian Standard Bk',
    'RUSAGB':'Joint-stock company Russian Agricultural Bank',
    'RUSRAI':'JSC Russian Railways',
    'SBERBANK':'Sberbank of Russia',
    'VIP':'VimpelCom Ltd.',
    'VTB':'VTB Bank (public joint-stock company)'}


    lKey = [key for key, value in lDictionary.iteritems() if value == lValue][0]



def change_gamma(df_port, df_prob):

    dic_count = {'Russian.Fedn':'Government of the Russian Federation',
    'Transneft':'Oil Transporting Jt Stk Co Transneft',
    'Vnesheconbk':'Vnesheconombank',
    'Bank.Moscow':'Bank of Moscow OJSC',
    'City.Moscow':'City Moscow',
    'GAZPROM':'GAZPROM PJSC',
    'Gazprom.Neft':'JSC Gazprom Neft',
    'Lukoil':'LUKOIL PJSC',
    'Mobile.Te.Sy':'Mobile Telesystems',
    'MDM.Bk':'MDM Bank JSC',
    'ALROSA':'ALROSA Company Ltd',
    'Rosneft':'Rosneftegaz OJSC',
    'Ru.Std.Bk':'Jt Stk Co Russian Standard Bk',
    'Ru.Agric.Bk':'Joint-stock company Russian Agricultural Bank',
    'Ru.Railways':'JSC Russian Railways',
    'SBERBANK':'Sberbank of Russia',
    'VIMPELCOM':'VimpelCom Ltd.',
    'VTB.Bk':'VTB Bank (public joint-stock company)'}

    df_prob.rename(index = dic_count, inplace = True)

    sorter = dict(zip(df_prob.index.tolist(),range(1,len(df_prob.index)+1)))
    sorter['Government of the Russian Federation'] = 0

    df_port['Ordered'] = df_port['NAME_LEGAL_ULTIMATE_PARENT'].map(sorter)
    df_port.sort_values(by = 'Ordered', ascending = True, inplace = True)
    df_port.drop('Ordered', axis = 1, inplace = True)
    df_port.index = range(1,len(df_port) + 1)

    count = [c for c in df_prob.index if c not in df_port['NAME_LEGAL_ULTIMATE_PARENT'].values]
    df_prob.drop(count, axis = 0, inplace = True)
    df_port.loc[2:,'gamma'] = df_prob['mean'].values

    return df_port

def write_new_portfolio_v2(df, time_para, delay,percent, period):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model'
    directory = 'time_'+str(time_para)+'_delay_'+str(delay)+'_per_'+str(percent)
    if not os.path.isdir(os.path.join(path,directory)):
        os.makedirs(os.path.join(path,directory))
    df.to_csv(os.path.join(path,directory, 'period_' + str(period) + '_portfolio.csv'))

def read_prob_comparison(time, delay, percent):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Sliced_ep_draw_new\original_dates'
    if percent == 0:
        directory = 'std_'+str(time)+'_long'
        filename = 'prob_'+str(time)+'_delay_'+str(delay)+'.csv'
    else:
        directory = 'std_'+str(time)+'_long_'+str(percent)
        filename = 'prob_'+str(time)+'_delay_'+str(delay)+'_'+str(percent)+'.csv'
    df = pd.read_csv(os.path.join(path,directory,filename),index_col = 0)
    return df


def write_new_portfolio_v3(df, time, delay, percent):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\compare_period'

    if percent == 0:
        directory = 'std_'+str(time)+'_long'
        filename = 'portfolio_'+str(time)+'_delay_'+str(delay)+'.csv'
    else:
        directory = 'std_'+str(time)+'_long_'+str(percent)
        filename = 'portfolio_'+str(time)+'_delay_'+str(delay)+'_'+str(percent)+'.csv'
    if not os.path.isdir(os.path.join(path,directory)):
        os.makedirs(os.path.join(path,directory))
    df.to_csv(os.path.join(path,directory,filename))

def main3():
    df_port, df_Omega = read_data()
    for time in [10,15,20]:
        for percent in [0,5]:
            for delay in [1,2,3]:
                df_prob = read_prob_comparison(time,delay, percent)
                df_prob.loc['RUSSIA','SD'] = 1
                df_new = change_gamma_v2(df_port, df_prob)
                write_new_portfolio_v3(df_new, time, delay, percent)

def main2():
    delay = 1
    percent = 0
    df_port, df_Omega = read_data()
    # print(df_port)
    for time_para in [10, 15, 20]:
        list_prob = read_prob_v2(time_para, delay, percent)
        i = 0
        for df_prob in list_prob:
            i += 1
            df_new = change_gamma_v2(df_port, df_prob)
            write_new_portfolio_v2(df_new, time_para, delay, percent, i)

def write_new_portfolio(df,**kwargs):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model'
    score = kwargs['score']
    alg = kwargs['algorithm']
    if alg == 'tabu':
        alg = alg + str(kwargs['tabu_param'])

    directory = score + '_' + alg + '_' + str(200) + '_' + str(100)

    if os.path.isdir(os.path.join(path, directory)):
        df.to_csv(os.path.join(path,directory,'PortfolioA_' + score + '_' + alg + '.csv'))
    else:
        os.makedirs(os.path.join(path, directory))
        df.to_csv(os.path.join(path,directory,'PortfolioA_' + score + '_' + alg + '.csv'))

def main():
    score = 'bds'
    algorithm = 'tabu10'
    df_port, df_Omega = read_data()
    df_prob = read_probabilities(score = score, algorithm = algorithm)
    df_port = change_gamma(df_port, df_prob)
    write_new_portfolio(df_port, score = score, algorithm = algorithm)

if __name__ == '__main__':
	main3()