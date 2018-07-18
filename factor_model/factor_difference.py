import numpy as np
import pandas as pd
from scipy.stats import mvn, norm, gaussian_kde
from scipy.optimize import fsolve
from matplotlib import pyplot as plt, ticker as ticker
from scipy import integrate
import os
import plotly.plotly as py
import plotly.graph_objs as go
import time
import matplotlib.pyplot as plt

def read_data():
    Portfolio = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\PortfolioA.csv', sep=',',index_col = 1)
    Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\Omega.csv', index_col = 0,header = 0)
    return Portfolio, Omega

def change_prob(df_portfolio, df_prob, choice):
    if choice == 'ticker':
        dic = {'RUSSIA':'Government of the Russian Federation',
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

    elif choice == 'names':
        dic = {'Russian Fedn':'Government of the Russian Federation',
        'Oil Transporting Jt Stk Co Transneft':'Oil Transporting Jt Stk Co Transneft',
        'Vnesheconombank':'Vnesheconombank',
        'Bk of Moscow':'Bank of Moscow OJSC',
        'City Moscow':'City Moscow',
        'JSC GAZPROM':'GAZPROM PJSC',
        'JSC Gazprom Neft':'JSC Gazprom Neft',
        'Lukoil Co':'LUKOIL PJSC',
        'Mobile Telesystems':'Mobile Telesystems',
        'MDM Bk Open Jt Stk Co':'MDM Bank JSC',
        'Open Jt Stk Co ALROSA':'ALROSA Company Ltd',
        'OJSC Oil Co Rosneft':'Rosneftegaz OJSC',
        'Jt Stk Co Russian Standard Bk':'Jt Stk Co Russian Standard Bk',
        'Russian Agric Bk':'Joint-stock company Russian Agricultural Bank',
        'JSC Russian Railways':'JSC Russian Railways',
        'SBERBANK':'Sberbank of Russia',
        'OPEN Jt Stk Co VIMPEL Comms':'VimpelCom Ltd.',
        'JSC VTB Bk':'VTB Bank (public joint-stock company)'}

    # print(df_prob)

    df_prob.rename(index = dic, inplace = True)
    # print(df_prob)
    df_prob.drop('EVRGSA',inplace = True)
    # print(df_portfolio.index)
    # print(df_prob.index)
    # print(df_prob)
    # print([i for i in df_portfolio.index if i not in df_prob.index])
    df_aux = df_portfolio.drop([i for i in df_portfolio.index if i not in df_prob.index])
    # print(df_aux)
    df_prob.drop([i for i in df_prob.index if i not in df_portfolio.index], inplace = True)

    # print(df_prob)

    for i in df_aux.index:
        df_aux.loc[i,'gamma'] = df_prob.loc[i,0]

    df_aux.loc['Government of the Russian Federation','gamma'] = 1.0
    df_aux.sort_values(by = 'gamma', ascending = False, inplace = True)

    return df_aux


def compare_no_cont():
    start = time.time()

    Portfolio, Omega = read_data()

    m = len(Portfolio)    # number of counterparties in the portfolio
    N = 10**6             # number of simulations
    p = Omega.shape[0]    # number of systematic factors

    # Now we get the beta, gamma (PDGSD), PD, EAD and LGD
    Beta = Portfolio[[col for col in list(Portfolio) if col.startswith('beta')]].values
    gamma = Portfolio['gamma'].values
    PD = Portfolio['PD'].values
    EAD = Portfolio['EAD'].values
    LGD = Portfolio['LGD'].values

    df_port = Portfolio[['SOV_ID','PD','EAD','LGD']]

    # Analytical Expected Loss
    EL_an = np.sum(PD*EAD*LGD)

    # Calibrate default thresholds with PDs
    d = norm.ppf(PD)
    # perform a Cholesky factorisation to sample normal distributed
    # numbers with covariaton matrix Omega
    L = np.linalg.cholesky(Omega)

    # np.random.seed(10)
    # generate independent normals
    Z = np.random.standard_normal((p, N))

    # convert independent unit normals to correlated
    F = np.dot(L, Z)

    # idiosyncratic loading s.t. the returns are standard normal
    id_load = np.diagonal(np.sqrt(1-np.dot(np.dot(Beta,Omega),Beta.T)))
    epsilon = np.random.standard_normal((N, m))
    # Put everything together to get the returns
    X = np.dot(Beta,F) + (id_load*epsilon).T
    X_df = pd.DataFrame(np.dot(Beta,F) + (id_load*epsilon).T)

    # Calculate UL with no contagion
    # construct default indicator
    I = (((X.T-d)<0))
    I_df = pd.DataFrame(I)
    L = (EAD*LGD*I).T
    
    # print(np.mean(L,axis=1))
    Loss=np.sum(L,axis=0)

    EL = np.mean(Loss)

    # UL_98 = np.percentile(Loss, 98)
    UL_99 = np.percentile(Loss, 99)
    UL_995 = np.percentile(Loss, 99.5)
    UL_999 = np.percentile(Loss, 99.9)
    UL_9999 = np.percentile(Loss, 99.99)

    UL = np.array([ UL_99, UL_995, UL_999, UL_9999])

    return UL


def compare_cont(Portfolio, Omega):

    start = time.time()
    # Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\Omega.csv', index_col = 0,header = 0)

    M = 1

    U_c = pd.DataFrame(np.zeros((4,M)),index = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'])

    m = len(Portfolio)    # number of counterparties in the portfolio
    N = 10**2             # number of simulations
    p = Omega.shape[0]    # number of systematic factors

    # Now we get the beta, gamma (PDGSD), PD, EAD and LGD
    Beta = Portfolio[[col for col in list(Portfolio) if col.startswith('beta')]].values
    gamma = Portfolio['gamma'].values
    PD = Portfolio['PD'].values
    EAD = Portfolio['EAD'].values
    LGD = Portfolio['LGD'].values

    df_port = Portfolio[['SOV_ID','PD','EAD','LGD']]

    # Analytical Expected Loss
    EL_an = np.sum(PD*EAD*LGD)

     # Calibrate default thresholds with PDs
    d = norm.ppf(PD)
    # perform a Cholesky factorisation to sample normal distributed
    # numbers with covariaton matrix Omega
    L = np.linalg.cholesky(Omega)

    # Calculate UL with contagion

    SOV_ID = Portfolio['SOV_ID'].values
    SOV_LINK = Portfolio['SOV_LINK'].values

    df_d = pd.DataFrame(np.zeros((m,3)), columns = ['SOV_ID','Dsd','Dnsd'])
    df_d['SOV_ID']=SOV_ID

    PDs = df_port[df_port['SOV_ID']==1]['PD'].values[0]

    Dsd = np.zeros(m)
    Dnsd = np.zeros(m)

    # With contagion
    for i in range(0,m):
        if SOV_ID[i] != 0:
            Dsd[i] = d[i]
            Dnsd[i] = d[i]
        else:
            sov_ind = np.nonzero(SOV_ID == SOV_LINK[i])[0][0]
            PDs = PD[sov_ind]
            corr = np.dot(np.dot((Beta[i]).T,Omega),(Beta[sov_ind]))

            Fsd = lambda x: mvn.mvndst([-100, -100],\
                [x, norm.ppf(PDs)],[0,0],corr)[1] / PDs - gamma[i]
            Dsd[i] = fsolve(Fsd, norm.ppf(gamma[i])) # is there a better initial guess?
            Fnsd = lambda x: mvn.mvndst([-100, norm.ppf(PDs)],\
                [x, 100],[0,1],corr)[1] - PD[i] + gamma[i]*PDs
            Dnsd[i] = fsolve(Fnsd, norm.ppf(PD[i])) # is there a better initial guess?
            if Dsd[i]< d[i] or PD[i]<PD[sov_ind]:
                Dsd[i] = d[i]
                Dnsd[i] = d[i]

    df_d['Dsd'] = Dsd
    df_d['Dnsd'] = Dnsd

    sov_ind = df_d[df_d['SOV_ID']==1].index[0]

    # Thresholds
    # D = np.array([Dnsd]*N).T
    # D_df = pd.concat([df_d['Dnsd']]*N,axis = 1)
    # D_df.columns = range(N)

    # D2 = D_df.transpose()

    for i in range(M):
        # np.random.seed(10)
        # generate independent normals
        Z = np.random.standard_normal((p, N))
    
        # convert independent unit normals to correlated
        F = np.dot(L, Z)

        # idiosyncratic loading s.t. the returns are standard normal
        id_load = np.diagonal(np.sqrt(1-np.dot(np.dot(Beta,Omega),Beta.T)))
        epsilon = np.random.standard_normal((N, m))
        # Put everything together to get the returns
        X = np.dot(Beta,F) + (id_load*epsilon).T
        X_df = pd.DataFrame(np.dot(Beta,F) + (id_load*epsilon).T)

        X2 = X_df.transpose()

        X_SD = X2[X2[sov_ind]<df_d.loc[sov_ind, 'Dsd']].copy()
        X_NSD = X2.drop(X_SD.index, axis = 0)


        I_SD = X_SD.lt(df_d['Dsd'], axis = 1)
        I_NSD = X_NSD.lt(df_d['Dnsd'],axis = 1)

        I_c = pd.concat([I_SD,I_NSD], axis = 0)

        I_aux = np.array(I_c)

        Loss = (EAD * LGD * I_aux)

        Loss_c = np.sum(Loss,axis=1)

        # Arithmetic mean of Loss
        EL_c = np.mean(Loss_c)

        # UL_98_c = np.percentile(Loss_c, 98)
        UL_99_c = np.percentile(Loss_c, 99)
        UL_995_c = np.percentile(Loss_c, 99.5)
        UL_999_c = np.percentile(Loss_c, 99.9)
        UL_9999_c = np.percentile(Loss_c, 99.99)
        U_c.loc[:,i] = [UL_99_c, UL_995_c, UL_999_c,UL_9999_c]

    # print(U_c)

    # print(np.mean(U_c.transpose(),0))


    # UL_c = np.array([ UL_99_c, UL_995_c, UL_999_c, UL_9999_c])

    UL_c = np.mean(U_c,1).tolist()
    # print(UL_c)
    UL_std = np.std(U_c,1).tolist()
    
    end = time.time()

    print(end-start)

    return UL_c, UL_std

def plot_three_diff(Uinitial,*arg):

    Uarray = [Uinitial[0]]

    for i in range(1,len(Uinitial)):
        U1 = Uinitial[i-1]
        U2 = Uinitial[i]
        # Uarray.append(U2)
        Uarray.append([max(int(U2[k]-U1[k]),0) for k in range(len(U1))])
    # print(Uarray)
    # print(Uinitial)

    quantile = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%']
    names = ['No Contagion','Contation Bayes k2',#'Contagion Bayes k2',
        'Contagion Bayes BDs','Contagion Sumit']


    colors_inside = ['rgb(158,202,225)',
                    'rgb(58,200,225)',#'rgb(51,255,51)',
                    'rgb(255, 51, 51)',
                    'rgb(255,51,255)']
    colors_inside = ['rgb(58,200,225)',
                    'rgb(51,255,51)',
                    'rgb(255, 51, 51)',
                    'rgb(255,51,255)']
    # colors_inside = ['rgb(158,202,225)','rgb(255, 51, 51)']

    color_text = 'rgb(0,0,0)'

    width = 0.4

    traces = []
    annotations = []
    # print('Uinitial')
    # print(Uinitial)
    # print('Uarray')
    # print(Uarray)

    for i in range(len(Uarray)):
        y = [int(u) for u in Uarray[i]]
        y_init = [int(u) for u in Uinitial[i]]
        y_text = [str(int(u/10**4)/10**2)+'M' for u in y_init]
        # y_text = [str(int(u/10**4)/10**2)+'M' for u in y]#"{:,}".format(int(u)) 
        # print(i)
        # print(y)
        # print(y_init)
        # print(y_text)
        traces.append(go.Bar(
            x = quantile,
            y = y,
            text = y_text,
            textposition = 'auto',
            textfont = dict(
                family = 'Arial',
                color= 'black',
                size = 15), 
            name = names[i],
            marker=dict(
                color=colors_inside[i],
                line=dict(
                    color='rgb(8,48,100)',
                    width=1.5),
                ),
            opacity=0.6,
            width = width
            )
        )

    layout = go.Layout(
      xaxis=dict(title= 'Quantile', titlefont = dict(family = 'Arial',
            size = 20, color = 'black'),color = color_text),
      yaxis = dict(title= 'Capital', titlefont = dict(family = 'Arial',
            size = 20, color = 'black'),color = color_text),
      barmode= 'stack',
      title = 'Capital Requirement from contagion with delay 3 and time 10 days',
      width=1500, height=1.4*640
    )

    # layout = go.Layout(
    #   'xaxis': {'title': 'Quantile', titlefont = dict(family = 'Arial',
    #         size = 16, color = 'black'),
    #      'color': color_text},
    #   'yaxis': {'title': 'Capital ', 'color': color_text},
    #   'barmode': 'group',
    #   'title': 'Capital Requirement from different delays',
    #   width=1000, height=640
    # )

    data = traces


    # annotations = []
    # for i in range(len(UL_c)):
    #     annotations.append(dict(x=quantile[i], y=y_cont[i] + y_no_cont[i] + 8*10**4, text=y_text_cont[i],
    #                               font=dict(family='Arial', size=10,
    #                               color=color_text),
    #                               showarrow=False,))
    # for i in range(len(UL)):
    #     annotations.append(dict(x=quantile[i], y=y_no_cont[i] - 10**5, text=y_text_no_cont[i],
    #                               font=dict(family='Arial', size=10,
    #                               color=color_text),
    #                               showarrow=False,))
    # layout['annotations'] = annotations

    fig = go.Figure(data=data, layout = layout)

    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\New_Capital'
    # folder = 'cap_comparisons_diff'
    # if not os.path.isdir(os.path.join(path,folder)):
    #     os.makedirs(os.path.join(path,folder))
    if len(arg) == 4:
    	time = arg[0]
    	delay = arg[1]
    	absolute = arg[2]
    	filtered = arg[3]
    	figure_name = 'time_' + str(time)+'_delay_'+str(delay)+'_absolute_'+str(absolute)+'_' +filtered + '.png'
    else:
    	figure_name = str(arg[0]) + '.eps'
    print('SAVING')
    plt.savefig(os.path.join(path, figure_name))
    # py.image.save_as(fig,os.path.join(path, figure_name))

    # py.plot(fig, filename = 'Capital Requirement')
    # plt.show()

    plt.close('all')

def main():
	path = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\probabilities\Comp_old_new'
	Portfolio, Omega = read_data()
	for time in [10]:
		for delay in [3]:
			folder_to_save = 'time_'+str(time)+'_delay_'+str(delay)
			for absolute in [0,10]:
				folder_to_read = 'time_'+str(time)+'_abs_'+str(absolute)
				for filtered in ['filter', 'notfilt']:
					file = filtered + '_' + str(time)+'_delay_' + str(delay) + '_abs_' + str(absolute) + '.csv'
					df = pd.read_csv(os.path.join(path,folder_to_read, file),index_col = 0)
					UL_c = []
					# UL_c.append(compare_no_cont)
					UL_c.append([1115153.28,1443009.111,2276912.391,3420615.947])
					for c in ['Bayes','Sumit New','Sumit Old']:
						print(c)
						temp = pd.DataFrame(data = df[c].values,index = df.index, columns = [0])
						print(temp)
						print(df[c])
						df_aux = change_prob(Portfolio, temp, 'name')
						print(df_aux['gamma'])
					print(UL_c)
					plot_three_diff(UL_c, time,delay, absolute, filtered)

def main2():
    Portfolio, Omega = read_data()
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\probabilities'
    time = 10
    delay = 3
    file = os.path.join(path,'bds_k2_log_prob.csv')
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered\probabilities\Comp_old_new\filtered'
    file = r'prob.csv'
    df = pd.read_csv(os.path.join(path,file), index_col = 0)
    UL_c = []
    std_c = []
    # UL_c.append(compare_no_cont)
    UL_c.append([1115153.28,1443009.111,2276912.391,3420615.947])
    # for c in ['bds','loglik','Sumit New']:
    for c in ['k2', 'BDs','Sumit']:
        temp = pd.DataFrame(data = df[c].values,index = df.index, columns = [0])
        df_aux = change_prob(Portfolio, temp, 'names')
        mean, std = compare_cont(df_aux, Omega)
        std_c.append(std)
        UL_c.append(mean)
        # UL_c.append(list(compare_cont(df_aux, Omega)))
    plot_three_diff(UL_c, 'test')
    print(std_c)

def main_total():


if __name__ == '__main__':
	main2()

