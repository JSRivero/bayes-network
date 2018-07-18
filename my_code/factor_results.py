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
    Portfolio = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\portfolio\PortfolioA.csv', sep=',',index_col = 2)
    Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\portfolio\Omega.csv', index_col = 0,header = 0)
    return Portfolio, Omega

def capital_no_contagion(Portfolio, Omega, N_0):
    start = time.time()

    m = len(Portfolio)    # number of counterparties in the portfolio
    N = N_0               # number of simulations
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

    return UL, EL

def capital_contagion(Portfolio, Omega, N_0, M):

    start = time.time()

    U_c = pd.DataFrame(np.zeros((4,M)),index = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'])

    m = len(Portfolio)    # number of counterparties in the portfolio
    N = N_0             # number of simulations
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

    EXP_LOSS = 0

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
        EXP_LOSS += np.mean(Loss_c)

        # UL_98_c = np.percentile(Loss_c, 98)
        UL_99_c = np.percentile(Loss_c, 99)
        UL_995_c = np.percentile(Loss_c, 99.5)
        UL_999_c = np.percentile(Loss_c, 99.9)
        UL_9999_c = np.percentile(Loss_c, 99.99)
        U_c.loc[:,i] = [UL_99_c, UL_995_c, UL_999_c,UL_9999_c]

    EL = EXP_LOSS/M

    # print(U_c)

    # print(np.mean(U_c.transpose(),0))


    # UL_c = np.array([ UL_99_c, UL_995_c, UL_999_c, UL_9999_c])

    UL_c = np.mean(U_c,1).tolist()
    # print(UL_c)
    
    end = time.time()

    print(end-start)

    if M!=1:
        UL_std = np.std(U_c,1).tolist()
        return UL_c, UL_std, EL
    else:
    	return UL_c, EL


def change_prob(df_port, df_prob):
    for i in df_port.index:
        if i !='RUSSIA':
            df_port.loc[i,'gamma'] = df_prob.loc[i,'SD']
    df_port.loc['RUSSIA','gamma']= 1.0
    df_port.sort_values(by = 'gamma', ascending = False, inplace = True)
    return df_port

def change_prob_series(df_port, df_prob, col):
    return pd.concat([df_port.drop('gamma',axis=1), df_prob[col]],axis = 1).rename(columns = {col:'gamma'})


def saving_plot(Uinitial, period, score, bar):
    if bar == 'stack':
        Uarray = [Uinitial[0]]
        for i in range(1,len(Uinitial)):
            U1 = Uinitial[i-1]
            U2 = Uinitial[i]
            # Uarray.append(U2)
            Uarray.append([max(int(U2[k]-U1[k]),0) for k in range(len(U1))])
        width = 0.35
        size_text = 25
    elif bar == 'group':
        Uarray = Uinitial[:]
        width = 'auto'
        size_text = 20

    # print(Uarray)
    # print(Uinitial)

    quantile = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%']

    # names = ['No Contagion','Contagion ' + score.upper() + ' score']
    names = ['Standard Model','BN Model', 'CountryRank Model']

    # colors_inside = ['rgb(58,200,225)',
    colors_inside = {names[0]:'rgb(51,235,51)',
                    names[1]:'rgb(255, 51, 51)',
                    names[2]:'rgb(58,200,225)'}
    color_line = 'rgb(8,48,100)'
    color_text = 'rgb(0,0,0)'

    traces = []
    annotations = []


    for i in range(len(Uarray)):
        y = [int(u) for u in Uarray[i]]
        y_init = [int(u) for u in Uinitial[i]]
        y_text = [str(round(u/10**4)/10**2)+'M' for u in y_init]
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
                color= 'rgb(0,0,0)',
                size = size_text), 
            name = names[i],
            marker=dict(
                color=colors_inside[names[i]],
                line=dict(
                    color='rgb(8,48,100)',
                    width=1.5),
                ),
            opacity=0.8,
            width = width
            )
        )

    layout = go.Layout(
      font = dict(family = 'Arial', size = 25, color = 'black'),
      xaxis=dict(title= 'Quantile', titlefont = dict(family = 'Arial',
            size = 25, color = 'black'),color = color_text),
      yaxis = dict(title= 'Loss', titlefont = dict(family = 'Arial',
            size = 25, color = 'black'),color = color_text),
      barmode= bar,
      title = 'Quantiles of the Loss distribution',
      titlefont = dict(family = 'Arial', size = 30, color = 'black'),
      width=1500, height=1.5*640,
      legend=dict(x = 0.35, y = -0.16, orientation='h')
    )

    fig = go.Figure(data=traces, layout = layout)

    path = os.path.join(r'C:\Users\Javier\Documents\MEGA\Thesis',r'results_'+period,'Capital_compare')
    figure_name = score+'_'+period+'_'+'capital_green_red_' + bar + '.png'

    py.image.save_as(fig,os.path.join(path, figure_name))

    plt.close('all')

def main():
    Portfolio, Omega = read_data()
    score = 'bic'
    period = '5Y'
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\results_' + period 
    dir_network = r'networks_not_filtered'
    filename = 'comp_' + score + '_sumit.xlsx'

    df_prob = pd.read_excel(os.path.join(path,dir_network,filename),index_col = 0)

    df_port_sumit = change_prob_series(Portfolio, df_prob, 'Sumit')
    df_port_bayes = change_prob_series(Portfolio, df_prob, 'Bayes')

    df_port_sumit = pd.read_csv(os.path.join(path,'portfolio',r'Portfolio_sumit.csv'))
    df_port_bayes = pd.read_csv(os.path.join(path,'portfolio',r'Portfolio_bayes.csv'))

    N = 10**6

    UL, EL = capital_no_contagion(Portfolio, Omega, N)

    M = -1
    if M == 1:
        UL_c, EL_c = capital_contagion(df_port, Omega, N, M)
        pd.DataFrame([UL,UL_c],columns = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'],
            index = ['No contagion', 'Contagion']).to_excel(os.path.join(
                r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital',score+'_UL_c.xlsx'))
        pd.DataFrame([EL, EL_c], index = ['No Contagion','Contagion']).to_excel(os.path.join(
                    r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital',score+'_exp_loss.xlsx'))
    elif M > 1:
        UL_c, SD_c, EL_c = capital_contagion(df_port, Omega, N, M)
        pd.DataFrame([UL,UL_c, SD_c],columns = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'],
            index = ['No contagion', 'Contagion', 'std']).to_excel(os.path.join(
                r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital',score+'_UL_c_SD.xlsx'))
        pd.DataFrame([EL, EL_c], index = ['No Contagion','Contagion']).to_excel(os.path.join(
                    r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital',score+'_exp_loss.xlsx'))
    else:
        temp_U = pd.read_excel(os.path.join(r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital_compare',score+'_sumit_quant_SD.xlsx'),index_col = 0)
        UL = temp_U.loc['No contagion',:].tolist()
        UL_bayes = temp_U.loc['Bayes',:].tolist()
        UL_sumit = temp_U.loc['Sumit',:].tolist()
    print(UL)
    print(UL_bayes)
    saving_plot([UL, UL_bayes], period, score, 'group')

def main_comparison():
    Portfolio, Omega = read_data()
    score = 'bic'
    period = '5Y'
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\results_' + period 
    dir_network = r'networks_not_filtered'
    filename = 'comp_' + score + '_sumit.xlsx'

    df_prob = pd.read_excel(os.path.join(path,dir_network,filename),index_col = 0)

    # df_port_sumit = change_prob_series(Portfolio, df_prob, 'Sumit')
    # df_port_bayes = change_prob_series(Portfolio, df_prob, 'Bayes')

    df_port_sumit = pd.read_csv(os.path.join(path,'portfolio',r'Portfolio_sumit.csv'))
    df_port_bayes = pd.read_csv(os.path.join(path,'portfolio',r'Portfolio_bayes.csv'))

    N = 10**6

    UL, EL = capital_no_contagion(Portfolio, Omega, N)

    M = -1
    if M == 1:
        UL_bayes, EL_bayes = capital_contagion(df_port_bayes, Omega, N, M)
        UL_sumit, EL_sumit = capital_contagion(df_port_sumit, Omega, N, M)
        pd.DataFrame([UL,UL_bayes,UL_sumit],columns = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'],
            index = ['No contagion', 'Bayes', 'Sumit']).to_excel(os.path.join(
                r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital_compare',score+'_sumit_quantiles.xlsx'))
        pd.DataFrame([EL, EL_bayes, EL_sumit], index = ['No Contagion','Bayes', 'Sumit']).to_excel(os.path.join(
                    r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital_compare',score+'_sumit_exp_loss.xlsx'))
    elif M > 1:
        UL_bayes, SD_bayes, EL_bayes = capital_contagion(df_port_bayes, Omega, N, M)
        UL_sumit, SD_sumit, EL_sumit = capital_contagion(df_port_sumit, Omega, N, M)
        pd.DataFrame([UL,UL_bayes, SD_bayes, UL_sumit, SD_sumit],columns = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'],
            index = ['No contagion', 'Bayes', 'std Bayes', 'Sumit', 'std Sumit']).to_excel(os.path.join(
                r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital_compare', score+'_sumit_quant_SD.xlsx'))
        pd.DataFrame([EL, EL_bayes, EL_sumit], index = ['No Contagion','Bayes','Sumit']).to_excel(os.path.join(
                    r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital_compare',score+'_sumit_SD_exp_loss.xlsx'))
    else:
        temp_U = pd.read_excel(os.path.join(r'C:\Users\Javier\Documents\MEGA\Thesis\results_5Y\Capital_compare',score+'_sumit_quant_SD.xlsx'),index_col = 0)
        UL = temp_U.loc['No contagion',:].tolist()
        UL_bayes = temp_U.loc['Bayes',:].tolist()
        UL_sumit = temp_U.loc['Sumit',:].tolist()
    print(UL)
    print(UL_bayes)
    print(UL_sumit)
    saving_plot([UL, UL_sumit, UL_bayes], period, score, 'group')


if __name__ == '__main__':
    main()