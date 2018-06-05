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

# Portfolio = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\PortfolioA.csv', sep=',',index_col = 0)

def simulation(time_para, delay,percent, period):
    
    start = time.time()

    # Read Portfolio and correlation matrix
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model'
    directory = 'time_' + str(time_para) + '_delay_' + str(delay) + '_per_' + str(percent)
    file = 'period_' + str(period) + '_portfolio.csv'

    Portfolio = pd.read_csv(os.path.join(path,directory,file), sep=',')
    # Read the correlation matrix from .csv
    Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\Omega.csv', index_col = 0,header = 0)

    m = len(Portfolio)    # number of counterparties in the portfolio
    N = 10**6             # number of simulations
    p = Omega.shape[0]    # number of systematic factors

    # Now we get the beta, gamma (PDGSD), PD, EAD and LGD
    Beta = Portfolio[[col for col in list(Portfolio) if col.startswith('beta')]].values
    gamma = Portfolio['gamma'].values
    PD = Portfolio['PD'].values
    EAD = Portfolio['EAD'].values
    LGD = Portfolio['LGD'].values

    # Analytical Expected Loss
    EL_an = np.sum(PD*EAD*LGD)

    # Calibrate default thresholds with PDs
    d = norm.ppf(PD)
    
    # perform a Cholesky factorisation to sample normal distributed
    # numbers with covariaton matrix Omega
    L = np.linalg.cholesky(Omega)

    np.random.seed(10)
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

    # Calculate UL with contagion

    SOV_ID = Portfolio['SOV_ID'].values
    SOV_LINK = Portfolio['SOV_LINK'].values

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
            Dsd[i] = fsolve(Fsd, norm.ppf(gamma[i])) # is thera a better initial guess?
            Fnsd = lambda x: mvn.mvndst([-100, norm.ppf(PDs)],\
                [x, 100],[0,1],corr)[1] - PD[i] + gamma[i]*PDs
            Dnsd[i] = fsolve(Fnsd, norm.ppf(PD[i])) # is there a better initial guess?
            if Dsd[i]< d[i] or PD[i]<PD[sov_ind]:
                Dsd[i] = d[i]
                Dnsd[i] = d[i]


    # Thresholds
    D = np.array([Dnsd]*N).T

    X_sov = X[np.nonzero(SOV_ID !=0)[0]]
    D_sov = D[np.nonzero(SOV_ID !=0)[0]]

    I_sov = (((X_sov-D_sov)<0))

    for i in range(0,N):
        for j in range(0,m):
            if SOV_ID[j]==0 and I_sov[SOV_LINK[j]-1,i] == 1:
                D[j,i] = Dsd[j]

    # construct default indicator
    I_c = ((X-D)<0).T

    L = (EAD*LGD*I_c).T
    # print(np.mean(L,axis=1))
    Loss_c=np.sum(L,axis=0)

    EL = np.mean(Loss)
    # Arithmetic mean of Loss
    EL_c = np.mean(Loss_c)

    # UL_98 = np.percentile(Loss, 98)
    UL_99 = np.percentile(Loss, 99)
    UL_995 = np.percentile(Loss, 99.5)
    UL_999 = np.percentile(Loss, 99.9)
    UL_9999 = np.percentile(Loss, 99.99)

    # UL_98_c = np.percentile(Loss_c, 98)
    UL_99_c = np.percentile(Loss_c, 99)
    UL_995_c = np.percentile(Loss_c, 99.5)
    UL_999_c = np.percentile(Loss_c, 99.9)
    UL_9999_c = np.percentile(Loss_c, 99.99)


    UL = np.array([ UL_99, UL_995, UL_999, UL_9999])
    UL_c = np.array([ UL_99_c, UL_995_c, UL_999_c, UL_9999_c])
    
    
    end = time.time()

    print(end-start)
    

    return UL, UL_c

def simulation_v2(time_para, delay, percent, period):
    start = time.time()

    # Read Portfolio and correlation matrix
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model'
    directory = 'time_' + str(time_para) + '_delay_' + str(delay) + '_per_' + str(percent)
    file = 'period_' + str(period) + '_portfolio.csv'

    Portfolio = pd.read_csv(os.path.join(path,directory,file), sep=',')
    # Read the correlation matrix from .csv
    Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\Omega.csv', index_col = 0,header = 0)

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

    np.random.seed(10)
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

    # Thresholds
    D = np.array([Dnsd]*N).T
    D_df = pd.concat([df_d['Dnsd']]*N,axis = 1)
    D_df.columns = range(N)

    X2 = X_df.transpose()
    D2 = D_df.transpose()

    sov_ind = df_d[df_d['SOV_ID']==1].index[0]

    X_SD = X2[X2[sov_ind]<df_d.loc[sov_ind, 'Dsd']].copy()

    X_NSD = X2.drop(X_SD.index, axis = 0)


    I_SD = X_SD.lt(df_d['Dsd'], axis = 1)
    I_NSD = X_NSD.lt(df_d['Dnsd'],axis = 1)

    I_c = pd.concat([I_SD,I_NSD], axis = 0)

    I_aux = np.array(I_c)

    L = (EAD * LGD * I_aux)

    Loss_c = np.sum(L,axis=1)

    EL = np.mean(Loss)
    # Arithmetic mean of Loss
    EL_c = np.mean(Loss_c)

    # UL_98 = np.percentile(Loss, 98)
    UL_99 = np.percentile(Loss, 99)
    UL_995 = np.percentile(Loss, 99.5)
    UL_999 = np.percentile(Loss, 99.9)
    UL_9999 = np.percentile(Loss, 99.99)

    # UL_98_c = np.percentile(Loss_c, 98)
    UL_99_c = np.percentile(Loss_c, 99)
    UL_995_c = np.percentile(Loss_c, 99.5)
    UL_999_c = np.percentile(Loss_c, 99.9)
    UL_9999_c = np.percentile(Loss_c, 99.99)


    UL = np.array([ UL_99, UL_995, UL_999, UL_9999])
    UL_c = np.array([ UL_99_c, UL_995_c, UL_999_c, UL_9999_c])
    
    end = time.time()

    print(end-start)

    return UL, UL_c

def compare_no_cont(time_para, delay, percent):
    start = time.time()

    # Read Portfolio and correlation matrix
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\compare_period'
    if percent != 0:
        directory = 'std_' + str(time_para) + '_long_' + str(percent)
        file = 'portfolio_' + str(time_para) + '_delay_' + str(delay)+'_'+str(percent)+'.csv'
    else:
        directory = 'std_' + str(time_para) + '_long'
        file = 'portfolio_' + str(time_para) + '_delay_' + str(delay)+'.csv'


    Portfolio = pd.read_csv(os.path.join(path,directory,file), sep=',')
    # Read the correlation matrix from .csv
    Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\Omega.csv', index_col = 0,header = 0)

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

def compare_cont(time_para, delay, percent):
    start = time.time()

    # Read Portfolio and correlation matrix
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\compare_period'
    if percent != 0:
        directory = 'std_' + str(time_para) + '_long_' + str(percent)
    else:
        directory = 'std_' + str(time_para) + '_long'

    file = 'portfolio_' + str(time_para) + '_delay_' + str(delay)+'.csv'

    Portfolio = pd.read_csv(os.path.join(path,directory,file), sep=',')
    # Read the correlation matrix from .csv
    Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\Omega.csv', index_col = 0,header = 0)

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

    # Thresholds
    D = np.array([Dnsd]*N).T
    D_df = pd.concat([df_d['Dnsd']]*N,axis = 1)
    D_df.columns = range(N)

    X2 = X_df.transpose()
    D2 = D_df.transpose()

    sov_ind = df_d[df_d['SOV_ID']==1].index[0]

    X_SD = X2[X2[sov_ind]<df_d.loc[sov_ind, 'Dsd']].copy()

    X_NSD = X2.drop(X_SD.index, axis = 0)


    I_SD = X_SD.lt(df_d['Dsd'], axis = 1)
    I_NSD = X_NSD.lt(df_d['Dnsd'],axis = 1)

    I_c = pd.concat([I_SD,I_NSD], axis = 0)

    I_aux = np.array(I_c)

    L = (EAD * LGD * I_aux)

    Loss_c = np.sum(L,axis=1)

    # Arithmetic mean of Loss
    EL_c = np.mean(Loss_c)

    # UL_98_c = np.percentile(Loss_c, 98)
    UL_99_c = np.percentile(Loss_c, 99)
    UL_995_c = np.percentile(Loss_c, 99.5)
    UL_999_c = np.percentile(Loss_c, 99.9)
    UL_9999_c = np.percentile(Loss_c, 99.99)


    UL_c = np.array([ UL_99_c, UL_995_c, UL_999_c, UL_9999_c])
    
    end = time.time()

    print(end-start)

    return UL_c

def compute_ioannis_capital():
    start = time.time()

    # Read the correlation matrix from .csv
    Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\Omega.csv', index_col = 0,header = 0)
    Portfolio = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\PortfolioA.csv')

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

    # Thresholds
    D = np.array([Dnsd]*N).T
    D_df = pd.concat([df_d['Dnsd']]*N,axis = 1)
    D_df.columns = range(N)

    X2 = X_df.transpose()
    D2 = D_df.transpose()

    sov_ind = df_d[df_d['SOV_ID']==1].index[0]

    X_SD = X2[X2[sov_ind]<df_d.loc[sov_ind, 'Dsd']].copy()

    X_NSD = X2.drop(X_SD.index, axis = 0)


    I_SD = X_SD.lt(df_d['Dsd'], axis = 1)
    I_NSD = X_NSD.lt(df_d['Dnsd'],axis = 1)

    I_c = pd.concat([I_SD,I_NSD], axis = 0)

    I_aux = np.array(I_c)

    L = (EAD * LGD * I_aux)

    Loss_c = np.sum(L,axis=1)

    # Arithmetic mean of Loss
    EL_c = np.mean(Loss_c)

    # UL_98_c = np.percentile(Loss_c, 98)
    UL_99_c = np.percentile(Loss_c, 99)
    UL_995_c = np.percentile(Loss_c, 99.5)
    UL_999_c = np.percentile(Loss_c, 99.9)
    UL_9999_c = np.percentile(Loss_c, 99.99)


    UL_c = np.array([ UL_99_c, UL_995_c, UL_999_c, UL_9999_c])
    
    end = time.time()

    print(end-start)

    return UL_c


def plot_result(UL, UL_c, *arg):
    # ind = np.arange(len(UL))
    # width = 0.5

    # p1 = plt.bar(ind, UL, width = width)
    # p2 = plt.bar(ind, np.maximum((UL_c-UL),0), bottom = UL, width = width)

    # plt.ylabel('Loss (millions)')
    # plt.xlabel('Quantile')
    # plt.xticks(ind, ('99%','99.5%', '99.9%','99.99%'))

    # plt.legend((p1[0],p2[0]),('Loss', 'Contagion add-on'))

    # if arg:
    #     path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods'
    #     folder = 'time_'+str(arg[0])+'_delay_'+str(arg[1])
    #     figure_name = 'Period_'+str(arg[2])+ '.png'
    #     plt.savefig(os.path.join(path,folder,figure_name))
    # else:
    #     plt.show()
    # plt.close('all')
    
    y_no_cont = [int(u) for u in UL]
    y_text_no_cont = ["{:,}".format(int(u)) for u in y_no_cont]
    quantile = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%']

    light_blue_inside = 'rgb(158,202,225)'
    light_blue_line = 'rgb(8,48,107)'
    blue_inside = 'rgb(58,200,225)'
    blue_line = 'rgb(8,48,107)'
    red_inside = 'rgb(220, 20, 60)'
    red_line = 'rgb(139, 0, 0)'

    width = 0.5

    trace_no_cont = go.Bar(
        x = quantile,
        y = y_no_cont,
        text = y_text_no_cont,
        textposition = 'no',
        name = 'No Contagion',
        marker=dict(
            color=light_blue_inside,
            line=dict(
                color=light_blue_line,
                width=1.5),
            ),
        width = width,
        opacity=0.6
        )

    y_cont = [max(int(UL_c[k]-UL[k]),0) for k in range(len(UL))]
    y_text_cont = ["{:,}".format(int(u)) for u in UL_c]

    trace_cont = go.Bar(
        x = quantile,
        y = y_cont,
        text = y_text_cont,
        textposition = 'no',
        name = 'Contagion',
        marker=dict(
            color=red_inside,
            line=dict(
                color=red_line,
                width=1.5),
            ),
        width = width,
        opacity=0.6
        )

    data = [trace_no_cont, trace_cont]

    dic = {1:'14th Nov 2010 to 10th Sep 2011',
        2:'10th Sep 2011 to 6th July 2012',
        3:'6th July 2012 to 2nd May 2013',
        4:'2nd May 2013 to 26th Feb 2014',
        5:'26th Feb 2014 to 23th Dec 2014',
        6:'24th Oct 2014 to 14th Aug 2015'}

    color_text = 'rgba(25, 25, 25, 1)'

    layout = {
      'xaxis': {'title': 'Quantile', 'color': color_text},
      'yaxis': {'title': 'Capital ', 'color': color_text},
      'barmode': 'stack',
      'title': 'Capital Requirement from ' + dic[arg[3]]
    }

    annotations = []
    for i in range(len(UL_c)):
        annotations.append(dict(x=quantile[i], y=y_cont[i] + y_no_cont[i] + 8*10**4, text=y_text_cont[i],
                                  font=dict(family='Arial', size=10,
                                  color=color_text),
                                  showarrow=False,))
    for i in range(len(UL)):
        annotations.append(dict(x=quantile[i], y=y_no_cont[i] - 10**5, text=y_text_no_cont[i],
                                  font=dict(family='Arial', size=10,
                                  color=color_text),
                                  showarrow=False,))
    layout['annotations'] = annotations

    fig = go.Figure(data=data, layout = layout)

    if arg:
        path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods'
        folder = 'time_'+str(arg[0])+'_delay_'+str(arg[1])+'_per_'+str(arg[2])
        if not os.path.isdir(os.path.join(path,folder)):
            os.makedirs(os.path.join(path,folder))
        figure_name = 'Period_'+str(arg[3])+ '.png'
        py.image.save_as(fig,os.path.join(path, folder,
                figure_name))
    else:
        py.plot(fig, filename = 'Capital Requirement')
        plt.show()
    plt.close('all')

    # Figure with the percentage in difference

    y_perc = [int((UL_c[k]-UL[k])/UL[k]*10**4)/100 for k in range(len(UL))]
    y_per_text = [str(u) + r'%' for u in y_perc]
    trace_perc = go.Bar(
        x = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'],
        y = y_perc,
        text =y_per_text,
        textposition = 'on',
        name = 'Contagion',
        marker=dict(
            color=blue_inside,
            line=dict(
                color=blue_line,
                width=1.5),
            ),
        width = width,
        opacity=0.6
        )

    data = [trace_perc]

    layout = {
      'xaxis': {'title': 'Quantile', 'color': color_text},
      'yaxis': {'title': 'Percentage ', 'color': color_text},
      'barmode': 'stack',
      'title': r'Difference in % of capital requirement from ' + dic[arg[3]]
    }

    annotations = []
    for i in range(len(UL_c)):
        annotations.append(dict(x=quantile[i], y=y_perc[i]+ 0.6, text=y_per_text[i],
                                  font=dict(family='Arial', size=10,
                                  color=color_text),
                                  showarrow=False,))

    layout['annotations'] = annotations

    fig = go.Figure(data=data, layout = layout)

    if arg:
        path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods'
        folder = 'time_'+str(arg[0])+'_delay_'+str(arg[1])+'_per_'+str(arg[2])
        if not os.path.isdir(os.path.join(path,folder)):
            os.makedirs(os.path.join(path,folder))
        figure_name = 'Difference_period_'+str(arg[3])+ '.png'
        py.image.save_as(fig,os.path.join(path, folder,
                figure_name))
    else:
        py.plot(fig, filename = 'Difference Capital Requirement')
        plt.show()
    plt.close('all')

def plot_three_comp(Uinitial,time):
    Uarray = [Uinitial[0]]

    for i in range(1,len(Uinitial)):
        U1 = Uinitial[i-1]
        U2 = Uinitial[i]
        Uarray.append([max(int(U2[k]-U1[k]),0) for k in range(len(U1))])

    quantile = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%']
    names = ['No Contagion','Contagion 1 day lag',
        'Contagion 2 day lag','Contagion 3 day lag']


    light_blue_inside = 'rgb(158,202,225)'
    light_blue_line = 'rgb(8,48,107)'
    blue_inside = 'rgb(58,200,225)'
    blue_line = 'rgb(8,48,107)'
    red_inside = 'rgb(220, 20, 60)'
    red_line = 'rgb(139, 0, 0)'


    colors_inside = ['rgb(158,202,225)','rgb(58,200,225)',
                    'rgb(51,255,51)','rgb(255, 51, 51)']
    colors_line = ['rgb(8,48,100)','rgb(0,0,255)',
                    'rgb(0, 255, 0)','rgb(255, 0, 0)']

    color_text = 'rgb(0,0,0)'

    width = 0.5

    traces = []
    annotations = []

    for i in range(len(Uarray)):
        y = [int(u) for u in Uarray[i]]
        y_init = [int(u) for u in Uinitial[i]]
        y_text = ["{:,}".format(int(u)) for u in y_init]

        traces.append(go.Bar(
            x = quantile,
            y = y,
            text = y_text,
            textposition = 'auto',
            textfont = dict(
                family = 'Arial',
                color= color_text,
                size = 10), 
            name = names[i],
            marker=dict(
                color=colors_inside[i],
                line=dict(
                    color='rgb(8,48,100)',
                    width=1.5),
                ),
            width = width,
            opacity=0.6
            )
        )

    layout = {
      'xaxis': {'title': 'Quantile', 'color': color_text},
      'yaxis': {'title': 'Capital ', 'color': color_text},
      'barmode': 'stack',
      'title': 'Capital Requirement from different delays',
      'width':800, 'height':640
    }

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

    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods'
    folder = 'cap_comparisons'
    if not os.path.isdir(os.path.join(path,folder)):
        os.makedirs(os.path.join(path,folder))
    figure_name = 'Time_'+str(time)+ '.png'
    py.image.save_as(fig,os.path.join(path, folder,
            figure_name))

    # py.plot(fig, filename = 'Capital Requirement')
    # plt.show()

    plt.close('all')

def plot_three_diff(Uinitial,time):

    Uarray = [Uinitial[0]]

    for i in range(1,len(Uinitial)):
        U1 = Uinitial[i-1]
        U2 = Uinitial[i]
        Uarray.append([max(int(U2[k]-U1[k]),0) for k in range(len(U1))])

    quantile = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%']
    names = ['No Contagion','Contagion 3 day lag',
        'Contagion 2 day lag','Contagion 3 day lag','Contagion CountryRank']


    light_blue_inside = 'rgb(158,202,225)'
    light_blue_line = 'rgb(8,48,107)'
    blue_inside = 'rgb(58,200,225)'
    blue_line = 'rgb(8,48,107)'
    red_inside = 'rgb(220, 20, 60)'
    red_line = 'rgb(139, 0, 0)'


    colors_inside = ['rgb(158,202,225)','rgb(58,200,225)',
                    'rgb(51,255,51)','rgb(255, 51, 51)', 'rgb(255,51,255)']
    colors_inside = ['rgb(158,202,225)','rgb(255, 51, 51)']

    color_text = 'rgb(0,0,0)'

    width = 0.6

    traces = []
    annotations = []

    for i in range(len(Uarray)):
        y = [int(u) for u in Uarray[i]]
        y_init = [int(u) for u in Uinitial[i]]
        y_text = [str(int(u/10**4)/10**2)+'M' for u in y_init]
        # y_text = [str(int(u/10**4)/10**2)+'M' for u in y]#"{:,}".format(int(u)) 

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
      title = 'Capital Requirement from contagion',
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

    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods'
    folder = 'cap_comparisons_diff'
    if not os.path.isdir(os.path.join(path,folder)):
        os.makedirs(os.path.join(path,folder))
    if len(Uarray) == 5:
        figure_name = 'Time_diff_'+str(time)+ '_comp_CountryRank.png'
    else:
        figure_name = 'Time_diff_only_delay_3_'+str(time)+ '.png'
    py.image.save_as(fig,os.path.join(path, folder,
            figure_name))

    # py.plot(fig, filename = 'Capital Requirement')
    # plt.show()

    plt.close('all')

def main():
    index = ['10_1','10_2','10_3','10_4','10_5','10_6','15_1',
        '15_2','15_3','15_4','15_5','15_6','20_1','20_2','20_3',
        '20_4','20_5','20_6']
    df = pd.DataFrame(0.0, index = index,
        columns = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'])
    df_c = pd.DataFrame(0.0, index = index,
        columns = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'])
    delay = 1
    percent = 0
    for time_para in [10,15,20]:
        for period in range(1,7):
            UL, UL_c = simulation_v2(time_para, delay, percent, period)
            df.loc[str(time_para) + '_' + str(period),:] = UL
            df_c.loc[str(time_para) + '_' + str(period),:] = UL_c
            print("UL:   ",UL)
            print("UL_c: ",UL_c)
            # plot_result(UL, UL_c, time_para, delay, percent, period)
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods'
    folder = 'results_delay_'+str(delay)+'_per_'+str(percent)
    if not os.path.isdir(os.path.join(path,folder)):
        os.makedirs(os.path.join(path,folder))
    df.to_csv(os.path.join(path,folder,'No_contagion_'+str(delay)+'_'+str(percent)+'.csv'))
    df_c.to_csv(os.path.join(path,folder,'Contagion_'+str(delay)+'_'+str(percent)+'.csv'))


    # print("EL_an: ",EL_an)
    # print("EL: ",EL)
    # print("EL_c: ",EL_c)

def main2():
    percent = 0
    index = ['10','10_1','10_2','10_3','15','15_1',
        '15_2','15_3','20','20_1','20_2','20_3']
    df = pd.DataFrame(0.0, index = index,
        columns = ['Q. 99%','Q. 99.5%', 'Q. 99.9%','Q. 99.99%'])
    U = []
    for time in [10,15,20]:
        U = []
        aux = compare_no_cont(time,1,percent)
        U.append(aux)
        df.loc[str(time),:] = aux
        print(time)
        for delay in [1,2,3]:
            aux = compare_cont(time,delay,percent)
            U.append(aux)
            df.loc[str(time)+'_'+str(delay)] = aux
        print(delay)
        plot_three_comp(U,time)
    df.to_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods\comp.csv')



def main_only_plot():
    percent = 0
    df = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods\comp.csv', index_col=0)
    UL_io = [1162329.05787076, 2972730.88912671, 5014999.11151231, 5713485.72490152]
    UL_io = [UL_io[0],UL_io[-1]]
    df = df.transpose()[['10','10_3','15','15_3','20','20_3']].transpose()
    for time in [10,15,20]:
        U = []
        for i in [i for i in df.index if str(time) in i]:
            U.append(df.loc[i,:].values)
        plot_three_diff(U,time)
        # if time == 10:
        #     U.append(UL_io)
        #     plot_three_diff(U,time)


if __name__ == '__main__':
    # UL_io = compute_ioannis_capital()
    # print(UL_io)
    main_only_plot()