import matplotlib.pyplot as plt
import pandas as pd
import functions
import importlib
importlib.reload(functions)
from functions import *

df = pd.read_excel('stockdata.xlsx')
df_1= df.dropna(thresh=3000,axis=1)
df_1= df_1.dropna()
df_1.set_index('Symbol Name', inplace=True)

training, testing = mean_cov_train_test(df_1,'2014-6','2017-6', df_1.columns[:200])
rc_value = rc(training,0.04)

lambdas = np.arange(0,100)*0.01

std_list = []
sharpe_list_outfunction = []
for i in lambdas:
    std_list.append(cross_validation(rc_value, i,training)[0])
    sharpe_list_outfunction.append(cross_validation(rc_value, i, training)[1])


plt.figure()
plt.subplot(1,2,1)
plt.plot(lambdas,std_list)
plt.subplot(1,2,2)
plt.plot(lambdas, sharpe_list_outfunction)
plt.clf()
np.cumsum(testing.mean(axis=1)).plot()
testing.mean(axis=1).mean()/testing.mean(axis=1).std()