# OR-BASELINES
# This file contains OR Baselines, without the use of machine learning models

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# data loading and preparation
df = pd.read_csv('data/Turbine_Data.csv', parse_dates=True, index_col=0)
data_available = ~(df['ActivePower'].isna() | df['AmbientTemperatue'].isna() | df['WindDirection'].isna() | df['WindSpeed'].isna())
df = df[data_available][['ActivePower', 'AmbientTemperatue', 'WindDirection', 'WindSpeed']].reset_index()
train = df.iloc[:59533].set_index('index')
test = df.iloc[59533:].set_index('index')

# Operations Research Model for Energy Dispatch
C = np.array([15, 20, 15, 20, 30, 25])
def get_action(renewables):
    g = cp.Variable(6, nonneg=True)
    o = cp.Variable()
    capacity = np.array([1, 0.5, 1, 1, 1, 0.5])
    demand = 4

    objective = cp.Minimize(o)
    constraints = [
        o == C@g,
        g<=capacity,
        demand <= cp.sum(g) + renewables
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    return g.value

# evaluate a decision
def getcost(action, renewable_power):
    cost = action@C
    shortcoming = 4 - action.sum() - renewable_power
    if shortcoming>0:
        cost += shortcoming*100
    return cost, shortcoming

## BASELINE 1 : Using the whole training dataset for optimization
ermaction = get_action(np.array(train.ActivePower)/1000)
cs = [getcost(ermaction, p/1000) for p in test.ActivePower]
costs = np.array([c[0] for c in cs])
np.save('results/baseline_erm.npy', costs)
print('Cost with ERM: {}'.format(costs.mean()))
plt.figure()
plt.hist(costs, bins=50);
plt.title('Production Cost')
plt.savefig('plots/baseline_erm_cost.pdf')
plt.figure()
plt.hist([c[1] for c in cs])
plt.title('Production Shortfall')
plt.savefig('plots/baseline_erm_shortfall.pdf')


## BASELINE 2 : Using the last renewables production for optimization
for shift in [1,3,6]:
    df_hist = pd.concat([df, 
                         df.drop('index', axis=1).shift(shift).add_suffix('_1')],axis=1)
    train = df_hist.iloc[:59533].set_index('index')[[column for column  in df_hist.columns if 'ActivePower' in column]].dropna(axis=0)
    test = df_hist.iloc[59533:].set_index('index')[[column for column  in df_hist.columns if 'ActivePower' in column]]
    test = test[test.index.minute%(shift*10)==0]
    cs = [getcost(get_action(np.array(hist[1:]).flatten()/1000), now/1000) for now, hist in zip(test.ActivePower, test.drop('ActivePower', axis=1).iterrows())]
    costs = np.array([c[0] for c in cs])
    np.save(f'results/baseline_last10_shift{shift}.npy', costs)
    print('Cost with Last 10 mins: {}'.format(costs.mean()))
    plt.figure()
    plt.hist(costs, bins=50)
    plt.title('Production Cost')
    plt.savefig(f'plots/baseline_last10_cost_shift{shift}.pdf')
    plt.figure()
    plt.hist([c[1] for c in cs])
    plt.title('Production Shortfall')
    plt.savefig(f'plots/baseline_last10_shortfall_shift{shift}.pdf')


    ## BASELINE 3 : Using Oracle Information
    cs = [getcost(get_action(p/1000), p/1000) for p in test.ActivePower]
    costs = np.array([c[0] for c in cs])
    np.save(f'results/baseline_oracle_shift{shift}.npy', costs)
    print('Cost with Oracle: {}'.format(costs.mean()))
    plt.figure()
    plt.hist(costs, bins=50);
    plt.title('Production Cost')
    plt.savefig('plots/baseline_oracle_cost.pdf')
    plt.figure()
    plt.hist([c[1] for c in cs])
    plt.title('Production Shortfall')
    plt.savefig('plots/baseline_oracle_shortfall.pdf')