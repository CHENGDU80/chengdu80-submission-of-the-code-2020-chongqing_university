import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json


df=pd.read_csv('./data/transaction_data.tsv',delimiter='\t')

grouped_dfs=df.groupby('TICKER')

company2movement=dict()
for name,company_trans in grouped_dfs:
    company_trans.set_index('date',inplace=True)
    company_trans.sort_index(axis=0,ascending=False)

    movement_trend=list()

    prices=company_trans['PRC'].tolist()
    for idx,cur_price in enumerate(prices):
        if idx==0:
            movement_trend.append(0)
            continue

        change_rate=(cur_price-prices[idx-1])/prices[idx-1]

        movement_trend.append(change_rate)
    company2movement[name]=np.array(movement_trend,np.float32)

company_correlation=dict()
for source_company_name, source_representation in company2movement.items():
    correlation_list=list()
    for target_company_name,target_representation in company2movement.items():
        if len(source_representation)!=len(target_representation):
            continue
        correlation_list.append((target_company_name,spearmanr(source_representation,target_representation)[0]))
    correlation_list.sort(key=lambda x:x[1],reverse=True)
    company_correlation[source_company_name]=correlation_list

with open('company_correlation.json','w',encoding='utf8') as fw:
    fw.write(json.dumps(company_correlation))