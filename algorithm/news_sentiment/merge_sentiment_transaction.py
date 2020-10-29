# -*- coding: utf-8 -*-
# @Time    : 21:40 2020/10/27 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : merge_file.py
import numpy as np

day_sentiment = dict()
with open('./news_sentiment.tsv', 'r', encoding='utf8') as fr:
    for line in fr:
        splited_line = line.strip().split('\t')
        day = splited_line[0]
        day = day.replace('-', '/')
        polarity = float(splited_line[2])
        subjectivity = float(splited_line[3])

        if day not in day_sentiment:
            day_sentiment[day] = list()

        day_sentiment[day].append((polarity, subjectivity))

day_avg_sentiment = dict()
for day, v in day_sentiment.items():
    data_numpy = np.array(v)
    data_numpy = np.mean(data_numpy, axis=0)

    avg_polarity, avg_subjectivity = data_numpy[0], data_numpy[1]

    day_avg_sentiment[day] = [avg_polarity, avg_subjectivity]


merged_data = list()
with open('./transaction_data.tsv', 'r', encoding='utf8') as fr:
    is_first = True
    for line in fr:
        if is_first:
            is_first = False
            continue

        splited_line = line.split('\t')

        day = splited_line[0].strip()

        try:
            splited_line.extend(day_avg_sentiment[day])
        except Exception as e:
            print(day)

        merged_data.append(splited_line)

with open('merged_sentiment.tsv', 'w', encoding='utf8') as fw:
    first_line = ['date', '	TICKER', 'COMNAM', 'BIDLO', '	ASKHI', 'OPENPRC', 'PRC', 'VOL', 'SHROUT']
    first_line.extend(['POLARITY', 'SUBJECTIVITY'])
    fw.write('\t'.join(first_line).strip() + '\n')

    for line in merged_data:
        fw.write('\t'.join([str(a).strip() for a in line]).strip() + '\n')
