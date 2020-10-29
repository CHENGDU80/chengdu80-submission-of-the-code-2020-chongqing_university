# -*- coding: utf-8 -*-
# @Time    : 20:54 2020/10/27 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : create_csv.py
import os
import pysentiment as ps

# ----------------read files-------------------
company_names = list()
with open('./company_names.txt', 'r', encoding='utf8') as fr:
    for line in fr:
        company_names.append(line.strip().lower())

industry_names = list()
with open('./industry.txt', 'r', encoding='utf8') as fr:
    for line in fr:
        industry_names.append(line.strip().lower())

path_prefix = '../dataset/2012_financial_news/'

# ----------------processing-----------------------
data = list()
for date_str in os.listdir(path_prefix):
    print(date_str)
    dir_name = os.path.join(path_prefix, date_str)

    for news_title in os.listdir(dir_name):
        file_path = os.path.join(dir_name, news_title)

        news_content = list()
        with open(file_path, 'r', encoding='utf8') as fr:
            for line in fr:
                if line.startswith('--'):
                    continue

                news_content.append(line)

        news_content = ' '.join(news_content)

        # remove irrelevant content
        position = news_content.find('To contact')
        if position != -1:
            news_content = news_content[:position]

        position = news_content.find('Read more')
        if position != -1:
            news_content = news_content[:position]

        news_content = news_content.strip()

        # -------------news sentiment--------------------
        hiv4 = ps.HIV4()

        words = hiv4.tokenize(news_content)
        # 将词语列表words传入hiv4.get_score，得到得分score
        score = hiv4.get_score(words)

        # --------  --company & industry info-------------
        cur_company = 'None'
        for one_company in company_names:
            if one_company in news_content.lower():
                cur_company = one_company
                break

        cur_industry = 'None'
        for one_industry in industry_names:
            if one_industry in news_content.lower():
                cur_industry = one_industry
                break

        data.append((date_str, news_title, str(score['Polarity']), str(score['Subjectivity']), cur_company, cur_industry))

# save file
with open('./news_sentiment.tsv', 'w', encoding='utf8') as fw:
    for one_line in data:
        fw.write('\t'.join(one_line) + '\n')
