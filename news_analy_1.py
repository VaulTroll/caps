import re
import sys
import os
import gensim
import math
import numpy as np
import pandas as pd
import nltk
import konlpy

from datetime import date
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from news_nlp import load_stopwords
from news_nlp import news_preprocess_stock
from news_nlp import news_tokenize
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.sentiment_analy import *
#np.seterr(all='warn')

#감성사전의 단어 불러오기
#good_emotion, bad_emotion에 저장되어 있다

#konlpy.jvm.init_jvm()

#저장된 w2v 불러오기
analy_model_cbow=KeyedVectors.load("w2v_cbow.kv")
analy_model_sg=KeyedVectors.load("w2v_sg.kv")
analy_model_sgns=KeyedVectors.load_word2vec_format("w2v_sgns.txt")

#학습 데이터를 저장할 리스트
data_list=[]

#주식종목->주식코드 연결
kospi_dict = {}
f = open("../data/kospi200_list.txt",'r',encoding="cp949")
lines = f.readlines()
for line in lines:
    a, b = line.split(':')
    kospi_dict[a] = b.strip()
f.close()

#주식코드->주식가격 연결
price_data = {}
fs = open("../data/kospi200_stock_1.txt",'r')
lines = fs.readlines()
c = 0
for line in lines:
    c+=1
    if(c%2==0):
        price_data[tmp]=line.strip()
    tmp=line.strip()
fs.close()

stopwords = load_stopwords()
timeList, stockList, testList, titleList = news_preprocess_stock('kospi200_news')

def get_emotions(stopwords):
    r_timeList = []
    r_npList = []
    r_codeList = []
    r_titleList = []
    es = 1e-8
    test_n = len(testList)#분류할 기사 갯수

    for n in range(test_n): #여기서 test_n을 len(testList)로 두면 전체 기사 분석
        stock_code = kospi_dict[stockList[n]]
        filt_code = ['001680','030200','000210','006260']
        if stock_code in filt_code:
            #print('filter')
            continue
    
        time_val = timeList[n]
        y = time_val[:4]
        m = time_val[4:6]
        d = time_val[6:]
        if date(int(y),int(m),int(d)).weekday() > 4: #기사가 나온 시점이 주말
            #print('weekend')
            continue
        
        pos_count=0
        nag_count=0
        t_pos=0.0
        t_nag=0.0
        np_val=0.0
        tsim=[]
        num=0
        
        
        tokens = konlpy.tag.Mecab().morphs(testList[n])
        tokens = [word for word in tokens if not word in stopwords]


        for k in range(len(tokens)): #토큰의 갯수만큼 반복
        
            #해당 토큰이 감성사전에 들어있는지 확인
            if tokens[k] in good_emotion:
                t_pos+=1
                pos_count+=1
            if tokens[k] in bad_emotion:
                t_nag-=1
                nag_count-=1
                
            #해당 토큰과 유사단어들을 가져옴
            if tokens[k] in analy_model_sgns:
                tsim = analy_model_sgns.most_similar(tokens[k])
                for s in range(10):
                    if tsim[s][0] in good_emotion:
                        t_pos += tsim[s][1]
                    if tsim[s][0] in bad_emotion:
                        t_nag -= tsim[s][1]

            #skip-gram 과 cbow는 가장 유사한 단어가 반대어가 나오는 경우가 빈번하여 첫 유사단어 제외
            
            if tokens[k] in analy_model_sg:
                tsim = analy_model_sg.most_similar(tokens[k])
                for s in range(9):
                    if tsim[s+1][0] in good_emotion:
                        t_pos += tsim[s+1][1]
                    if tsim[s+1][0] in bad_emotion:
                        t_nag -= tsim[s+1][1]
            
            if tokens[k] in analy_model_cbow:
                tsim = analy_model_cbow.most_similar(tokens[k])
                for s in range(9):
                    if tsim[s+1][0] in good_emotion:
                        t_pos += tsim[s+1][1]
                    if tsim[s+1][0] in bad_emotion:
                        t_nag -= tsim[s+1][1]
                        
                        
        np_val = t_pos+t_nag
        if abs(np_val) < 5: #np값이 너무 작으면
            continue
        
        r_timeList.append(timeList[n])
        r_npList.append(np_val)
        r_codeList.append(stockList[n])
        r_titleList.append(titleList[n])
        
        print("해당 기사 제목: ",titleList[n])
        print("총 긍정 value 합: ",t_pos)
        print("총 부정 value 합: ",t_nag)
        #np_div = np_val/(pos_count+nag_count+es)
        print("NP value: ", np_val)
        if np_val ==0:
            print("np를 측정하지 못했습니다. ")
        #print('------------------------------------------------------------------')
        
        
    return r_timeList, r_npList, r_codeList, r_titleList

#######################################################

def get_price(time_val, stock_code, np_val):
    kd = kospi_dict[stock_code]
    tmp = price_data[kd]
    p_list = tmp.split(', ')
    for a in range(len(p_list)):
        if time_val in p_list[a]:
            new_data = p_list[a-2:a+2]
            
            check=0
            for d in new_data:
                input_np = '0'
                check+=1
                if check == 3: input_np = np_val
                d=str(d)
                d = d.replace('[','')
                d = d.replace(']','')
                time, open, high, low, end, volume, _ = d.split(',')
                
                #최종 학습 데이터 저장
                data_list.append([input_np, open, high, low, end, volume])
                    
                
            
##############################################################################
timeL, npL, codeL, titleL = get_emotions(stopwords)

for n in range(len(timeL)):
    get_price(timeL[n], codeL[n], npL[n])
    print(n)
#



df = pd.DataFrame(data_list, columns=['np_val','open','high','low','close','volume'])
df.to_csv('train_3.csv', index=False, encoding='cp949')





