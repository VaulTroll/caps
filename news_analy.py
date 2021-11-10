import re
import sys
import os
import gensim
import math
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk
import konlpy
from news_nlp import load_stopwords
from news_nlp import news_preprocess
from news_nlp import news_tokenize
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.sentiment_analy import *
#감성사전의 단어 불러오기
#good_emotion, bad_emotion에 저장되어 있다

#konlpy.jvm.init_jvm()

#저장된 w2v 불러오기
analy_model_cbow=KeyedVectors.load("w2v_cbow.kv")
analy_model_sg=KeyedVectors.load("w2v_sg.kv")

stopwords = load_stopwords()
testList, titleList = news_preprocess('naver_data')



def get_emotions(stopwords):
    es = 1e-8
    test_n = 100 #분류할 기사 갯수

    for n in range(test_n): #여기서 test_n을 len(testList)로 두면 전체 기사 분석
        pos_count=0
        nag_count=0
        pos=0.0
        nag=0.0
        t_pos=0.0
        t_nag=0.0
        np_val=0.0
        np_div=0.0
        np_log =0.0

        tokens = konlpy.tag.Mecab().morphs(testList[n])
        tokens = [word for word in tokens if not word in stopwords]
        #print(tokens)
        for k in range(len(tokens)): #토큰의 갯수만큼 반복
            #해당 토큰과 유사단어들을 가져옴
            if tokens[k] in analy_model_cbow:
                sim= analy_model_cbow.most_similar(tokens[k])
            elif tokens[k] in analy_model_sg:
                sim = analy_model_sg.most_similar(tokens[k])
            else: break
            #print(sim)
            #print(tokens[k])

            #해당 토큰이 감성사전에 들어있는지 확인
            if tokens[k] in good_emotion:
                t_pos+=1
                pos+=1
                pos_count+=1
            if tokens[k] in bad_emotion:
                t_nag-=1
                nag-=1
                nag_count-=1


            #다음으로는 유사 단어들을 통해 계산
            #여기서 첫번째 유사단어는 제외한 이유는 대부분 첫 단어에는 반대 단어가 들어가있는 경우가 많기 때문
            for t in range(9):
                if sim[t+1][0] in good_emotion:
                    #print(sim[t][0])
                    pos += sim[t+1][1]
                    pos_count +=1
                if sim[t+1][0] in bad_emotion:
                    #print(sim[t][0])
                    nag -= sim[t+1][1]
                    nag_count +=1

        print("해당 기사 제목: ",titleList[n])
        print("총 긍정 value 합: ",pos)
        print("총 부정 value 합: ",nag)

        np_val = pos+nag
        np_div = np_val/(pos_count+nag_count+es)
        print("NP value: ", np_val)
        print("비교 위한 test NP: ", t_pos + t_nag)

        if np_val ==0:
            print("np를 측정하지 못했습니다. ")
            break

        if np_val<0:
            np_log = math.log(abs(np_val))
            np_log = -np_log
        else:
            np_log = math.log(abs(np_val))

        print("NP div: ", np_div) #값이 너무 작아짐
        print("np_log: ", np_log) #학습에 유리할지 고민
        print('---------------------------------------------------------------------')


#######################################################

get_emotions(stopwords)


