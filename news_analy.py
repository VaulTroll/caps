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

"""
def t_softmax(p, n):

    a = np.array([p,-n])
    print(a)
    return np.exp(a) / np.sum(np.exp(a), axis=0)
"""



def get_emotions(stopwords):
    es = 1e-8
    test_n = 100 #분류할 기사 갯수

    for n in range(test_n): #여기서 test_n을 len(testList)로 두면 전체 기사 분석
        print("해당 기사 제목: ",titleList[n])
        pos_count =0
        nag_count=0
        pos= 0.0
        nag=0.0

        tokens = konlpy.tag.Mecab().morphs(testList[n])
        tokens = [word for word in tokens if not word in stopwords]
        for k in range(len(tokens)): #토큰의 갯수만큼 반복
            #해당 토큰과 유사단어들을 가져옴
            if tokens[k] in analy_model_cbow:
                sim= analy_model_cbow.most_similar(tokens[k])
            else: break

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

        print("긍정 value: ",pos)
        print("부정 value: ",nag)
        #print(pos_count)
        #print(nag_count)
        print("NP value :",pos+nag)
        print("NP 비율: ",pos_count/(pos_count+nag_count+es))
        print("해당 기사의 NP 지수:  ",pos/(pos-nag+es))
        #print((pos+nag)/(pos_count+nag_count))
        print('---------------------------------------------------------------------')



get_emotions(stopwords)
"""
감성사전을 만드는데 이용한 단어들
print(analy_model_sg.most_similar("악재"))
print(analy_model_sg.most_similar("부정"))
print(analy_model_sg.most_similar("힘든"))
print(analy_model_sg.most_similar("하락"))
print(analy_model_sg.most_similar("불안감"))
print(analy_model_sg.most_similar("악영향"))
print(analy_model_sg.most_similar("둔감"))
print(analy_model_sg.most_similar("불안"))
print(analy_model_sg.most_similar("하락"))
print(analy_model_sg.most_similar("급락"))
print(analy_model_sg.most_similar("하락세"))
print(analy_model_sg.most_similar("후폭풍"))

print('------------------------------------')

print(analy_model_sg.most_similar("호재"))
print(analy_model_sg.most_similar("긍정"))
print(analy_model_sg.most_similar("희소식"))
print(analy_model_sg.most_similar("상승"))
print(analy_model_sg.most_similar("기대감"))
print(analy_model_sg.most_similar("반등"))
print(analy_model_sg.most_similar("편안"))
print(analy_model_sg.most_similar("급등"))
print(analy_model_sg.most_similar("상승세"))
print(analy_model_sg.most_similar("기대"))
print(analy_model_sg.most_similar("촉매제"))
print(analy_model_sg.most_similar("기폭제"))
print(analy_model_sg.most_similar("급상승"))
print(analy_model_sg.most_similar("관심"))
print(analy_model_sg.most_similar("올랐"))
print(analy_model_sg.most_similar("낭보"))
"""



