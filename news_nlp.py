import re
import nltk
import konlpy
import time
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import gensim
from collections import Counter

"""
path = jpype.getDefaultJVMPath()
jpype.startJVM(path)
if jpype.isJVMStarted():
    jpype.attachThreadToJVM()

"""

konlpy.jvm.init_jvm()
#nltk.download('punkt')

#############################################################
# 불용어 불러오기

def load_stopwords():
    stopwords = []
    f = open('../data/stopwords.txt','r',encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        stopwords.append(line.strip())
    f.close()
    return stopwords

##############################################################
# 뉴스 데이터 불러오기 및 간단한 전처리

def news_preprocess(data_file_name):
    newsList =[]
    titleList =[]
    file_path = '../data/' + data_file_name +'.txt'
    f = open(file_path ,'r',encoding='utf-8')
    lines = f.readlines()
    #title_l=lines.split('\t')

    #titleList.append(title_l[2])
    for line in lines:
        #긍정도 평가를 시각적으로 보기 위한 제목 추출
        title_l = line.split('\t')
        titleList.append(title_l[2])

        #실제 자연어처리를 위한 내용 추출
        line = re.sub('[^a-zA-Z가-힣,. ]','',line)
        line = re.sub('[,.]',' ',line)
        newsList.append(line)
    print("총 기사 갯수 : ",len(newsList))
    f.close()
    return newsList, titleList

############################################################
# Mecab를 이용한 형태소 분류, 불용어 제거

def news_tokenize(newsList, stopwords):
    tokenized_data=[]
    newsList = newsList

    for sentence in newsList:
        tokens = konlpy.tag.Mecab().morphs(sentence)
        tokens = [word for word in tokens if not word in stopwords]
        tokenized_data.append(tokens)
    return tokenized_data

##################################################################
if __name__=="__main__":

    data_file = 'naver_data'
    stopwords = load_stopwords()
    newsList, _ = news_preprocess(data_file)

    newsList1 = newsList[:99] #100개의 데이터로 테스트가능
    tokenized_data = news_tokenize(newsList1, stopwords)

########################################################
# 10개의 뉴스기사를 통해 여러 toknizer 비교
"""
    tokenizers = [
        {'name': 'KoNLPy Hannanum','tokenizer': konlpy.tag.Hannanum().morphs},
        {'name': 'KoNLPy Kokoma','tokenizer':konlpy.tag.Kkma().morphs},
        {'name': 'KoNLPy Komoran','tokenizer':konlpy.tag.Komoran().morphs},
        {'name': 'KoNLPy OpenKoreanText','tokenizer':konlpy.tag.Okt().morphs},
        {'name': 'KoNLPy MeCab-Ko','tokenizer':konlpy.tag.Mecab().morphs},
    ]
    for nlist in newsList1:
        for tw in tokenizers:
            start = time.time()
            tokens = tw['tokenizer'](nlist)
            elapse = time.time() - start
            tokens = [word for word in tokens if not word in stopwords]
            print(f"### {tw['name']} (time: {elapse:.3f} [sec])")
            #print(tokens)
        print('--------------------------------------------')

# 비교 결과 OpenKoreanText와 MeCab-Ko가 괜찮은 분류 정확도와 빠른 속도를 보여줌
# MeCab-ko 를 이용하기로 결정
"""

############################################################
# preprocessing 후의 word count
"""
    count_list = [elem for arr in tokenized_data for elem in arr]

    count = Counter(count_list)
    count_data = count.most_common(100)
    print(count_data)
"""
###########################################################
#Word2Vec 를 통해 단어들을 벡터화 후 저장
#sg=1 이면 skip-gram, sg=0 이면 cbow, 최소단어 30

"""
    modelsg = Word2Vec(sentences = tokenized_data, min_count = 30, workers = 4, sg=1)
    modelcbow = Word2Vec(sentences = tokenized_data, min_count = 30, workers = 4, sg =0)
    print("Vec done")

    modelsg.wv.save('w2v_sg.kv')
    modelcbow.wv.save('w2v_cbow.kv')
    print('save complet')

    print(modelsg.wv.most_similar("악재"))
    print(modelcbow.wv.most_similar("악재"))
"""
############################################################
