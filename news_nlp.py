import re
import nltk
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import konlpy
import time
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import gensim
import pydot
import tensorflow as tf
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name = 'cpu')

from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

"""
path = jpype.getDefaultJVMPath()
jpype.startJVM(path)
if jpype.isJVMStarted():
    jpype.attachThreadToJVM()

"""
#tf.compat.v1.disable_eager_execution()
print(tf.__version__)
#konlpy.jvm.init_jvm()
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
    
    for line in lines:
        #긍정도 평가를 시각적으로 보기 위한 제목 추출
        title_l = line.split('\t')
        titleList.append(title_l[2])

        #실제 자연어처리를 위한 내용 추출
        line = re.sub('[^a-zA-Z가-힣,.↑↓+ ]','',line)
        line = re.sub('[,.]',' ',line)
        newsList.append(line)
    #print("총 기사 갯수 : ",len(newsList))
    f.close()
    return newsList, titleList
    
#############################################################
# 주식 뉴스 데이터 불러와서 전처리
def news_preprocess_stock(data_file_name):
    newsList =[]
    titleList =[]
    timeList = []
    stockList = []
    file_path = '../data/' + data_file_name +'.txt'
    f = open(file_path ,'r',encoding='utf-8')
    lines = f.readlines()
    
    for line in lines:
        #긍정도 평가를 시각적으로 보기 위한 제목 추출
        title_l = line.split('\t')
        if(len(title_l)==5):
            timeS = re.sub('-','',title_l[0])
            timeList.append(timeS)
            stockList.append(title_l[2])
            titleList.append(title_l[3])

            #실제 자연어처리를 위한 내용 추출
            line = re.sub('[^a-zA-Z가-힣,.↑↓+ ]','',line)
            line = re.sub('[,.]',' ',line)
            newsList.append(line)
    
    print("총 기사 갯수 : ",len(newsList))
    f.close()
    
    return timeList, stockList, newsList, titleList

############################################################
# Mecab를 이용한 형태소 분류, 불용어 제거

def news_tokenize(newsList, stopwords):
    tokenized_data=[]

    for sentence in newsList:
        tokens = konlpy.tag.Mecab().morphs(sentence)
        tokens = [word for word in tokens if not word in stopwords]
        tokenized_data.append(tokens)

    return tokenized_data

##########################
def tfidf(newsList, stopwords):
    tfidfv = TfidfVectorizer(min_df=10).fit(newsList)
    a = tfidfv.transform(newsList).toarray()
    b = tfidfv.vocabulary_
    c= sorted(tfidfv.vocabulary_.items())
    print(c)



##################################################################
if __name__=="__main__":

    data_file = 'naver_data'
    stopwords = load_stopwords()
    newsList, _ = news_preprocess(data_file)

    #newsList1 = newsList[:9999]

    token_data = news_tokenize(newsList, stopwords)

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(token_data)

    word2idx = tokenizer.word_index
    idx2word = {value : key for key, value in word2idx.items()}
    encoded = tokenizer.texts_to_sequences(token_data)

    vocab_size = len(word2idx) + 1
    print('단어 집합의 크기 :', vocab_size)

    skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]
    pairs, labels = skip_grams[0][0], skip_grams[0][1]
    for i in range(5):
        print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
            idx2word[pairs[i][0]], pairs[i][0],
            idx2word[pairs[i][1]], pairs[i][1],
            labels[i]))
    embed_size = 100
    # 중심 단어를 위한 임베딩 테이블
    w_inputs = Input(shape=(1, ), dtype='int32')
    word_embedding = Embedding(vocab_size, embed_size)(w_inputs)

    # 주변 단어를 위한 임베딩 테이블
    c_inputs = Input(shape=(1, ), dtype='int32')
    context_embedding  = Embedding(vocab_size, embed_size)(c_inputs)

    dot_product = Dot(axes=2)([word_embedding, context_embedding])
    dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
    output = Activation('sigmoid')(dot_product)

    model = Model(inputs=[w_inputs, c_inputs], outputs=output)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    #plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    for epoch in range(1, 6):
        #loss = 0
        for _, elem in enumerate(skip_grams):
            first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
            second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
            labels = np.array(elem[1], dtype='int32')
            X = [first_elem, second_elem]
            Y = labels
            #loss += model.fit(X,Y)
            model.fit(X,Y)
        print('Epoch :',epoch)


    f = open('vectors.txt' ,'w')
    f.write('{} {}\n'.format(vocab_size-1, embed_size))
    vectors = model.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()

    w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
    print(w2v.most_similar(positive=['악재']))
    print(w2v.most_similar(positive=['상승']))


"""
    corpus= Corpus()
    corpus.fit(token_data, window=5)
    glove = Glove(no_components=100, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    print(glove.most_similar("악재"))
"""

    #modelsg = Word2Vec(sentences = token_data,  min_count = 30, workers = 4, sg=1)
    #modelcbow = Word2Vec(sentences = token_data, min_count = 30, workers = 4, sg =0)

    #print(modelsg.wv.most_similar("악재"))
    #print(modelcbow.wv.most_similar("악재"))
    #print(modelsg.wv.vectors.shape)
    #print(modelcbow.wv.vectors.shape)
########################################################
# 100개의 뉴스기사를 통해 여러 toknizer 비교
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
"""
# 비교 결과 OpenKoreanText와 MeCab-Ko가 괜찮은 분류 정확도와 빠른 속도를 보여줌
# MeCab-ko 를 이용하기로 결정


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
    token_data = news_tokenize(newsList1, stopwords)

    modelsg = Word2Vec(sentences = token_data, min_count = 30, workers = 4, sg=1)
    modelcbow = Word2Vec(sentences = token_data, min_count = 30, workers = 4, sg =0)

    print(modelsg.wv.most_similar("악재"))
    print(modelcbow.wv.most_similar("악재"))
"""
############################################################
