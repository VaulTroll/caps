import requests
import time
import re
import datetime
from bs4 import BeautifulSoup
#from selenium import webdriver

kospi_list=[]
f = open("../data/kospi200_list.txt",'r',encoding="cp949")
lines = f.readlines()
for line in lines:
    a, b = line.split(':')
    kospi_list.append(a)
f.close()
#driver = webdriver.Chrome('/Users/vaultroll/Desktop/caps/chromedriver')
count=0
to_date = '20150420'
date_form = '%Y%m%d'

naver_basic_url ="https://finance.naver.com"
naver_craw_url = "https://finance.naver.com/news/news_list.nhn?mode=LSS2D&section_id=101&section_id2=258&date="


def text_to_line(date, time, kospi, title, content):
    #크롤링한 내용을 하나로 합쳐 탭으로 항목 구분
    news_data = '{0}\t{1}\t{2}\t{3}\t{4}'.format(date,time,kospi,title,content)
    return news_data

def iter_date(date):
    #datetime을 이용해 하루 전 날짜를 계산
    yester_date = datetime.datetime.strftime(datetime.datetime.strptime(date, date_form) - datetime.timedelta(days=1), date_form)
    return yester_date


def naver_craw_def(date):
    page = 0
    count = 0
    check = 0

    while True:
        naver_data_list=[]
        check_reuters_list=[]
        page+=1
        print("page : ",page)

        base_raw = requests.get(naver_craw_url+str(date)+'&page='+str(page),headers={'User-Agent':'Mozilla/5.0'})
        html = BeautifulSoup(base_raw.text, "html.parser")

        #기사 중 로이터증권의 기사는 네이버가 밤 00:00에 자동으로 수집하는 짧은 외국 기사로 크롤링 양만 많아지고 학습에는 크게 도움이 되지 않아서 수집하지 않기 위해 체크 후 스킵
        check_reuters = html.select("dd.articleSummary > .press")
        for ch in check_reuters:
            check_reuters_list.append(ch.text)
        if "로이터 증권(신)" in check_reuters_list:
            break

        #네이버 실시간 뉴스 수집
        naver_newslists  = html.select("ul.realtimeNewsList > li")
        for newslist in naver_newslists:
            check = 0
            naver_data = newslist.select("dl > dt")

            for data in naver_data:
                news_url = data.select_one("a")['href'] #해당 페이지의 url 가져오기
                news_url_r = re.sub('§','&sec',news_url) #&sec가 자동으로 변환되어url이 작동하지 않기에 치환

                each_raw = requests.get(naver_basic_url+news_url_r, headers = {"User-Agent" : "Mozilla/5.0"})
                each_html = BeautifulSoup(each_raw.text, 'html.parser')

                naver_contents = each_html.select("#contentarea_left")

                for content in naver_contents:
                    n_title = content.select_one("div.article_info > h3").text.strip()
                    content_title = re.sub('[^a-zA-Z가-힣&]','',n_title)
                    for k_list in kospi_list:
                        m = re.search(k_list,content_title)
                        if m is not None:
                            n_kospi = m.group()


                            n_wdate = content.select_one("div.article_sponsor > span.article_date").text.strip()
                            n_contents = content.select_one("div#content")
                            #기사 내용 중 관련없는 추가 광고 뉴스 제거 작업
                            for a in n_contents.find_all('a'):
                                a.decompose()
                            for span in n_contents.find_all('span'):
                                span.decompose()
                            n_content = n_contents.text.strip()

                            n_date, n_time = n_wdate.split(' ')
                            n_data = text_to_line(n_date, n_time,n_kospi, n_title, n_content)
                            naver_data_list.append(n_data)
                            check+=1 #마지막 페이지인지 체크
                            count+=1 #하루에 총 수집한 기사 수 카운트
                            print(n_kospi, n_title)


        #기사를 파일에 저장
        f=open('../data/kospi200_news.txt','a',encoding='utf-8')
        for n in naver_data_list:
            f.write(n)
            f.write('\n')
        f.close()
        if check == 0: break

    return count

def naver_craw(date):
    ch_date = date
    n_count = 0
    while True:
        n_count = naver_craw_def(ch_date)
        print("now date : ", ch_date," || collected : ", n_count)
        ch_date = iter_date(ch_date)

    return 0



naver_craw(to_date)

