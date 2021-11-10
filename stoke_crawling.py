import requests
import time
import re
import datetime
from bs4 import BeautifulSoup

stoke_code_list=[]
f = open("../data/kospi200_list.txt",'r',encoding="cp949")
lines = f.readlines()
for line in lines:
    a, b = line.split(':')
    stoke_code_list.append(b.strip())
f.close()

start_date = '20150101'
end_date = '20211026'


def stock_craw_def():
    count = 0

    for stoke_code in stoke_code_list:
        base_raw = requests.get("https://api.finance.naver.com/siseJson.naver?symbol="+stoke_code+"&requestType=1&startTime="+start_date+"&endTime="+end_date+"&timeframe=day",headers={'User-Agent':'Mozilla/5.0'})
            
            
                    #기사를 파일에 저장
        f=open('../data/kospi200_stock.txt','a',encoding='utf-8')
        f.write(stoke_code+":")
        f.write(base_raw.text)
        f.write('\n')
        f.close()
            
        count+=1
        print(stoke_code + ": is collected || count : " + str(count))
            

stock_craw_def()
