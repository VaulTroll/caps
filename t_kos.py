import csv
f = open('kospi200.csv','r',encoding='cp949')
fs = open("kospi200_list.txt",'w',encoding="cp949")

data = csv.reader(f)
for line in data:
    code = line[1] + ":" + line[0] + "\n"
    fs.write(code)

fs.close()
f.close()
