import os
import re
import random

####################     part 1 划分训练集和测试集   ###################
def is_chinese(contents):
    """判断一个unicode是否是汉字"""
    zhmodel = re.compile(u'[\u4e00-\u9fa5]')   
    if zhmodel.search(contents):
        return True
    else:
        return False

def to_pure_chinese(content):
    """过滤掉非中字符"""
    #content = unicode(content,'gkb')
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str+i
    return content_str

def loadFile(filePath):
    X_content = []          # X_content存储各个sample的内容
    Y = []
    if filePath == "data/女性":
        y_value = 1             # y_value的含义：1-女性；2-体育；3-文学出版；4-校园
    elif filePath == "data/体育":
        y_value = 2
    elif filePath == "data/文学出版":
        y_value = 3
    else:
        y_value = 4    
    Doc_names =  os.listdir(filePath)
    #print("number of documents = ", len(Doc_names))
    for item in Doc_names:
        try:
            doc = open(filePath+str('/')+item ,'r',encoding='gbk')
        except:
            print("got exception when combining file address")
            print(item)
            continue
            
        try:
            content = doc.read()     # 文本内容
            content_str=to_pure_chinese(content)
            #print(content)
            #print(content_str)
        except:
            print("got exception when reading document content")
            print(item)
            continue
        X_content.append(content_str)
        Y.append(y_value)
        doc.close()
    return X_content,Y

loadFile("data/女性")
def combine_list(O,A):
    for item in A:
        O.append(item)
    return O


Xraw,Y = loadFile("data/校园")
Xraw_new,Y_new = loadFile("data/女性")
Xraw = combine_list(Xraw,Xraw_new)
Y = combine_list(Y,Y_new)

Xraw_new,Y_new = loadFile("data/文学出版")
Xraw = combine_list(Xraw,Xraw_new)
Y = combine_list(Y,Y_new)

Xraw_new,Y_new = loadFile("data/体育")
Xraw = combine_list(Xraw,Xraw_new)
Y = combine_list(Y,Y_new)

'''print(len(Xraw))
print(len(Y))
print(Xraw[0])
print(Y[0])'''

from sklearn.model_selection import train_test_split
Xraw_train,Xraw_test,Y_train,Y_test=train_test_split(Xraw,Y,test_size=0.1)
'''print('number of Xraw_train: ',len(Xraw_train))
print('number of Y_train: ',len(Y_train))
print('number of Xraw_test: ',len(Xraw_test))
print('number of Y_test: ',len(Y_test))

print(Xraw_test[0])
print(Y_test[0])'''

####################     part 2 生成关键字字典（特征）   ###################
def dict_keywords(Xraw_s):
    """统计X_raw中汉字的种数及每种字出现的字数"""
    keywords={}
    for item in Xraw_s:
        for jj in item:
            if jj in keywords:
                #print("item = ",item)
                keywords[jj]=keywords[jj]+1
            else:
                keywords[jj]=1

    return keywords
    
keywords=dict_keywords(Xraw_train)
print("the length of keywords = ",len(keywords))

####################     part 3 创建特征向量（即把Xraw_train,Xraw_test变成X_train,X_test）      ###################
def find_index_in_dict(item,dict_):
    index=0
    for ii in dict_.keys():
        if ii == item:
            break
        else:
            index = index +1
    return index

def convert_to_feature_vec(Xraw_item):
    feature_vec=[0 for x in range(len(keywords))]
    for zh_char in Xraw_item:
        if zh_char in keywords:
            index=find_index_in_dict(zh_char,keywords)
            feature_vec[index]=feature_vec[index]+1
        '''else:
            print(zh_char,":this Chinese word is not included in keywords")'''
    return feature_vec

X_train=[]
for item in Xraw_train:
    X_train.append(convert_to_feature_vec(item))

X_test=[]
for item in Xraw_test:
    X_test.append(convert_to_feature_vec(item))

#print(X_train[0])
#print(X_test[0])

####################     part 4 朴素贝叶斯训练      ###################
print("== training ==") 
from sklearn.naive_bayes import MultinomialNB   #options:MultinomialNB,BernoulliNB,GaussianNB
clf =MultinomialNB()
clf.fit(X_train, Y_train)
####################     part 5 预测               ###################
print("== Predict result ==") 
from sklearn.metrics import accuracy_score
print('accuracy = ',accuracy_score(Y_test,clf.predict(X_test)))
