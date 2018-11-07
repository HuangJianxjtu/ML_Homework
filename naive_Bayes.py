def readCSV2List(filePath,title):
    try:
        file=open(filePath,'r')
        context = file.read()
        list_result=context.split("\n")
        length=len(list_result)
        list_result_final=[]
        if title==0:        # no title
        	title_result=None
        	for i in range(length-1):
        		list_result_final.append(list_result[i].split(","))
        else:
        	title_result=list_result[0].split(",")
        	for i in range(1,length-1):
        		list_result_final.append(list_result[i].split(","))
        return title_result,list_result_final
    except Exception :
        print("failure in reading files,please check the file path and the encoding type!")
    finally:
        file.close();

######################   step 1 :reading date  ###########################
Title,A=readCSV2List('career_data.csv',1)
print('=========== Original Data =============')
print('title= ',Title)
print('A= ',A)
################     step 2 :convert str to int      ################
#### first: 985
for i in range(len(A)):
	if A[i][0]=='Yes':
		A[i][0]=1
	else:
		A[i][0]=0
#### second:education
for i in range(len(A)):
	if A[i][1]=='bachlor':
		A[i][1]=1
	elif A[i][1]=='master':
		A[i][1]=2
	else:
		A[i][1]=3
#### third:skill
for i in range(len(A)):
	if A[i][2]=='C++':
		A[i][2]=1
	else:
		A[i][2]=2
#### fourth:enrolled
for i in range(len(A)):
	if A[i][3]=='Yes':
		A[i][3]=1
	else:
		A[i][3]=0

print('A=',A)

#### get X and Y
Y=[]
for i in range(len(A)):
	Y.append(A[i][len(A[i])-1])
	A[i].pop()
X=A
print('X=',X)
print('Y=',Y)

###################### step 3 : divide the data set into train set and test set  ###########
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
print('X_train',X_train)
print('Y_train',Y_train)
print('X_test',X_test)
print('Y_test',Y_test)

##############  step 4 :training     #################################
from sklearn.naive_bayes import GaussianNB   #options:MultinomialNB,BernoulliNB,GaussianNB
clf =GaussianNB()
clf.fit(X_train, Y_train)

##############  step 5 :predicting     ###########################
print("==Predict result==") 
print(clf.predict(X_test))
from sklearn.metrics import accuracy_score
print('accuracy = ',accuracy_score(Y_test,clf.predict(X_test)))
