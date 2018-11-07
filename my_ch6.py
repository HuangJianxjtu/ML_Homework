import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import  Counter
from sklearn.cross_decomposition import CCA
import matplotlib
import matplotlib.pyplot as plt

from DataGen import Get_Data_CLass1,Get_Data_CLass2

#################### Function Definition Zone #############################
def z_scores(x,axis):
	'''将数据标准化。标准化后数据的均值为(0),方差为1'''
	# x为数据，axis是作用方向
	x_certered = x-x.mean(axis=axis)        # shape:与x相同
	x_std = x_certered/np.std(x,axis=axis)  #np.std(),standard variance；shape:与x相同
	return x_std

def my_pca(x,num_low_dim):
	''' PAC算法'''
	# x为数据，num_low_dim是所要降低到的维度
	X_ = z_scores(x,axis=0)
	covMat = np.cov(X_,rowvar=False)              #rowvar=False表示每一列为一个维度，行中的元素为观察值; shape:(8,8)
	eigVals,eigVects = np.linalg.eig(covMat)      #eigVals.shape=(8,),eigVects=(8,8)
	eigValInd=np.argsort(eigVals)                 #排序，从小到大排列，并返回各个的索引;shape:(8,)
	eigValInd=eigValInd[:-(num_low_dim+1):-1]     #倒序取值。eigValInd中的特征值从大到小排列;shape:(num_low_dim,)
	W=eigVects[:,eigValInd]                       #shape:(8,num_low_dim)
	x_lowD=X_.dot(W)                              #shape:(1024,num_low_dim)
	return x_lowD

def my_lda_2class(X,Y):
	''' LDA算法,2类'''
	# X是输入，Y是标签,且Y只有0,1两类
	# 1. 计算每一个类的平均值
	u = [] 
	for i in range(2):#2 classes
		u.append(np.mean(X[Y==i], axis=0))
	# 2.计算类内散度矩阵Sw
	m,n = np.shape(X) 
	Sw = np.zeros((n,n))                         # shape=(n,n),即（8,8）
	for i in range(m):
		x_tmp = X[i].reshape(n,1) 
		if Y[i] == 0:
			u_tmp = u[0].reshape(n,1)
		if Y[i] == 1:
			u_tmp = u[1].reshape(n,1)
		Sw += np.dot( x_tmp - u_tmp, (x_tmp - u_tmp).T )
	# 3.计算类间散度矩阵Sb
	Sb = np.dot(u[0]-u[1],(u[0]-u[1]).T)         # shape=(n,n),即（8,8）
	# 4.计算矩阵S_w^(-1)*S_b
	S = np.dot(np.linalg.inv(Sw),Sb)             # shape=(n,n),即（8,8） 
	# 5.计算S_w^(-1)*S_w的最大的(d)个特征值和对应的(d个特征向量),得到投影矩阵W
	eigVals,eigVects = np.linalg.eig(S)
	eigValInd=np.argsort(eigVals)
	eigValInd=eigValInd[-1]
	W=eigVects[eigValInd]                        # shape:(n,1),即(8,1)
	# 6.得到输出的样本
	lowDData=X.dot(W)                            # shape:(m,1),即(2048,1)
	return lowDData

def plt_lda_2d(X,Y):
	'''绘制LDA降维（2类）之后的图形'''
	print('printing the graph')
	X0 = X[Y==0]
	X1 = X[Y==1]
	plt.scatter(X0,np.zeros(X0.shape[0]),c='b')   # 蓝色
	plt.scatter(X1,np.zeros(X1.shape[0]),c='r')   # 红色
	plt.show()

def my_cca(X,Y):
	''' CCA算法'''
	X = z_scores(X,axis=0)                       # shape:(1024,8)
	Y = z_scores(Y,axis=0)                       # shape:(1024,8)

	S = np.cov(X,Y,rowvar=False)                   # shape:(16,16)
	Sxx = S[0:8,0:8]                               # shape:(8,8)
	Sxy = S[0:8,8:16]                              # shape:(8,8)
	Syx = S[8:16,0:8]                              # shape:(8,8)
	Syy = S[8:16,8:16]                             # shape:(8,8)

	A=np.dot(np.dot(np.dot(np.linalg.inv(Sxx),Sxy),np.linalg.inv(Syy)),Syx)    # shape:(8,8)
	B=np.dot(np.dot(np.dot(np.linalg.inv(Syy),Syx),np.linalg.inv(Sxx)),Sxy)    # shape:(8,8)

	A_eigVals,A_eigVects = np.linalg.eig(A)
	A_eigValInd=np.argsort(A_eigVals)
	A_eigValInd=A_eigValInd[-1]
	a=A_eigVects[:,A_eigValInd]                  # shape:(8,)

	B_eigVals,B_eigVects = np.linalg.eig(B)
	B_eigValInd=np.argsort(B_eigVals)           
	B_eigValInd=B_eigValInd[-1]
	b=B_eigVects[:,B_eigValInd]                  # shape:(8,)

	rho =  np.dot(np.dot(a.T,Sxy),b)/(np.sqrt(np.dot(np.dot(a.T,Sxx),a))*np.sqrt(np.dot(np.dot(b.T,Syy),b)))  # correlation coefficient

	return rho,a,b

########################    Zone End     ##################################


X1 = Get_Data_CLass1()                         # shape:(1024,8)
X2 = Get_Data_CLass2()                         # shape:(1024,8)
#  将class1的label设置成0，将class2的label设置成1
Y1 = np.zeros(X1.shape[0])                     # shape:(1024,)
Y2 = np.ones(X2.shape[0])                      # shape:(1024,)

X1_std = z_scores(X1,axis=0)                   # shape:(1024,8)
X2_std = z_scores(X2,axis=0)                   # shape:(1024,8)
X = np.vstack((X1,X2))                         # shape:(2048,8)
X_std = np.vstack((X1_std,X2_std))             # shape:(2048,8)
Y = np.hstack((Y1,Y2))                         # shape:(2048,)


#########################    PCA Zone    ##################################
# numpy implementation of PCA
lowD_X1=my_pca(X1,num_low_dim=2)               # shape:(1024,2)
print('\n\nnumpy implementation:',lowD_X1.shape,'\n',lowD_X1)

# sklearn implementation of PCA
pca=PCA(n_components=2)
X1_2D= pca.fit_transform(X1_std)               # shape:(1024,2)
print('sklearn implementation:',X1_2D.shape,'\n',X1_2D)
########################    Zone End     ##################################


#########################    LDA Zone    ##################################
#numpy implementation of LDA
print('\n\nnumpy implementation of LDA')
X_my_lda = my_lda_2class(X,Y)                 # shape = (2048,1)
plt_lda_2d(X_my_lda,Y)
########################    Zone End     ##################################


########################    CCA Zone     ##################################
#numpy implementation of LDA
print('\n\nnumpy implementation of CCA')
rho,w,v = my_cca(X1,X2)
print('correlation coefficient =',rho)
print("projection vector of X1:\n",w)
print("projection vector of X2:\n",v)
########################    Zone End     ##################################