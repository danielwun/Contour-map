import csv ;
import numpy as np ;
from math import exp ;
import matplotlib.pyplot as plt ;

def predictedArray( y,x,s ):
    # complete one of columns in design matrix 
    h = [] ;
    dim = 41 ;  #dimension of gaussian function
    sep = 1080/(dim-1) ;    #seperation of two gaussian function 
    h.append(1) ;
    for i in range(dim) :
        for j in range(dim) :
            h.append( exp( -1*pow((x-sep*(i)),2)/2/pow(s,2)-pow((y-sep*(j)),2)/2/pow(s,2)) ) ;    
    return h ;    

def makePrior( p, dm, variance, t ):
    tdm = list(zip(*dm)) ;
    sni= np.linalg.inv(p[1])+ np.dot(tdm,dm)/variance ;  #inverse matrix of Sn
    sni = np.linalg.inv(sni) ;
    
    mean = np.dot( np.linalg.inv(p[1]),p[0] ) ;
    mean = mean + np.dot(tdm,t)/variance ;
    mean = np.dot( sni, mean ) ;
    del p ;
    p = [mean,sni] ;
    return p ;

def errorFunction( dm, w, t, num, s ):
    # with 1/2 one
    lamb = s*s ;    #lambda
    y = np.dot( dm, w ) ;
    me = [ y[i]-t[i] for i in range(num)] ; #mean square error
    me = 0.5/num*sum(me[i]*me[i] for i in range(num)) ;
   # me = me + lamb/2*sum(w[i]*w[i] for i in range(len(w))) ;
    return me ;

    
# read information of coordinates 
# ----------------------main function----------------------------------
count = 0 ;
datanum = 40000; #   Data number
sdatanum = 100 ; #   how many datas in a set ;
dm = [] ;   #a set of design matrix
tdm = [] ;  #total dm  
betai = 25 ;  # 1/beta
alphai = 2500000000 ; # 1/alpha
mean = [ 0 for i in range(1682)] ;
print(alphai)
x = [] ;
f = open( 'X_train.csv' , 'r') ;    #location file
for row in csv.reader(f) :
    if count >= datanum  :
        break ;
    for coor in row :
        x.append(coor) ;
    count = count+1 ;
f.close();

t = [] ; #target data
count = 0 ;
tf = open( 'T_train.csv' , 'r');    #target file
for row in csv.reader(tf) :
    if count< datanum :
        for coor in row :
            t.append(int(coor)) ;        
tf.close() ;

#learn a set of datas once a time ;
p = [mean,alphai*np.identity(1682)]   #parameter of gaussian
for time in range(int(datanum/sdatanum)) :
    for i in range(sdatanum):
        temp = predictedArray( float(x.pop(1)),float(x.pop(0)) ,betai) ;
        dm.append( temp ) ;
        tdm.append( temp ) ;
    p = makePrior(p, dm, betai, t[sdatanum*time:(time+1)*sdatanum] ) ;
    del dm ;
    dm=[] ;
    
w = p[0] ;
#error = errorFunction( tdm, w, t[:datanum], datanum, float(betai/alphai) ) ;
#print() ;
#print(error) ;

print("print output file!!") ;
#read output file
of = open( 'X_test.csv' , 'r');    #read output file
o = [] ;
ans=[] ;
count = 0 ;
for row in csv.reader(of) :
    o.append(row) ;
    h = np.dot( predictedArray(float(o[count].pop(1)), float(o[count].pop(0)), betai),w) ;  #height
    ans.append( [h] if h>0 else [0] ) ;
    count = count +1 ;
of.close() ;
print(len(o)) ;

#print output file
pof = open( 'bayesian.csv' , 'w', newline='') ;
wri = csv.writer(pof) ;
for row in ans  :
    wri.writerow(row) ;
pof.close()   

#show graph

grid = np.linspace(0,1080,216) ;
X,Y = np.meshgrid(grid, grid) ;
z = np.zeros((216,216)) ;
for i in range(216):
    for j in range(216):
        z[i][j] = np.dot( predictedArray(Y[i][j],X[i][j], betai), w )  ;
        if z[i][j]<0 :
            z[i][j] = 0 ;
fig = plt.figure() ;
CS = plt.contour(X, Y, z, 20, linewidth=.5) ;
plt.show()

