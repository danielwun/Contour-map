import csv ;
import numpy as np ;
from math import exp ;
import matplotlib.pyplot as plt ;

def predictedArray( y,x ):
    # complete one of columns in design matrix 
    s = 25 ;
    h = [] ;
    dim = 41 ;  #dimension of gaussian function
    sep = 1080/(dim-1) ;    #seperation of two gaussian function
    h.append(1) ;
    for i in range(dim) :
        for j in range(dim) :
            #need to add one constant dimension
            h.append( exp( -1*pow((x-sep*(i)),2)/2/pow(s,2)-pow((y-sep*(j)),2)/2/pow(s,2)) ) ;
    
    return h ;     

def vectorW( dm, test ):
    # making w list 
    tdm = list(zip(*dm)) ;  #transpose matrix
    dm2 = np.dot(tdm,dm) ;   # dot product of two matrixes
    idm = np.linalg.inv(dm2) ;    # inverse dm2;
    dm3 = np.dot(idm,tdm) ; 
    final = np.dot(dm3,test) ;
    del dm2,idm,tdm,dm3 ;
    return final ;

def errorFunction( dm, w, t, num ):
    # with 1/2 one
    y = np.dot( dm, w ) ;
    me = [ 0-t[i] if y[i]<0 else y[i]-t[i] for i in range(num)] ; #need to be rectify
    me = 0.5*sum(me[i]*me[i] for i in range(num))/num ;
    return me ;

#!!!!!!!!Need feature extraction function !!!!!!!!!!!!
    
# read information of coordinates 
# main function
ldatanum = 40000; #   Data number that is going to be learned
dm = [] ;   #design matrix

x = [] ;
f = open( 'X_train.csv' , 'r') ;    #location file
for row in csv.reader(f) :
    for coor in row :
        x.append(coor) ;
    dm.append( predictedArray( float(x.pop()) ,float(x.pop()) )) ;
f.close();

t = [] ; #target data
tl = [] ; #left target data
tf = open( 'T_train.csv' , 'r');    #target file
for row in csv.reader(tf) :
    for coor in row :
       t.append(int(coor)) ;    
tf.close() ;

w = vectorW( dm,t ) ;
#error = errorFunction( dm, w, t, ldatanum ) ;
#print() ;
#print(error) ;
'''
print("printing output file") ;
#read output file
of = open( 'X_test.csv' , 'r');    #read output file
o = [] ;
ans=[] ;
count = 0 ;
for row in csv.reader(of) :
    o.append(row) ;
    h = np.dot( predictedArray(float(o[count].pop(1)), float(o[count].pop(0))),w) ;  #height
    ans.append( [h] if h>0 else [0] ) ;
    count = count +1 ;
of.close() ;
print(len(ans)) ;

#print output file
pof = open( 'X_ml.csv' , 'w', newline='') ;
wri = csv.writer(pof) ;
for row in ans  :
    wri.writerow(row) ;
pof.close()
'''
#show graph
grid = np.linspace(0,1080,216) ;
X,Y = np.meshgrid(grid, grid) ;
z = np.zeros((216,216)) ;
for i in range(216):
    for j in range(216):
        z[i][j] = np.dot( predictedArray(Y[i][j],X[i][j]) ,w )  ;
        if z[i][j]<0 :
            z[i][j] = 0 ;
fig = plt.figure() ;
CS = plt.contour(X, Y, z, 20, linewidth=.5) ;
plt.show();

