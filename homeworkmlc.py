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
ldatanum = 20000; #   Data number that is going to be learned

n = 2 ;
dm = [] ;   #design matrix

x = [] ;
f = open( 'X_train.csv' , 'r') ;    #location file
for row in csv.reader(f) :
    x.append(row) ;
f.close();

t = [] ; #target data
tf = open( 'T_train.csv' , 'r');    #target file
for row in csv.reader(tf) :
    for coor in row :    
        t.append(int(coor)) ;
tf.close() ;

w = [] ;
error = [] ;
#write some stuff
for k in range(2) :
    for i in range(ldatanum*(k), ldatanum*(k+1)) :
        dm.append( predictedArray( float(x[i][1]) ,float(x[i][0]) )) ;
    w.append(vectorW(dm,t[ldatanum*(k):ldatanum*(k+1)])) ;
    if k == 0 :
        error.append(errorFunction( dm, w[k], t[ldatanum:2*ldatanum], ldatanum )) ;
        print(error) ;
    elif k == 1 :
        error.append(errorFunction( dm, w[k], t[0:ldatanum], ldatanum )) ;
        print(error) ;
    del dm ;
    dm = [] ;
#choose the one has smaller error 
if error[0] > error[1] :
    answ = w[1] ;
    print(error[1]) ;
else:
    answ = w[0] ; 
    print(error[0]) ;
del w ;

#show graph
grid = np.linspace(0,1080,216) ;
X,Y = np.meshgrid(grid, grid) ;
z = np.zeros((216,216)) ;
for i in range(216):
    for j in range(216):
        z[i][j] = np.dot( predictedArray(Y[i][j],X[i][j]) ,answ )  ;
        if z[i][j]<0 :
            z[i][j] = 0 ;
fig = plt.figure() ;
CS = plt.contour(X, Y, z, 20, linewidth=.5) ;
plt.show();

