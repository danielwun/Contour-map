import csv ;
import numpy as np ;
from math import exp ;
import matplotlib.pyplot as plt ;

def predictedArray( y,x ):
    # complete one of columns in design matrix 
    s = 25 ;
    h = [] ;
    dim = 31 ;  #dimension of gaussian function
    sep = 1080/(dim-1) ;    #seperation of two gaussian function 
    for i in range(dim) :
        for j in range(dim) :
            h.append( exp( -1*pow((x-sep*(i)),2)/2/pow(s,2)-pow((y-sep*(j)),2)/2/pow(s,2)) ) ;
    
    return h ;    

def vectorW( dm, test ):
    # making w list 
    tdm = list(zip(*dm)) ;  #transpose matrix
    dm2 = np.dot(tdm,dm) ;   # dot product of two matrixes
    im = 0.00000001 * np.identity(len(dm[0])) ;    #identity matrix 
    idm = np.linalg.inv( dm2+im ) ;    # inverse dm2;
    dm3 = np.dot(idm,tdm) ; 
    final = np.dot(dm3,test) ;
    del dm2,idm,tdm,dm3,im ;
    return final ;

def errorFunction( dm, w, t, num ):
    # with 1/2 one
    lamb = 0.00000001 ;    #lambda
    y = np.dot( dm, w ) ;
    me = [ y[i]-t[i] for i in range(num)] ; #mean square error
    me = 0.5/num*sum(me[i]*me[i] for i in range(num)) ;
   #me = me + lamb/2*sum(w[i]*w[i] for i in range(len(w))) ;
    return me ;

#!!!!!!!!Need feature extraction function !!!!!!!!!!!!
    
# read information of coordinates 
# main function
count = 0 ;
ldatanum = 40000; #   Data number that is going to be learned
udatanum = 40 ; #   how many number of datas do we want to use ;
dm = [] ;   #design matrix
x = [] ;

f = open( 'X_train.csv' , 'r') ;    #location file
for row in csv.reader(f) :
    if count >= ldatanum  :
        break ;
    for coor in row :
        x.append(coor) ;
    dm.append( predictedArray( float(x.pop()) ,float(x.pop()) )) ;
    count = count+1 ;
f.close();

tcount = 0 ;
t = [] ; #target data
tf = open( 'T_train.csv' , 'r');    #target file
for row in csv.reader(tf) :
    for coor in row :
        if tcount < ldatanum  :            
            t.append(int(coor)) ;
    tcount = tcount+1 ;
tf.close() ;

w = vectorW( dm,t ) ;
error = errorFunction( dm, w, t, ldatanum ) ;
print() ;
print(error) ;

'''
#read output file
of = open( 'X_test.csv' , 'r');    #read output file
o = [] ;
ans = [] ;
count = 0 ;
for row in csv.reader(of) :
    o.append(row) ;
    h = np.dot( predictedArray(float(o[count].pop(1)), float(o[count].pop(0))),w) ;  #height
    ans.append( [h] if h>0 else [0] ) ;
    count = count +1 ;
of.close() ;


#print output file
pof = open( 'X_map.csv' , 'w', newline='') ;
wri = csv.writer(pof) ;
for row in ans  :
    wri.writerow(row) ;
pof.close()  
'''
#show graph
'''
grid = np.linspace(0,1080,216) ;
X,Y = np.meshgrid(grid, grid) ;
z = np.zeros((216,216)) ;
for i in range(216):
    for j in range(216):
        z[i][j] = np.dot( predictedArray(Y[i][j],X[i][j]) ,w )  ;
        if z[i][j]<0 :
            z[i][j]=0 ;
fig = plt.figure() ;
CS = plt.contour(X, Y, z, 20, linewidth=.5) ;
plt.show()
'''
