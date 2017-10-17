from cvxopt import blas, lapack, solvers, matrix, spmatrix, sparse
import numpy as np
from numpy import array as na
from numpy.linalg import eig, eigh, cholesky 

import sympy as sy

import scipy

from numpy import zeros, ones, empty, pi
from numpy import identity as Id

from copy import deepcopy as dp

import sympy
from sympy import sin as sin, cos as cos, tan as tan 
from sympy import Matrix as sMa, zeros as sMz, ones as sM1

sMe = sympy.eye
from sympy.tensor.array import MutableDenseNDimArray as sTa

from matplotlib import pyplot as plt

import control as crtl

import itertools

import pickle

from functools import partial as partialFunc

from cythonUtils import *

from multiprocessing import Process, Queue, cpu_count

import time

glbDBG = True#Use full parallelization -> False; Better error tracking -> True

cpuCount = cpu_count()

def printDBG(aStr):
    if glbDBG:
        print(aStr)
    return None
#######################################################################
#Numerical constants definitions
numericEps = 1.0e-10 #Value considered as 0
numericEpsSdp = .0e-5 #If Q>>0 is demanded than Q >> numericEpsSdp is enforced

#######################################################################
#cython shorthands
toMat = lambda x: toMatC(x.squeeze())

#######################################################################
def toSo2(aArray):
    #Replacing all values larger or smaller than -pi or pi 
    #with values inside this interval representing the same angle
    if isinstance(aArray, np.ndarray):
        thisShape = aArray.shape
        newArray = aArray.reshape((-1,))
        newArray = toSo2_Cy(newArray).reshape(thisShape)
        return newArray
    else:
        return float( toSo2_Cy_D(aArray) )
#######################################################################
def replaceSo2(aArray, dims):
    #Helper function for toSo2
    if not(dims is None):
        aArray[dims, :] = toSo2(aArray[dims,:])
    return aArray
#######################################################################
#Helper functions to calculate the norms of vectors stocked as columns in a matrix
colWiseSquaredNorm = lambda X : np.sum( np.square( X ), 0)
colWiseNorm = lambda X : np.linalg.norm(X, axis=0)#  np.sqrt(np.sum( np.square( X ), 0))
#######################################################################
colWiseSquaredKerNorm = lambda P, X : np.square(np.sum( np.square( np.dot( np.linalg.cholesky(P).T, X ) ), 0)) 
colWiseKerNorm = lambda P, X : np.sum( np.square( np.dot( np.linalg.cholesky(P).T, X ) ), 0)  #np.linalg.norm(np.dot( np.linalg.cholesky(P).T, X ), axis=0) #np.sqrt( np.sum( np.square( np.dot( np.linalg.cholesky(P).T, X )), 0) )
#######################################################################
colWiseSquaredKerCNorm = lambda C, X : np.sum( np.square( np.dot( C, X ) ), 0) 
colWiseKerCNorm = lambda C, X : np.linalg.norm(np.dot( C, X ), axis=0)#np.sqrt( np.sum( np.square( np.dot( C, X )), 0) )
#######################################################################

#######################################################################
lmap = lambda func, aIterable: list(map(func, aIterable)) # using "map" here will cause infinite recursion
########################################################################

def myEvalf(expr, prec=64, subsDict={}):
    #Evaluate sympy expression
    return np.array(expr.evalf(prec, subs=subsDict)).astype(np.float_)

########################################################################
#functions to access and treat degrees
#this is necessary to map the monomials between different spaces
#and to generate monomials
def deg2str(arr, varNames = None):
    #Take the powers of a monomial stocked in a vector ang get its sting representation
    aStr = ''
    if varNames is None:
        for nr in arr:
            aStr+='{0:d}'.format(nr)
    else:
        for nr in arr:
            aStr+='*{0}'.format(varNames[nr])
        aStr = aStr[1:]
    return aStr
################
def str2deg(aStr):
    #Inverse function of deg2str: Takes a string and returns a vector of integers corresponding to the powers
    return np.array(list(map(int, aStr))).astype(np.int_)
    
#########################################################################
def evalPolyDict(polyDict, x):
    #Compute the value of a polynomial whos monomials(keys) and coefficients(values)
    #are stored in the given dictionnary
    sumP = 0.
    x = x.squeeze()
    for aMon,aCoef in polyDict.items():
        sumP += aCoef*np.prod( np.power(x, str2deg(aMon)) )
    
    return sumP
    
    
#########################################################################

#Return P^{-1/2}, ie getPm2(P)*P*getPm2(P).T = eye() 
#def getPm2(P):
#    a,b = np.linalg.eigh(P)
#    return np.dot(np.diag(1./np.sqrt(a)), b.T)
def getPm2(P):
    return np.linalg.inv(np.linalg.cholesky(P).T)
#####################################
#Return P^{1/2}, ie getP2(P).T . getP2(P) = P 
#def getP2(P):
#    a,b = np.linalg.eigh(P)
#    return np.dot(np.diag(np.sqrt(a)), b.T)
def getP2(P):
    return np.linalg.cholesky(P).T


#####################################
#Convenience functions
ndot = lambda *args: np.linalg.multi_dot(args)
def nmult(*args):
    out = args[0]
    for aM in args[1:]:
        out = np.multiply(out, aM)
    return out
def ndiv(*args):
    out = args[0]
    for aM in args[1:]:
        out = np.divide(out, aM)
    return out 

#####################################
#Interpolation functions used to compute the funnel shape 
def standardInterpol(Pn, alphan, Pn1, alphan1, t, t0, t1):
    #The quadratic matrix P is linearly interpolated between two given time steps
    dT = t1-t0
    return [Pn*alphan*(1.-(t-t0)/dT)+Pn1*alphan1*(t-t0)/dT, 1./dT*(alphan1*Pn1-alphan*Pn)]
#####################################
def cholInterpol(Cn, alphan, Cn1, alphan1, t, t0, t1, preDecomp = False):
    #The cholesky factorization is linearly interpolated
    if not preDecomp:
        Cn = np.linalg.cholesky(Cn)
        Cn1 = np.linalg.cholesky(Cn1)
    else:
        #"Normal" chol is P = C.C'
        #We use P=C'.C for convenience
        Cn = Cn.T
        Cn1 = Cn1.T 
    dT = t1-t0
    Ct = Cn*alphan*(1.-(t-t0)/dT)+Cn1*alphan1*(t-t0)/dT
    dCt = (-Cn*alphan+Cn1*alphan1)/dT
    return [ndot(Ct, Ct.T), ndot(Ct, dCt.T)+ndot(dCt,Ct.T)]
#####################################
def standardInterpolNoDeriv(Pn, alphan, Pn1, alphan1, t, t0, t1):
    #Like standardInterpol but the derivative is set to 0
    #This is useful when computing the region of stabilizabilty of
    #of an eauilibrium point. i.e. the reference does not move 
    dT = t1-t0
    return [Pn*alphan*(1.-(t-t0)/dT)+Pn1*alphan1*(t-t0)/dT, np.zeros_like(Pn)]
#####################################
def cholInterpolNoDeriv(Cn, alphan, Cn1, alphan1, t, t0, t1, preDecomp = False):
    #Like cholInterpol but the derivative is set to 0
    #This is useful when computing the region of stabilizabilty of
    #of an eauilibrium point. i.e. the reference does not move 
    if not preDecomp:
        Cn = np.linalg.cholesky(Cn)
        Cn1 = np.linalg.cholesky(Cn1)
    else:
        #"Normal" chol is P = C.C'
        #We use P=C'.C for convenience
        Cn = Cn.T
        Cn1 = Cn1.T 
    dT = t1-t0
    Ct = Cn*alphan*(1.-(t-t0)/dT)+Cn1*alphan1*(t-t0)/dT
    return [ndot(Ct, Ct.T), np.zeros_like(Cn)]
#####################################
def backPropInterPol(PnTnList, alphan, PnTn1List, alphan1, t, t0, t1, *args):
    #Interpolator when multiple substeps are stored in PnTnList
    #Pn1Tn1List not needed, only passed for completeness
    PnL=PnTnList[0]
    TnA=PnTnList[1] 
    
    t=np.maximum(np.minimum(t, np.nextafter(t1,t1-1.)), np.nextafter(t0,t0+1.))
    ind = np.searchsorted(TnA, t)-1
    PnS = PnL[ind]
    PnS1 = PnL[ind+1]
    alphanS = 1.+(alphan-1)*(1-(TnA[ind]-t0)/(t1-t0))
    alphanS1 = 1.+(alphan-1)*(1-(TnA[ind+1]-t0)/(t1-t0)) 
    
    return standardInterpol(PnS, alphanS, PnS1, alphanS1, t, TnA[ind], TnA[ind+1])
#####################################
def tLQRinterPol(PnPdnTnList, alphan, PnPdnTn1List, alphan1, t, t0, t1, *args):
    #Specific interpolator for tLQR class results.
    #Interpolation is nearest
    
    PnL=PnPdnTnList[0]
    PdnL=PnPdnTnList[1]
    TnA=PnPdnTnList[2]
    
    t=np.maximum(np.minimum(t, np.nextafter(t1,t1-1.)), np.nextafter(t0,t0+1.))
    #ind = np.searchsorted(TnA, t)-1
    ind = np.argmin(np.abs(TnA-t)) #closest
    tp = TnA[ind]
    alphanS = 1.+(alphan-1)*(1-(tp-t0)/(t1-t0))
    alphandS = -(alphan-1)/(t1-t0)
    PnS = PnL[ind]*alphanS
    PdnS = PdnL[ind]*alphanS + PnL[ind]*alphandS
    
    return [PnS, PdnS]
#####################################
def tLQRinterPol2(PnPdnTnList, alphan, PnPdnTn1List, alphan1, t, t0, t1, *args):
    #Specific interpolator for tLQR class results.
    #Interpolation is cholInterpolator
    PnL=PnPdnTnList[0]
    PdnL=PnPdnTnList[1]
    TnA=PnPdnTnList[2]
    
    t=np.maximum(np.minimum(t, np.nextafter(t1,t1-1.)), np.nextafter(t0,t0+1.))
    ind = np.searchsorted(TnA, t)-1
    t0i = TnA[ind]
    t1i = TnA[ind+1]
    alphanS0 = 1.+(alphan-1)*(1-(t0i-t0)/(t1-t0))
    alphanS1 = 1.+(alphan-1)*(1-(t1i-t0)/(t1-t0))
    PnS0 = PnL[ind]
    PnS1 = PnL[ind+1]
    
    return cholInterpol(PnS0, alphanS0, PnS1, alphanS1, t, t0i, t1i, preDecomp=False)
#####################################
interpList = [standardInterpol, cholInterpol, standardInterpolNoDeriv, cholInterpolNoDeriv, backPropInterPol, tLQRinterPol, tLQRinterPol2]
#####################################
#Get the variables needed for polynomial (up to deg 4) relaxation
#firstCond is 
#| 1, x.T | 
#| x,  X  | >> 0
#Second cond is 
#| 1, x.T, z.T | 
#| x,  X ,  Y  | >> 0
#| z, Y.T,  Z  |
#thirdCond is 
#|  X , Y | 
#| Y.T, Z | >> 0 
#which is implied but sometimes hemps to converge//overcome numerical probs

def getVarsForPolyRelax(dim):
    
    dimX = (dim*(dim+1))//2    
    
    firstCondA = np.zeros(((dim+1)**2, 0))
    #firstCondB = np.zeros((dim+1,dim+1)).reshape((-1,1))
    firstCondB = -numericEpsSdp*np.identity(dim+1).reshape((-1,1))
    firstCondB[0] += 1.
    secondCondA = np.zeros(((dim+dimX+1)**2,0))
    #secondCondB = np.zeros((dim+dimX+1,dim+dimX+1)).reshape((-1,1))
    secondCondB = -numericEpsSdp*np.identity(dim+dimX+1).reshape((-1,1))
    secondCondB[0] += 1.
    thirdCondA = np.zeros(((dimX+1)**2,0))
    thirdCondB = -numericEpsSdp*np.identity(dimX+1).reshape((-1,1))
    thirdCondB[0] += 1.
    
    indMatrixx = np.zeros((dim,))
    indMatrixX = np.zeros((dim,dim))
    indMatrixZtilde = np.zeros((dim+dimX,dim+dimX))
    indMatrixAll = -np.ones((dim+dimX+1,dim+dimX+1))
    indMatrixXtoz = np.zeros((dim,dim))
    #New
    indMatrixzZ = -np.ones((dimX+1,dimX+1))
    
        
    thisVar = 0
    aCondMatrix = np.zeros((dim+1,dim+1))
    bCondMatrix = np.zeros((dim+dimX+1,dim+dimX+1))
    ###This is ugly and everything needs to be redone for the new version
    NProb = [0]
    #get everything related to x
    for k in range(dim):
        indMatrixx[k] = thisVar
        indMatrixAll[0,1+k] = thisVar
        indMatrixAll[1+k,0] = thisVar
        #condition for X > x.x' -> [1,x';x,X]>0
        aCondMatrix[0,1+k]=1.; aCondMatrix[1+k,0]=1.
        #firstCondA[:,thisVar] = aCondMatrix.reshape((-1,))
        firstCondA = np.hstack((firstCondA, aCondMatrix.reshape((-1,1)) ))
        aCondMatrix[0,1+k]=0.; aCondMatrix[1+k,0]=0.
        #condition for Ztilde > [x;z].[x,z]'-> [1,x',z';x,X,x.z';z,z.x',z.z']>0
        bCondMatrix[0,1+k]=1.; bCondMatrix[1+k,0]=1.
        #secondCondA[:,thisVar] = bCondMatrix.reshape((-1,))
        secondCondA = np.hstack((  secondCondA, bCondMatrix.reshape((-1,1)) ))
        bCondMatrix[0,1+k]=0.; bCondMatrix[1+k,0]=0.
        thisVar += 1
        NProb[0] += 1
    #get everything related to z <-> z=vec(X)
    NProb.append( dp(NProb[0]) )
    linInd = 0
    for i in range(dim):
        for j in range(i,dim):
            indMatrixX[i,j] = thisVar
            indMatrixX[j,i] = thisVar
            indMatrixXtoz[j,i] = linInd
            indMatrixXtoz[i,j] = linInd
            indMatrixZtilde[i,j] = thisVar
            indMatrixZtilde[j,i] = thisVar
            indMatrixAll[1+i,1+j] = thisVar
            indMatrixAll[1+j,1+i] = thisVar
            indMatrixAll[0,1+dim+linInd] = thisVar
            indMatrixAll[1+dim+linInd,0] = thisVar
            
            #condition for X > x.x' -> [1,x';x,X]>0
            aCondMatrix[1+i,1+j]=1.; aCondMatrix[1+j,1+i]=1.
            #firstCondA[:,thisVar] = aCondMatrix.reshape((-1,))
            firstCondA = np.hstack(( firstCondA, aCondMatrix.reshape((-1,1)) ))
            aCondMatrix[1+i,1+j]=0.; aCondMatrix[1+j,1+i]=0.
            #condition for Ztilde > [x;z].[x,z]'-> [1,x',z';x,X,x.z';z,z.x',z.z']>0
            #X
            bCondMatrix[1+i,1+j]=1.; bCondMatrix[1+j,1+i]=1.
            #z
            bCondMatrix[0,1+dim+linInd]=1.; bCondMatrix[1+dim+linInd,0]=1.
            #secondCondA[:,thisVar] = bCondMatrix.reshape((-1,))
            secondCondA = np.hstack(( secondCondA, bCondMatrix.reshape((-1,1)) ))
            bCondMatrix[1+i,1+j]=0.; bCondMatrix[1+j,1+i]=0.
            bCondMatrix[0,1+dim+linInd]=0.; bCondMatrix[1+dim+linInd,0]=0.
            
            linInd += 1
            thisVar += 1
            NProb[1] += 1
    NProb.append( dp(NProb[1]) )
    
    #get everything related to x.z'
    for i in range(dim):
        for j in range(dimX):
            indMatrixZtilde[i,dim+j] = thisVar
            indMatrixZtilde[dim+j,i] = thisVar
            indMatrixAll[1+i,1+dim+j] = thisVar
            indMatrixAll[1+dim+j,1+i] = thisVar
            
            thisVar += 1
            NProb[2] += 1
    NProb.append( dp(NProb[2]) )
    
    thisVarbeforeZ = thisVar
    #get everything related to Z>z.z'
    for i in range(dimX):
        for j in range(i,dimX):
            indMatrixZtilde[dim+i,dim+j] = thisVar
            indMatrixZtilde[dim+j,dim+i] = thisVar
            indMatrixAll[1+dim+i,1+dim+j] = thisVar
            indMatrixAll[1+dim+j,1+dim+i] = thisVar
            
            thisVar += 1
            NProb[3] += 1
    
    indMatrixx = indMatrixx.astype(np.int_)
    indMatrixX = indMatrixX.astype(np.int_)
    indMatrixXtoz = indMatrixXtoz.astype(np.int_)
    indMatrixZtilde = indMatrixZtilde.astype(np.int_)
    indMatrixAll = indMatrixAll.astype(np.int_)
    
    
    numRedundant = 0
    #Replace redundant variables in x.z
    for i in range(dim):
        #X_ii*x_j = X_ij*x_i
        for j in range(dim):
            if i==j:
                continue
            
            NProb[2] -= 1
            NProb[3] -= 1
            
            zii = indMatrixXtoz[i,i]
            zij = indMatrixXtoz[i,j]
            Xiixj = indMatrixZtilde[j, dim+zii]
            Xijxi = indMatrixZtilde[i, dim+zij]
            
            ind2Use = min(Xiixj, Xijxi)
            ind2Erase = max(Xiixj, Xijxi)
            indMatrixZtilde[indMatrixZtilde==Xiixj] = ind2Use
            indMatrixAll[indMatrixAll==Xiixj] = ind2Use
            indMatrixZtilde[indMatrixZtilde==Xijxi] = ind2Use
            indMatrixAll[indMatrixAll==Xijxi] = ind2Use
            numRedundant += 1
            for k in range(ind2Erase+1, thisVar):
                indMatrixAll[indMatrixAll==k] = k-1
                indMatrixZtilde[indMatrixZtilde==k] = k-1
    #Replace redundant variables in X.X
    for i in range(dim):
        for j in range(i+1,dim):
            #X_ij*X_ij=X_ii*X_jj
            #Z_ij^ij=Z_ii^jj
            NProb[3] -= 1
            zii = indMatrixXtoz[i,i]
            zjj = indMatrixXtoz[j,j]
            zij = indMatrixXtoz[i,j]
            Ziijj = indMatrixZtilde[dim+zii,dim+zjj]
            Zijij = indMatrixZtilde[dim+zij,dim+zij]
            
            ind2Use = min(Ziijj, Zijij)
            ind2Erase = max(Ziijj, Zijij)
            indMatrixZtilde[indMatrixZtilde==Ziijj] = ind2Use
            indMatrixAll[indMatrixAll==Ziijj] = ind2Use
            indMatrixZtilde[indMatrixZtilde==Zijij] = ind2Use
            indMatrixAll[indMatrixAll==Zijij] = ind2Use
            numRedundant += 1
            for k in range(ind2Erase+1, thisVar):
                indMatrixAll[indMatrixAll==k] = k-1
                indMatrixZtilde[indMatrixZtilde==k] = k-1
                
    #Fill the additional Matrix
    indMatrixzZ[0,1:] = indMatrixAll[0,1+dim:]
    indMatrixzZ[1:,0] = indMatrixAll[1+dim:,0]
    indMatrixzZ[1:,1:] = indMatrixAll[1+dim:,1+dim:]
    indMatrixzZ = indMatrixzZ.astype(np.int_)
    
    numActZ = thisVar-thisVarbeforeZ-numRedundant
    aCondMatrix[:] = 0.
    for aVar in range( NProb[1], NProb[-1] ):
        firstCondA = np.hstack(( firstCondA, aCondMatrix.reshape((-1,1)) ))
        bCondMatrix[:] = 0
        bCondMatrix[1:,1:] = (indMatrixZtilde == aVar).astype(np.float_)
        secondCondA = np.hstack(( secondCondA, bCondMatrix.reshape((-1,1)) ))
    ###
    #Integrate this better
    cCondMatrix = np.zeros((1+dimX,1+dimX))
    for aVar in range(NProb[-1]):
        cCondMatrix[:] = 0
        cCondMatrix = (indMatrixzZ == aVar).astype(np.float_)
        thirdCondA = np.hstack(( thirdCondA, cCondMatrix.reshape((-1,1)) ))
    
    
    #Completely redo the LMI conditions
     
    
    
    dimProb = firstCondA.shape[1]
    dTraceX = np.zeros((dimProb,))
    dTraceZtilde = np.zeros((dimProb,))
    
    for k in range(dimProb):
        #fill up the trace
        dTraceX[k] = np.trace(firstCondA[:,k].reshape((dim+1,dim+1)))
        dTraceZtilde[k] = np.trace(secondCondA[:,k].reshape((dim+dimX+1,dim+dimX+1)))
    
    #Adjust signs
    #A+B>0 -> -A < B
    firstCondA = -firstCondA
    secondCondA  = -secondCondA
    thirdCondA  = -thirdCondA
    
    #Count the power of of each term in x
    powerList = [np.zeros((dim,)).astype(np.int_) for k in range(dim)]
    #also build forward and backward dicts
    ind2degrees = {}
    
    for i in range(dim):
        powerList[i][i]=1
        ind2degrees[indMatrixx[i]] = powerList[i]
    
    for i in range(dim):
        for j in range(i,dim):
            powerList.append( powerList[i]+powerList[j] )
            
    powerMatrix = [[ [] for i in range(dim+dimX) ] for j in range(dim+dimX)]        
    for i in range(dim+dimX):
        for j in range(dim+dimX):
            powerMatrix[i][j] = powerList[i]+powerList[j]
            thisInd = indMatrixZtilde[i][j]
            if ( not (thisInd in ind2degrees.keys()) ):
                ind2degrees[thisInd] = powerMatrix[i][j] 
    
    degrees2ind = {deg2str(v): k for k, v in ind2degrees.items()}
        
    return degrees2ind, ind2degrees, indMatrixx, indMatrixX, indMatrixZtilde, indMatrixzZ, indMatrixAll, indMatrixXtoz, dTraceX, dTraceZtilde, [firstCondA, firstCondB], [secondCondA, secondCondB], [thirdCondA, thirdCondB], powerList, powerMatrix

#####################################################################

def noRedundCstrApprox(A, b, c=None):
    #get an heuristic approximation of the nonredudant constraint set
    #Partially taken from noredund from matlabcentral
    L,N = A.shape
    
    if (not (c is None)):
        c = np.array(c).reshape((-1,1))
    
    if ( (c is None) or np.any( np.dot( A, c ) > b )):
        #Non c or inappriopriate c given
        #Impose a minimal distance since it should re
        Atilde = np.vstack(( np.hstack(( np.ones((L,1)), A )), np.array([[-1.]+[0.]*N]) ))
        
        
        
        obj = matrix(0.0, (N+1,1))
        obj[0] = -1.
        
        solLP = solvers.lp( obj, G=sparse(matrix(Atilde)), h=matrix(np.vstack((b, -0.01*np.ones((1,1))))) )
        if glbDBG:
            try:
                assert solLP['status'] == 'optimal', 'Incompatible Cstr'
            except:
                print('bb - LP failed')
        else:
            assert solLP['status'] == 'optimal', 'Incompatible Cstr'
        
        c = np.array(solLP['x'][1:]).reshape((-1,1))
    
    # move polytope to contain origin
    bk = b.copy() # preserve
    b = b - np.dot(A,c) # polytope A*x <= b now includes the origin
    # obtain dual polytope vertices: Simply use min max in each direction as approx
    D = np.divide(A , b) #<-> D = np.divide(A , np.tile(b,[1, N]))
    limMinI = np.argmin(D, axis=0)
    limMaxI = np.argmax(D, axis=0)
    
    nr = np.unique( np.hstack(( limMinI, limMaxI )) )
    
    An=A[nr,:]
    bn=bk[nr]
    
    return An, bn, nr

#####################################################################


def getMonoEvaluator(nx, N):
    polyDegrees=[np.zeros((nx,)).astype(np.int_) for i in range(nx)]
    
    x = sympy.symbols('x0:{0}'.format(nx))
    
    nCount=nx
    #Linear
    for k in range(nx):
        polyDegrees[k][k]=1
    linearTerms = dp(polyDegrees)
    #other
    for k in range(1,N):
        currNum = len*polyDegrees
        for j in range(currNum):
            for i in range(nx):
                thisMono = linearTerms[i]+polyDegrees[j]
                #test uniqueness
                if not np.any(np.min(thisMono == polyDegrees, axis=1)):
                    polyDegrees.append(thisMono)
                    nCount +=1 
    
    polyVars=[[] for i in range(len(polyDegrees))]
    for k in range(len(polyDegrees)):
        for i in range(nx):
                polyVars[k] += polyDegrees[k][i]*x[i]
    
    polyVarsM = sMa(list(map( sympy.prod, polyVars)))
        
    #get a fast polynomial function
    polyEval = sympy.lambdify( x, polyVarsM, 'numpy' )
    polyEvalS = lambda *args: polyEval(*args).reshape((nCount, args[0].size))
    
    return polyEvalS


