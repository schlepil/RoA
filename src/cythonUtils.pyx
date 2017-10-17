#from ctypes import c_bool
cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
import math

from libc.math cimport remainder, M_PI

#cython -a cythonUtils.pyx; gcc-5 -fopenmp -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python3.4 -o cythonUtils.so cythonUtils.c

@cython.boundscheck(False) #Decomment if debugged


cdef double dPi = 2.*M_PI

###########################################
@cython.wraparound(False)
cpdef toSo2_Cy_D(double val):
    return remainder(val, 2.*dPi);

cpdef toSo2_Cy(double[::1] aArray):
    cdef size_t l,i 
    cdef double[::1] outArr = np.zeros_like(aArray)
    l = aArray.shape[0]
     
    for i in prange(l, nogil=True): #Why can't i nogil it???
        outArr[i] = remainder(aArray[i], dPi)
    
    return np.asarray(outArr)

###########################################
@cython.wraparound(True)
def toVecI(double[:,::1] X, long[:,::1] Ind):
    cdef double[:,::1] x = np.zeros((X.shape[0]*(X.shape[0]+1)/2,1))#This is more space than needed, The last found will win
    cdef size_t i,j
    cdef long thisInd
    
    for i in range(X.shape[0]):
        for j in range(i,X.shape[0]):
            thisInd = Ind[i,j]
            if thisInd >= 0:
                x[thisInd] = X[i,j]
    
    return np.asarray(x)
###########################################
def toMatI(double[::1] x, long[:,::1] Ind):
    cdef double[:,::1] X = np.zeros_like(Ind, dtype=np.float_)
    cdef size_t i,j
    cdef long thisInd
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            thisInd = Ind[i,j]
            if thisInd >= 0:
                X[i,j] = x[thisInd]
            else:
                X[i,j] = 1.
    
    return np.asarray(X)
@cython.wraparound(False)
###########################################
def toVec(double[:,::1] X):
    cdef double[:,::1] x = np.zeros((X.shape[0]*(X.shape[0]+1)/2,1))
    cdef size_t k = 0
    cdef size_t i,j
    
    for i in range(X.shape[0]):
        for j in range(i,X.shape[0]):
            x[k] = X[i,j]
            k = k+1
    
    return np.asarray(x)
###########################################
def toMatC(double[::1] x):
    cdef size_t n = int((-1+math.sqrt(1+4*2*x.size))/2)
    print(n)
    cdef double [:,::1] X = np.zeros((n,n))
    cdef size_t i,j
    cdef size_t k = 0
    
    for i in range(n):
        for j in range(i,n):
            X[i,j] = x[k]
            X[j,i] = x[k]
            k = k+1
    
    return np.asarray(X)
###########################################

cpdef quadForm( double[:,::1] P, double[:,::1] x):
    #Evaluate x.T.P.x
    cdef size_t i,j,k
    cdef double[::1] out = np.zeros(x.shape[1],)
    cdef double s
    cdef double xik
    cdef int dim = P.shape[0]
    cdef size_t nPt = x.shape[1] 
    #Loop over x
    #with nogil:
    for k in prange(nPt, nogil=True):
        s=0
        #Get diag terms
        for i in range(dim):
            s = s + P[i,i]*x[i,k]*x[i,k]
        #Get off diags
        for i in range(dim):
            xik = x[i,k]
            for j in range(i+1, dim):
                s=s + 2.*P[i,j]*xik*x[j,k]
        out[k] = s
    
    return np.asarray(out)

##########################################
cpdef cdot3( const double[:,:,::1] a, const double[:,::1] b ):
    #c[:,i] = a[:,:,i].b[:,i]
        
    cdef size_t i,j,k
    
    cdef double s
    
    cdef double[:,::1] c = np.zeros((a.shape[0],a.shape[2]))
    
    cdef size_t dim0, dim1, dim2
    dim0,dim1,dim2=a.shape[0],a.shape[1],a.shape[2]
    
    if dim2 == 1:
        for i in range(a.shape[2]):
            #Loop over every vector in c
            for j in range(a.shape[0]):
                #Every entry in that vector
                s = 0.
                for k in range(a.shape[1]):
                    #c[j,i] = sum_k a[i,j,k]*b[k,i]
                    s = s + a[j,k,i]*b[k,0]
                c[j,i] = s #I dont know if this is faster than accessing it multiple times
    else:
        for i in prange(dim2, nogil=True):
            #Loop over every vector in c
            for j in range(dim0):
                #Every entry in that vector
                s = 0.
                for k in range(dim1):
                    #c[j,i] = sum_k a[j,k,i]*b[k,i]
                    s = s + a[j,k,i]*b[k,i]
                c[j,i] = s #I dont know if this is faster than accessing it multiple times
    
    return np.asarray(c)
#########################################


#########################################
#Functions to create rlt
#Attention A and b are assumed to be compliant to
#A.x <= b
#No check on the validity is performed!
#b is assumed to be a vector column vector
#It will output (A1,b1) wedge (A2,b2)
#If A2,b2 has zero dims, it does (A1,b1) wedge (A1,b1)
@cython.wraparound(False)
cpdef getRLTxxCy(double[:,::1] A1, double[:,::1] b1, double[:,::1] A2, double[:,::1] b2, long[:,::1] iMa, long nx):

    cdef size_t nCstr1, nVar1, nCstr2, nVar2
    
    cdef size_t i,j,nNewCstr
    
    cdef size_t allCstr
    
    nCstr1, nVar1 = A1.shape[0], A1.shape[1]
    nCstr2, nVar2 = A2.shape[0], A2.shape[1]
    
    if nCstr2 == 0:
        allCstr = (nCstr1*(nCstr1+1))/2
    else:
        allCstr = nCstr1*nCstr2
    
    cdef double[:,::1] Aout = np.zeros(( allCstr, nVar1 ))
    cdef double[:,::1] bout = np.zeros(( allCstr, 1 ))
    
    cdef double[::1] AplusTemp = np.zeros(( nVar1+1 ))
    
    cdef long[::1] ix = iMa[0,1:1+nx]
    
    cdef size_t iInner,jInner
    cdef long iiInner
    
    nNewCstr = 0
    
    if nCstr2 == 0:
        for i in range(nCstr1):
            for j in range(i,nCstr1):
                
                for iInner in range(nx):
                    iiInner = iMa[0,1+iInner]
                    
                    #Multiplied by scalar terms so - b2*A1*x - b1*A2*x
                    Aout[nNewCstr, ix[iInner]] += b1[j,0]*A1[i,ix[iInner]]+b1[i,0]*A1[j,ix[iInner]]
                    
                    #get A1ii*x_ii*A2ii*x_ii
                    Aout[nNewCstr, iMa[1+iInner,1+iInner]] -= A1[i,iiInner]*A1[j,iiInner]
                    
                    for jInner in range(iInner+1,nx):
                            #get A1i*x_i*A2j*x_j + A1j*x_j*A2i*x_i
                            Aout[nNewCstr, iMa[1+iInner,1+jInner]] -= A1[i,iMa[0,1+iInner]]*A1[j,iMa[0,1+jInner]] + A1[i,iMa[0,1+jInner]]*A1[j,iMa[0,1+iInner]]
                
                #Get the B with correct sign -b1*b2
                bout[nNewCstr,0] = b1[i,0]*b1[j,0]
                
                nNewCstr += 1
    else:
        for i in range(nCstr1):
            for j in range(nCstr2):
                
                #Get the quadratic terms A1*A2*x*x
                for iInner in range(nx):
                    iiInner = iMa[0,1+iInner]
                    
                    #Multiplied by scalar terms so - b2*A1*x - b1*A2*x
                    Aout[nNewCstr, ix[iInner]] += b2[j,0]*A1[i,ix[iInner]]+b1[i,0]*A2[j,ix[iInner]]
                    
                    #get A1ii*x_ii*A2ii*x_ii
                    Aout[nNewCstr, iMa[1+iInner,1+iInner]] -= A1[i,iiInner]*A2[j,iiInner]
                    
                    for jInner in range(iInner+1,nx):
                            #get A1i*x_i*A2j*x_j + A1j*x_j*A2i*x_i
                            Aout[nNewCstr, iMa[1+iInner,1+jInner]] -= A1[i,iMa[0,1+iInner]]*A2[j,iMa[0,1+jInner]] + A1[i,iMa[0,1+jInner]]*A2[j,iMa[0,1+iInner]]
                
                #Get the B with correct sign -b1*b2
                bout[nNewCstr,0] = b1[i,0]*b2[j,0]
                
                nNewCstr += 1
    
    return np.asarray(Aout), np.asarray(bout)

#########################################################################################################
cpdef getSpecifiedRLT(const double[:,::1] A1, const double[:,::1] b1, double[:,::1] A2, double[:,::1] b2, const long[:,::1] iMa, const long[:,::1] iMa1, const long[:,::1] iMX2z, const long nx, const long nX, const long nxz, const long[::1] types):
    
    if A1.shape[0] == 0:
        return np.zeros((0, max(A1.shape[1],A2.shape[1]))), np.zeros((0,1)) 
    
    if types[0]==1 and types[1]==1:
        #getRLTxxCy(double[:,::1] A1, double[:,::1] b1, double[:,::1] A2, double[:,::1] b2, long[:,::1] iMa, long nx):
        return getRLTxxCy(A1, b1, A2, b2, iMa, nx)
    
    if (types[0]>=3 and types[1]>=2) or (types[0]>=2 and types[1]>=3):
        #Constraints of type xz cross xz can not be treated
        return np.zeros((0,A1.shape[1])), np.zeros((0,1)),np.zeros((0,A1.shape[1])), np.zeros((0,1))
    
    cdef bint isA2, doxz
    cdef size_t nCstr1, nCstr2, nVar, nNewCstr, allCstr
    cdef size_t quadLoop1, jBoundActual
    
    cdef size_t i,j,iInner,jInner
    
    cdef long ix0, ix1, thisXZvar, aInner
    
    cdef long[::1] ixX = iMa[0,1:]
    
    if A2.shape[0]==0:
        A2 = A1
        b2 = b1
        isA2 = 0
        types[1]=types[0]
    else:
        isA2 = 1
    
    if types[1]>=3:
        doxz = 1 #types[1]>=3
    else:
        doxz = 0
    
    if types[0] == 1:
        quadLoop1 = nx
    elif types[0]==2:
        quadLoop1 = nx+nX
    else:
        assert 0
        
    nCstr1, nVar, nCstr2 = A1.shape[0], A1.shape[1], A2.shape[0]
    
    if isA2:
        allCstr = nCstr1*nCstr2
    else:
        allCstr = nCstr1*(nCstr1+1)/2
    
    cdef double[:,::1] Aout = np.zeros((allCstr, nVar))
    cdef double[:,::1] bout = np.zeros((allCstr, 1))
    
    nNewCstr = 0
    
    for i in range(nCstr1):
        if isA2:
            jBoundActual = 0
        else:
            jBoundActual = i
        for j in range(jBoundActual,nCstr2):
            
            bout[nNewCstr,0] = b1[i,0]*b2[j,0]
            
            for iInner in range(nVar):
                #Linear terms
                Aout[nNewCstr, iInner] += b2[j,0]*A1[i,iInner]+b1[i,0]*A2[j,iInner]
            
            for iInner in range(nx+nX):
                #Multiplied by scalar terms so - b2*A1*x - b1*A2*x
                #Aout[nNewCstr, ixX[iInner]] += b2[j,0]*A1[i,ixX[iInner]]+b1[i,0]*A2[j,ixX[iInner]]
                
                for jInner in range(nx+nX):
                    #terms in x and X multiplied by terms in x and x
                    Aout[nNewCstr, iMa[1+iInner,1+jInner]] -= A1[i,ixX[iInner]]*A2[j,ixX[jInner]]
                    #Aout[nNewCstr, iMa[1+iInner,1+jInner]] -= A1[i,iMa[0,1+iInner]]*A2[j,iMa[1+jInner,0]]
                    #print("{0} : {1} : {2} : {3} : {4} : {5}".format(iMa[1+iInner,1+jInner], Aout[nNewCstr, iMa[1+iInner,1+jInner]],iMa[0,1+iInner],iMa[1+jInner,0],A1[i,iMa[0,1+iInner]],A2[j,iMa[1+jInner,0]]))
            
            #If doxz then we have to get the terms in x (of A1) and xz (of A2)
            if doxz:
                for ix0 in range(nx):
                    #ix0 covers all linear terms of one constraint
                    #Take care that no terms appear twice
                    #This is probably the worst part, do differently
                    for ix1 in range(nx):
                        for jInner in range(nX):
                            thisXZvar = iMa1[1+ix1,1+nx+jInner]
                            if thisXZvar == -1:
                                continue
                            aInner = iMX2z[ix0,ix1]
                            Aout[nNewCstr, iMa[1+nx+aInner, 1+nx+jInner]] -= A1[i,iMa[0,1+ix0]]*A2[j,thisXZvar]
                            #print("{0} : {1}".format(iMa[1+nx+aInner, 1+nx+jInner], Aout[nNewCstr, iMa[1+nx+aInner, 1+nx+jInner]]))
            nNewCstr += 1
    return Aout, bout     

###############################################################################
#getRLTGeneralCy(A1x, b1x, A1X, b1X, A2x, b2x, A2X, b2X, iMa=iMA, iMa1=iMAonlyOnce, nx=nx, types=np.array([1,3]))
def getRLTGeneralCy(const double[:,::1] A1x, const double[:,::1] b1x, const double[:,::1] A1X, const double[:,::1] b1X, const double[:,::1] A2x, const double[:,::1] b2x, const double[:,::1] A2X, const double[:,::1] b2X, const long[:,::1] iMa, const long[:,::1] iMa1, const long[:,::1]iMX2z, const long nx, const long nX, const long nxz, const long[::1] types1, const long[::1] types2):
    #It is assumed that all entries corresponding to z.z' are zero
    #TBD this needs to be improved
    
    cdef size_t nCstr1x,nCstr1X,nCstr2x,nCstr2X,nVar
    cdef double[:,::1] A1xA2x, A1xA2X, A1XA2x, A1XA2X
    cdef double[:,::1] b1xb2x, b1xb2X, b1Xb2x, b1Xb2X
    
    cdef bint isA2
    
    cdef long[::1] thisTypes = np.zeros(2,).astype(np.int_)
    
    nCstr1x,nCstr1X,nCstr2x,nCstr2X = A1x.shape[0], A1X.shape[0], A2x.shape[0], A2X.shape[0]
    
    assert nCstr1x>0 or nCstr1X>0
    
    if nCstr1x>0:
        nVar = A1x.shape[1]
    else:
        nVar = A1X.shape[1]
    
    if nCstr2x == 0 and nCstr2X==0:
        isA2 = 0
    else:
        isA2 = 1
    
    #Do the actual work
    #getSpecifiedRLT(const double[:,::1] A1, const double[:,::1] b1, const double[:,::1] A2, const double[:,::1] b2, const long[:,::1] iMa, const long[:,::1] iMa1, const long[:,::1] iMX2x, const long nx, const long nX, const long[2] types):
    #Constraint  x x
    if (isA2 and nCstr2x>0 and nCstr1x>0):
        if types1[0]<=types2[0]:
            thisTypes[0] = types1[0]; thisTypes[1] = types2[0]
            A1xA2x, b1xb2x = getSpecifiedRLT(A1x, b1x, A2x, b2x, iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
        else:
            thisTypes[0] = types2[0]; thisTypes[1] = types1[0]
            A1xA2x, b1xb2x = getSpecifiedRLT(A2x, b2x, A1x, b1x, iMa, iMa1, iMX2z, nx, nX, nxz,thisTypes)
    elif( (not isA2) and nCstr1x>0):
        thisTypes[0] = types1[0]; thisTypes[1] = types1[0]
        A1xA2x, b1xb2x = getSpecifiedRLT(A1x, b1x, np.zeros((0,0)), np.zeros((0,0)), iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
    else:
        #A2X is given do nothing here
        A1xA2x = np.zeros((0,nVar))
        b1xb2x = np.zeros((0,1))
    
    #Constraint x X
    if (isA2 and nCstr2X>0  and nCstr1x>0):
        if types1[0]<=types2[1]:
            thisTypes[0] = types1[0]; thisTypes[1] = types2[1]
            A1xA2X, b1xb2X = getSpecifiedRLT(A1x, b1x, A2X, b2X, iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
        else:
            thisTypes[0] = types2[1]; thisTypes[1] = types1[0]
            A1xA2X, b1xb2X = getSpecifiedRLT(A2X, b2X, A1x, b1x, iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
    elif( (not isA2) and nCstr1x>0 and nCstr1X>0):
        thisTypes[0] = types1[0]; thisTypes[1] = types1[1]
        A1xA2X, b1xb2X = getSpecifiedRLT(A1x, b1x, A1X, b1X, iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
    else:
        #A2X is given do nothing here
        A1xA2X = np.zeros((0,nVar))
        b1xb2X = np.zeros((0,1))
    
    #Constraint  X x
    if (isA2 and nCstr2x>0 and nCstr1X>0):
        if types1[1]<=types2[0]:
            thisTypes[0] = types1[1]; thisTypes[1] = types2[0]
            A1XA2x, b1Xb2x = getSpecifiedRLT(A1X, b1X, A2x, b2x, iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
        else:
            thisTypes[0] = types2[0]; thisTypes[1] = types1[1]
            A1XA2x, b1Xb2x = getSpecifiedRLT(A2x, b2x, A1X, b1X, iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
    else:
        #In this case A1XA2x = A1xA2X
        #A1X is not given do nothing here
        A1XA2x = np.zeros((0,nVar))
        b1Xb2x = np.zeros((0,1))
    
    #Constraint  X X
    if (isA2 and nCstr2X>0 and nCstr1X>0):
        if types1[1]<=types2[1]:
            thisTypes[0] = types1[1]; thisTypes[1] = types2[1]
            A1XA2X, b1Xb2X = getSpecifiedRLT(A1X, b1X, A2X, b2X, iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
        else:
            thisTypes[0] = types2[1]; thisTypes[1] = types1[1]
            A1XA2X, b1Xb2X = getSpecifiedRLT(A2X, b2X, A1X, b1X, iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
    elif( (not isA2) and nCstr1X>0):
        thisTypes[0] = types1[1]; thisTypes[1] = types1[1]
        A1XA2X, b1Xb2X = getSpecifiedRLT(A1X, b1X, np.zeros((0,0)), np.zeros((0,0)), iMa, iMa1, iMX2z, nx, nX, nxz, thisTypes)
    else:
        #A2X is given do nothing here
        A1XA2X = np.zeros((0,nVar))
        b1Xb2X = np.zeros((0,1))
    
    return [[np.asarray(A1xA2x)], [np.asarray(b1xb2x)]], [[np.asarray(A1xA2X),np.asarray(A1XA2x)], [np.asarray(b1xb2X),np.asarray(b1Xb2x)]], [[np.asarray(A1XA2X)],[np.asarray(b1Xb2X)]] 

################################################################################################################
cpdef returnFirstLargerCy(const double[::1] sortedArr, const double minValue ):
    
    cdef int currentInd
    cdef int lenSA = sortedArr.shape[0]
    
    if minValue >= sortedArr[lenSA-1]:
        return lenSA-1
    
    
    for currentInd in range(lenSA):
        if minValue >= sortedArr[currentInd]:
            return currentInd
################################################################################################################


#@cython.wraparound(True)






