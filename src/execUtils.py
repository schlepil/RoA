from coreUtils import *
from couenneUtils import *

import cvxpy

import os

## Utils that will be imported using exec
#this way all variables that seem to be undefined are known.
#These are like template functions in c++

#NOTE: The variable set denoted Y in the article corresponds to xz

##########################

#iMA is a matrices holding the variable indices:
#
#      |1, x.T, z.T |
#iMA = |x,  X ,  Y  |
#      |z, Y.T,  Z  |
#

####
#Helpers for speed up
nxz = int(len(np.unique(iMA[1:1+nx,1+nx:].reshape((-1,)))))
iMAonlyOnce = -np.ones_like(iMA)
checked = []
for i in range(iMA.shape[0]):
    for j in range(i,iMA.shape[0]):
        k = iMA[i,j]
        if k in checked:
            continue
        checked.append(k)
        thisId = np.where(iMA==k)
        iMAonlyOnce[i,j] = k
##########################

#Cython utils depending on actual prob
toVecInd = lambda X, Ind, nMax=numVars: (toVecI(X,Ind)[:nMax]).reshape((nMax,1))
toMatInd = lambda x, Ind=iMA: toMatI(x.squeeze(), Ind)

#Function to calculate product constraints where both sets contain only 
#constraints linear in x
getRLTxxCyWrap = lambda A1x, b1x, A2x=np.zeros((0,0)), b2x=np.zeros((0,0)): getRLTxxCy(A1x, b1x, A2x, b2x, iMa=iMA, nx=nx)

#Functions to calculate product constraints where the first set contains only 
#constraints linear in x  
noCstr = np.zeros((0,0))
#and the second constraint linear in x \cup X (linear and quadratic in x) 
getRLTxXCyWrap = lambda A1x=noCstr, b1x=noCstr, A1X=noCstr, b1X=noCstr, A2x=noCstr, b2x=noCstr, A2X=np.zeros((0,0)), b2X=np.zeros((0,0)):  lmap( lambda Al: [np.vstack(Al[0]), np.vstack(Al[1])], getRLTGeneralCy(A1x, b1x, A1X, b1X, A2x, b2x, A2X, b2X, iMa=iMA, iMa1=np.zeros((0,0)).astype(np.int_), iMX2z=np.zeros((0,0)).astype(np.int_), nx=nx, nX=nX, nxz=nxz,types1=np.array([1,2]).astype(np.int_), types2=np.array([1,2]).astype(np.int_)) )
#and the second constraint set linear in x \cup X \cup Y (linear, quadratic und cubic in x)
getRLTxxzCyWrap = lambda A1x=noCstr, b1x=noCstr, A1X=noCstr, b1X=noCstr, A2x=noCstr, b2x=noCstr, A2X=np.zeros((0,0)), b2X=np.zeros((0,0)): lmap( lambda Al: [np.vstack(Al[0]), np.vstack(Al[1])], getRLTGeneralCy(A1x, b1x, A1X, b1X, A2x, b2x, A2X, b2X, iMa=iMA, iMa1=iMAonlyOnce, iMX2z=iMX2z, nx=nx, nX=nX, nxz=nxz, types1=np.array([1,3]).astype(np.int_), types2=np.array([1,3]).astype(np.int_)) )

#getSpecifiedRLT( A1,  b1,  A2, b2, iMa, iMa1, iMX2z, nx, nX, types):
getSpecifiedRLTCyWrap = lambda A1, b1, A2=np.zeros((0,0)), b2=np.zeros((0,0)), types=None: lmap( lambda a: np.asarray(a),  getSpecifiedRLT(A1, b1, A2, b2, iMa=iMA, iMa1=iMAonlyOnce, iMX2z=iMX2z, nx=nx, nX=nX, nxz=nxz, types = np.array(types).squeeze().astype(np.int_) ) )
##########################
def getPoly(x):
    #Get the value of the new variables
    x = x.reshape((nx,1))
    X = np.dot(x,x.T)
    z = np.vstack((x,toVec(X)))
    Y = np.identity(1+z.size)
    Y[0,1:] = z.squeeze()
    Y[1:,0] = z.squeeze()
    Y[1:,1:] = np.dot(z,z.T)
    YVec = toVecInd(Y,iMA)
    
    return YVec, Y
##########################    
    
def checkResult(xSol):
    #Get the difference between exact and relaxed solution
    xe = xSol[:nx]
    Xe = np.dot(xe, xe.T)
    ze = np.zeros((nx*(nx+1)//2,1))
    nZ = 0
    for i in range(nx):
        for j in range(i,nx):
            ze[nZ] = Xe[i,j]
            nZ+=1
    Ze = np.dot(ze, ze.T)
    ZeTilde = np.dot( np.vstack((xe,ze)), np.vstack((xe,ze)).T )
    
    XSol = xSol[iMZt[:nx,:nx].reshape((-1,))].reshape((nx,nx)) 
    ZSol = xSol[iMZt[nx:,nx:].reshape((-1,))].reshape((nZ,nZ))
    ZtildeSol = xSol[iMZt.reshape((-1,))].reshape(iMZt.shape)
    
    deltaZtilde = ZtildeSol - ZeTilde
    eigDeltaX = np.linalg.eigvalsh(XSol-Xe)
    eigDeltaZ = np.linalg.eigvalsh(ZSol-Ze)
    eigDeltaZtilde = np.linalg.eigvalsh(ZtildeSol-ZeTilde)
    print('Difference Matrix:')
    print(deltaZtilde)
    print('eigVals:')
    print(eigDeltaX)
    print(eigDeltaZ)
    print(eigDeltaZtilde)
    
    return deltaZtilde, eigDeltaX, eigDeltaZ, eigDeltaZtilde

#####################################################################

def doSortIndices(A):
    #Function to classify constraints / determine their highest order
    iX = np.unique( iMA[0,1+nx:] )
    ixz = np.unique( iMA[1:1+nx, 1+nx:] )
    iZ =  np.unique( iMA[1+nx:, 1+nx:] )
    
    indX = np.any(np.abs(A[:,iX])>1e-13 , 1)
    indxz = np.any(np.abs(A[:,ixz])>1e-13 , 1)
    indZ = np.any(np.abs(A[:,iZ])>1e-13 , 1)
    
    #indOutx = np.logical_not( np.logical_or.reduce([indX, indxz, indZ]) )
    #indOutX = np.logical_and.reduce([np.logical_not( np.logical_or.reduce([indxz, indZ]) ), indX])
    #indOutxz = np.logical_and.reduce([np.logical_not( indZ ), indxz])
    #indoutZ = indZ
    
    return np.where(np.logical_not( np.logical_or.reduce([indX, indxz, indZ]) ))[0], np.where(np.logical_and.reduce([np.logical_not( np.logical_or.reduce([indxz, indZ]) ), indX]))[0], np.where(np.logical_and.reduce([np.logical_not( indZ ), indxz]))[0], np.where(indZ)[0]

################################################################################

def crtGeneralRLTPolyMin(A, B):
    #Pure python implementation
    #of product constraints. Rather slow but no presorting needed
    #TBD this needs to be improved
    
    N = A.shape[1]
    L = A.shape[0]
    M = L*(L+1)//2
    Aout = np.zeros((M,N))
    Bout = np.zeros((M,1))
    
    ZInd = iMA[1+nx:,1+nx:].reshape((-1,))
    xzInd = np.unique(iMA[1:1+nx:,1+nx:].reshape((-1,)))
    XxzInd = iMA[1:1+nx:,1:].reshape((-1,)) 
    ind = 0
    for k in range(L):
        for l in range(k,L):
            
            ak = -A[k,:]
            al = -A[l,:]
            bk =  B[k]
            bl =  B[l]
            
            if (np.any(np.abs(ak[ZInd]) > 1e-13) or np.any(np.abs(al[ZInd]) > 1e-13)):
                #Z*Z can not be represented
                continue 
            if ((np.any(np.abs(ak[xzInd]) > 1e-13) and np.any(np.abs(al[XxzInd]) > 1e-13)) or (np.any(np.abs(al[xzInd]) > 1e-13) and np.any(np.abs(ak[XxzInd]) > 1e-13))):
                #xz*xz, X*xz can not be represented, x*xz can
                continue 
            
            Aout[ind,:] = 0
            #Linear terms -> ALin = bk*al+bl*ak
            Aout[ind,:] = bk*al+ak*bl
            #terms in x and X
            for i in range(nx+nX):
                for j in range(nx+nX):
                    Aout[ind, iMA[1+i,1+j]] += ak[iMA[0,1+i]]*al[iMA[1+j,0]]
            
            doxz = False
            if np.any(np.abs(ak[xzInd]) > 1e-13):
                axz = ak
                ax = al
                doxz = True
            elif np.any(np.abs(al[xzInd]) > 1e-13):
                axz = al
                ax = ak
                doxz = True
            
            #All that is left to treat is the term x*xz
            #x_i*x_j*z_a -> z_b*z_a = Z_ab
            if doxz:
                for ix0 in range(nx):
                    checkedVars = []
                    #ix0 covers all linear terms of one constraint
                    #Take care that no terms appear twice
                    #This is probably the worst part, do differently
                    for ix1 in range(nx):
                        for j in range(nX):
                            thisXZvar = iMA[1+ix1,1+nx+j]
                            if thisXZvar in checkedVars:
                                continue
                            checkedVars.append(thisXZvar)
                            a = iMX2z[ix0,ix1]
                            Aout[ind, iMA[1+nx+a, 1+nx+j]] += ax[iMA[0,1+ix0]]*axz[thisXZvar]
            
            #Constant term
            Bout[ind] = bk*bl
            ind += 1
    
    #Delete unused
    Aout = Aout[:ind,:]
    Bout = Bout[:ind,0].reshape((ind,1))
    #Switch sign to conform with cvx    
    Aout = -Aout
    
    #Append original cstr
    #Aout = np.vstack((A, Aout))
    #Bout = np.vstack((B, Bout))

    return Aout, Bout

#####################################################################
def crtGeneralRLTPolyMin2(A, B):
    #Differing in some implementation details from crtGeneralRLTPolyMin
    #TBD this needs to be improved
    
    N = A.shape[1]
    L = A.shape[0]
    M = L*(L+1)//2
    Aout = np.zeros((M,N))
    Bout = np.zeros((M,1))
    
    ZInd = iMA[1+nx:,1+nx:].reshape((-1,))
    xzInd = np.unique(iMA[1:1+nx:,1+nx:].reshape((-1,)))
    XxzInd = iMA[1:1+nx:,1:].reshape((-1,)) 
    ind = 0
    for k in range(L):
        for l in range(k,L):
            
            ak = -A[k,:]
            al = -A[l,:]
            bk =  B[k]
            bl =  B[l]
            
            if (np.any(np.abs(ak[ZInd]) > 1e-13) or np.any(np.abs(al[ZInd]) > 1e-13)):
                #Z*Z can not be represented
                continue 
            if ((np.any(np.abs(ak[xzInd]) > 1e-13) and np.any(np.abs(al[XxzInd]) > 1e-13)) or (np.any(np.abs(al[xzInd]) > 1e-13) and np.any(np.abs(ak[XxzInd]) > 1e-13))):
                #xz*xz, X*xz can not be represented, x*xz can
                continue 
            
            Aout[ind,:] = 0
            #Linear terms -> ALin = bk*al+bl*ak
            Aout[ind,:] = bk*al+ak*bl
            #terms in x and X
            for i in range(nx+nX):
                for j in range(nx+nX):
                    Aout[ind, iMA[1+i,1+j]] += ak[iMA[0,1+i]]*al[iMA[1+j,0]]
            
            doxz = False
            if np.any(np.abs(ak[xzInd]) > 1e-13):
                axz = ak
                ax = al
                doxz = True
            elif np.any(np.abs(al[xzInd]) > 1e-13):
                axz = al
                ax = ak
                doxz = True
            
            #All that is left to treat is the term x*xz
            #x_i*x_j*z_a -> z_b*z_a = Z_ab
            if doxz:
                for ix0 in range(nx):
                    #ix0 covers all linear terms of one constraint
                    #Take care that no terms appear twice
                    #This is probably the worst part, do differently
                    for ix1 in range(nx):
                        for j in range(nX):
                            thisXZvar = iMAonlyOnce[1+ix1,1+nx+j]
                            if thisXZvar == -1:
                                continue
                            a = iMX2z[ix0,ix1]
                            Aout[ind, iMA[1+nx+a, 1+nx+j]] += ax[iMA[0,1+ix0]]*axz[thisXZvar]
            
            #Constant term
            Bout[ind] = bk*bl
            ind += 1
    
    #Delete unused
    Aout = Aout[:ind,:]
    Bout = Bout[:ind,0].reshape((ind,1))
    #Switch sign to conform with cvx    
    Aout = -Aout
    
    #Append original cstr
    #Aout = np.vstack((A, Aout))
    #Bout = np.vstack((B, Bout))

    return Aout, Bout

#####################################################################
def doSortMatrices(A, B):
    
    iix, iiX, iixz, iiZ = doSortIndices(A)
    
    Ax = A[iix, :]
    Bx = B[iix]
    
    AX = A[iiX, :]
    BX = B[iiX]
    
    Axz = A[iixz, :]
    Bxz = B[iixz]
    
    AZ = A[iiZ, :]
    BZ = B[iiZ]
    
    return Ax, Bx, AX, BX, Axz, Bxz, AZ, BZ    

#####################################################################

def getRLTxx(Ax, Bx ):
    #Improved pure python implementation for constraints C_x \otimes C_x
    Lx, N = Ax.shape
    
    ix = iMA[0,1:1+nx]
    
    Axx = np.zeros((Lx*(Lx+1)//2,N))
    Bxx = np.zeros((Lx*(Lx+1)//2,1))
    
    #Do x \wedge x
    indStore = 0
    #a0.x <= b0
    #a1.x <= b1
    #->
    #(a0*x-b0)*(a1*x-b1)<=0
    #a0*a1*x*x - b1*a0*x - b0*a1*x  + b0*b1 <= 0
    #a0*a1*x*x - b1*a0*x - b0*a1*x <= -b0*b1
    
    for k in range(Lx):
        indAdd = Lx-k
        nNewCstr=Lx-k
        indNewCstr = np.arange(k,Lx)
        indNewStore = np.arange(indStore, indStore+indAdd)
        
        #Multiplied by scalar terms so - b1*a0*x - b0*a1*x 
        Axx[np.ix_(indNewStore, ix)] = - Bx[k]*Ax[np.ix_(indNewCstr, ix)] - np.multiply( np.broadcast_to(Bx[indNewCstr], (nNewCstr,nx)), np.broadcast_to(Ax[k,ix], (nNewCstr,nx)) )
         
        #Get the quadratic terms a0*a1*x*x
        for i in range(nx):
            ii = iMA[0,1+i]
            #get a_0ii*x_ii*a_1ii*x_ii
            Axx[indNewStore, iMA[1+i,1+i]] += Ax[k,ii]*Ax[indNewCstr,ii]
             
        for i in range(nx): 
            for j in range(i+1,nx):
                #get a_0i*x_i*a_1j*x_j + a_0j*x_j*a_1i*x_i
                Axx[indNewStore, iMA[1+i,1+j]] += Ax[k,iMA[0,1+i]]*Ax[indNewCstr, iMA[0,1+j]] + Ax[k,iMA[0,1+j]]*Ax[indNewCstr, iMA[0,1+i]] 
         
        #Get the B with correct sign -b0*b1
        Bxx[indNewStore] = Bx[k]*Bx[indNewCstr]
        #increment
        indStore += indAdd
    
    Axx=-Axx
    
    return Axx, Bxx

#####################################################################

def crtGeneralRLTPolyMinSortedAll(Ax, Bx, AX=None, BX=None, Axz=None, Bxz=None):
    #Pure python implementation to get product constraints for sorted constraint sets
    zzA = np.zeros((0,Ax.shape[1]))
    zzB = np.zeros((0,1))
    
    AX = AX if not(AX is None) else zzA
    Axz = Axz if not(Axz is None) else zzA
    BX = BX if not(BX is None) else zzB
    Bxz = Bxz if not(Bxz is None) else zzB
    
    #Lazy Mans way for the moment before implementing better versions
    getRLTxX = getRLTxxz = getRLTXX = lambda Axi, Bxi, AXi, BXi, *args: crtGeneralRLTPolyMin(np.vstack((Axi,AXi)), np.vstack((Bxi,BXi)))

    #Set of x \otimes x
    Axx, Bxx = getRLTxx(Ax, Bx )
    
    #Set of x \otimes X
    AxX, BxX = getRLTxX(Ax, Bx, AX, BX )
    
    #Set of x \otimes ( x \wedge x )
    Axxx, Bxxx = getRLTxX(Ax, Bx, Axx, Bxx )
    
    #Set of x \otimes xz
    Axxz, Bxxz = getRLTxxz(Ax, Bx, Axz, Bxz )
    
    #Set of x \otimes ( x \wedge X)
    Ax0xX, Bx0xX = getRLTxxz(Ax, Bx, AxX, BxX ) #Produces a lot of cstrs
    
    #Set of ( x \otimes  x ) \wedge X
    Axx0X, Bxx0X = getRLTXX(Axx, Bxx, AX, BX )
    
    #Set of X \otimes X
    AXX, BXX = getRLTXX(AX, BX, AX, BX )
    
    #Set of X \otimes X
    Axxxx, Bxxxx = getRLTXX(Axx, Bxx, Axx, Bxx ) #Produces a lot of cstrs
    
    return  [Axx], [AxX, Axxx], [Axxz, Ax0xX, Axx0X, AXX, Axxxx], [Bxx], [BxX, Bxxx], [Bxxz, Bx0xX, Bxx0X, BXX, Bxxxx]

#####################################################################

def crtGeneralRLTPolyMinSortedAllCy(Axin, Bxin, AXin=None, BXin=None, Axzin=None, Bxzin=None):
    #Efficient cython based implementation of product constraints.
    #Attention: It is not verified whether Axin contains only constraints in x etc.
    #Results are false if this is not true 
    zzA = np.zeros((0,Axin.shape[1]))
    zzB = np.zeros((0,1))
    
    AXin = AXin if not(AXin is None) else zzA
    Axzin = Axzin if not(Axzin is None) else zzA
    BXin = BXin if not(BXin is None) else zzB
    Bxzin = Bxzin if not(Bxzin is None) else zzB
    
    #Atest,Btest = getSpecifiedRLTCyWrap(Axin[[0],:], Bxin[[0],:], Axzin[[0],:], Bxzin[[0],:], [1,3])
    #Atest0,Btest0 = crtGeneralRLTPolyMin2(np.vstack((Axin[[0],:],Axzin[[0],:])), np.vstack((Bxin[[0],:],Bxzin[[0],:])))
    
    #Set of (x, X) \wedge (x, X)
    [Axx, Bxx], [AxX, BxX], [AXX,BXX] = getRLTxXCyWrap(Axin, Bxin, AXin, BXin )
    
    #Set of x \wedge ( x \wedge x )
    #[_,_],[Axxx, Bxxx],[_,_] = getRLTxXCyWrap(Axin, Bxin, A2X=Axx, b2X=Bxx )
    Axxx, Bxxx = getSpecifiedRLTCyWrap(Axin, Bxin, Axx, Bxx, [1,2])
    
    #Set of x \wedge xz
    #[_,_],[Axxz, Bxxz],[_,_] = getRLTxxzCyWrap(Axin, Bxin, A2X=Axzin, b2X=Bxzin )
    Axxz, Bxxz = getSpecifiedRLTCyWrap(Axin, Bxin, Axzin, Bxzin, [1,3])
    
    #Set of x \wedge ( x \wedge X)
    #[_,_],[Ax0xX, Bx0xX],[_,_] = getRLTxxzCyWrap(Axin, Bxin, A2X=AxX, b2X=BxX ) #Produces a lot of cstrs
    Ax0xX, Bx0xX = getSpecifiedRLTCyWrap(Axin, Bxin, AxX, BxX, [1,3])
    
    #This can be done in one call
    
    #Set of ( x \wedge  x ) \wedge X
    #[_,_],[_,_],[Axx0X, Bxx0X] = getRLTxXCyWrap(A1X=Axx, b1X=Bxx, A2X=AXin, b2X=BXin )
    Axx0X, Bxx0X = getSpecifiedRLTCyWrap(Axx, Bxx, AXin, BXin, [2,2] )
    
    #Set of X \wedge X
    #[_,_],[_,_],[AXX, BXX] = getRLTxXCyWrap(A1X=AX, b1X=BX )
    #AXXin, BXXin = getSpecifiedRLTCyWrap(AXin, BXin, [2,2] )
    
    #Set of X \wedge X
    #[_,_],[_,_],[Axxxx, Bxxxx] = getRLTxXCyWrap(A1X=Axx, b1X=Bxx ) #Produces a lot of cstrs
    Axxxx, Bxxxx = getSpecifiedRLTCyWrap(Axx, Bxx, types=[2,2])
    
    return  [Axx], [AxX, Axxx], [Axxz, Ax0xX, Axx0X, AXX, Axxxx], [Bxx], [BxX, Bxxx], [Bxxz, Bx0xX, Bxx0X, BXX, Bxxxx]

#####################################################################
def crtGeneralRLTPolyMinSortedAddLinCy(Axp, Bxp, Ax, Bx, AX=None, BX=None, Axz=None, Bxz=None):
    #An efficient implementation of adding a set of constraints linear in x (i.e. the partitionning of the state-space)
    #to sets of existing constraints and computing all possible product constraints
    zzA = np.zeros((0,Ax.shape[1]))
    zzB = np.zeros((0,1))
    
    AX = AX if not(AX is None) else zzA
    Axz = Axz if not(Axz is None) else zzA
    BX = BX if not(BX is None) else zzB
    Bxz = Bxz if not(Bxz is None) else zzB
    
    #Set of (x, xp) \wedge (xp)
    [Axxp, Bxxp], [Axpxp, Bxpxp], [_,_] = getRLTxXCyWrap(A1x=Ax, b1x=Bx, A1X=Axp, b1X=Bxp, A2x=Axp, b2x=Bxp )
    
    #getSpecifiedRLT( A1,  b1,  A2, b2, types):
    #Set of x \wedge xp \wedge (xp)
    Axxpxp,Bxxpxp = getSpecifiedRLTCyWrap(Axp, Bxp, Axxp, Bxxp, [1,2])
    #Set of x \wedge xp \wedge (xp)
    Axpxpxp,Bxpxpxp = getSpecifiedRLTCyWrap(Axp, Bxp, Axpxp, Bxpxp, [1,2])
    
    #Set of x \wedge x \wedge xp #IS equal to (??) X \wedge xp
    AXxp,BXxp = getSpecifiedRLTCyWrap(Axp, Bxp, AX, BX, [1,2])
    
    #Set xp \wedge [xz, Xxp, xxpxp, xpxpxp]
    AdivZ1, BdivZ1 = getSpecifiedRLTCyWrap(Axp, Bxp, np.vstack([Axz,AXxp,Axxpxp,Axpxpxp]), np.vstack([Bxz,BXxp,Bxxpxp,Bxpxpxp]), [1,3])
    
    #Set x \wedge Xxp
    AXxpx, BXxpx = getSpecifiedRLTCyWrap(Ax, Bx, AXxp, BXxp, [1,3])
        
    return  [Axxp, np.vstack([Axxpxp, Axpxpxp, AXxp]), np.vstack([AdivZ1, AXxpx])], [Bxxp, np.vstack([Bxxpxp, Bxpxpxp, BXxp]), np.vstack([BdivZ1, BXxpx])]

#####################################################################
def getRedundInd( inList ):
    #Get a list of numerically identitical constraints
    A, indList = inList
    
    indList.squeeze()
    
    identicalCstr = [ [] for _ in range(indList.size) ]
    
    for k, aInd in enumerate(indList):
        identicalCstr[k] = np.where(np.all(np.abs( A - A[aInd,:] ) < 1e-10, 1))[0]
    
    return identicalCstr

#####################################################################
def noRedundCstrApprox2(A, b, c=None, indList = None, nx=None, iMa=None):
    #get an heuristic approximation of the nonredudant constraint set
    #Partially taken from noredund from matlabcentral
    #It computes the "most" constraining constraints by computing
    #the dual polytope and use vertices "far away" from the origin
    L = A.shape[0]
    N = A.shape[1]
    
    if (not (c is None)):
        c = np.array(c).reshape((-1,1))
    
    if ( (c is None) or np.any( np.dot( A, c ) > b )):
        #Non c or inappriopriate c given
        Atilde = np.hstack(( np.ones((L,1)), A ))
         
        obj = matrix(0.0, (N+1,1))
        obj[0] = -1.
        
        solLP = solvers.lp( obj, G=sparse(matrix(Atilde)), h=matrix(b) )
        
        c = np.array(solLP['x'][1:]).reshape((-1,1))
    
    # move polytope to contain origin
    bk = b.copy() # preserve
    b = b - np.dot(A,c) # polytope A*x <= b now includes the origin
    # obtain dual polytope vertices: Simply use min max in each direction as approx
    D = np.divide(A , np.tile(b,[1, N]))
    #limMinI = np.argmin(D, axis=0)
    #limMaxI = np.argmax(D, axis=0)
    
    #nr = np.unique( np.hstack(( limMinI, limMaxI )) )
    if indList is None:
        indList = doSortIndices(A, nx, iMa)
    
    taskList = zip( len(indList)*[D], indList )
    
    if glbDBG:
        identicalCstr = list(map(getRedundInd, taskList))
    else:
        #The parallelisation needs to be changed to pipe
        identicalCstr = list(map(getRedundInd, taskList))
    
    identicalCstr = list(itertools.chain.from_iterable(identicalCstr))
    
    nr = np.unique(np.array(list(map( lambda aInd: aInd[0], identicalCstr ))))
    
    An=A[nr,:]
    bn=bk[nr]
    
    return An, bn, nr

############################################

#Construct the upper and lower bounds for each element in the new variable set
#This function depends on couenne to work. However for the example problems they are precomputed
from couenneUtils import *
#AlimUpLowx, BlimUpLowx, AlimUpLowX, BlimUpLowX, AlimUpLowxz, BlimUpLowxz, AlimUpLowZ, BlimUpLowZ = boundCalculator( self.nx, (self.nx*(self.nx+1))//2, self.numVars, self.iMx, self.iMZt, self.powerxz, self.powerZtilde, mode = 'SDR')
def getBounds( mode = 'SDR', doRecompute = False ):
    #Since they are always the same its worth storint them
    thisName = '../varBounds/RES_{0:d}_{1:d}_{2:d}_{3}.pickle'.format(nx, nX, numVars, mode)
    if not doRecompute:
        try:
            with open(thisName, 'rb') as f:
                sol = pickle.load(f)
            return sol['AlimUpLowx'], sol['BlimUpLowx'], sol['AlimUpLowX'], sol['BlimUpLowX'], sol['AlimUpLowxz'], sol['BlimUpLowxz'], sol['AlimUpLowZ'], sol['BlimUpLowZ'] 
        except:
            print('Either recomputing or new demand')
    
    ABBcouenne = np.vstack(( np.identity(nx), -np.identity(nx) ))
    PsphereCouenne = np.identity(nx)
    assert  mode in ('SDR', 'DNN'), 'either SDR or DNN'
    if mode == 'SDR':
        BBBcouenne = np.ones((2*nx,1))
        LsphereCouenne = np.zeros((nx,1))
        csphereCouenne = 1.0
        xU = 1.
        xL = -1.
        XDU = 1.0
        XDL = 0.0
        XOU = 0.5
        XOL = -0.5
    elif mode == 'DNN':
        BBBcouenne = np.vstack(( np.ones((nx,1)), np.zeros((nx,1)) ))
        LsphereCouenne = -np.ones((nx,1))
        csphereCouenne = (1.0-nx)/4.
        xU = 1.
        xL = 0.
        XDU = 1.0
        XDL = 0.0
        XOU = 0.5+0.5/2**0.5
        XOL = 0.
    
    limLower = np.zeros((numVars,))
    limUpper = np.zeros((numVars,))
    
    
    AlimUpLowx = np.zeros(( 2*nx, numVars ))
    BlimUpLowx = np.zeros(( 2*nx, 1 ))
    indx = 0
    AlimUpLowX = np.zeros(( 2*nX, numVars ))
    BlimUpLowX = np.zeros(( 2*nX, 1 ))
    indX = 0
    AlimUpLowxz = np.zeros(( 2*nx*nX, numVars ))
    BlimUpLowxz = np.zeros(( 2*nx*nX, 1 ))
    indxz = 0
    AlimUpLowZ = np.zeros(( 2*(nX*(nX+1)//2), numVars ))
    BlimUpLowZ = np.zeros(( 2*(nX*(nX+1)//2), 1 ))
    #AlimUpLowZ = np.zeros(( 2*(NProb-nx*nx-nx-nx), NProb ))
    #BlimUpLowZ = np.zeros(( 2*(NProb-nx*nx-nx-nx), 1 ))
    indZ = 0
    #x
    for k in range(nx):
        limUpper[iMx[k]] = xU
        limLower[iMx[k]] = xL
        AlimUpLowx[indx, iMx[k]]  = 1.
        BlimUpLowx[indx]  = limUpper[iMx[k]]
        AlimUpLowx[indx+1, iMx[k]]  = -1.
        BlimUpLowx[indx+1]  = -limLower[iMx[k]]
        indx += 2
        
    #X vec(X)=z
    for i in range(nx):
        for j in range(i,nx):
            if i==j:
                limUpper[iMZt[i,j]] = XDU
                limLower[iMZt[i,j]] = XDL
            else:
                limUpper[iMZt[i,j]] = XOU
                limLower[iMZt[i,j]] = XOL
            AlimUpLowX[indX, iMZt[i,j]]  = 1.
            BlimUpLowX[indX]  = limUpper[iMZt[i,j]]
            AlimUpLowX[indX+1, iMZt[i,j]]  = -1.
            BlimUpLowX[indX+1]  = -limLower[iMZt[i,j]]
            indX += 2
    
    
    #Without redundancy in x.z
    checkedVars = []
    for i in range(nx):
        for j in range(nX):
            thisVar = iMZt[i,nx+j]
            if thisVar in checkedVars:
                continue        
            checkedVars.append(thisVar)
            #Min
            thisCInst = couenneInst()
            for k in range(nx):
                thisCInst.addVar()
            thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
            thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
            thisCInst.addObjMono(relaxationClass.powerZtilde[i][nx+j], 'MIN')
            objective, varsVal, solTime = thisCInst.solve()
            limLower[iMZt[i,nx+j]] = objective[0]
            
            #Max
            thisCInst = couenneInst()
            for k in range(nx):
                thisCInst.addVar()
            thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
            thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
            thisCInst.addObjMono(relaxationClass.powerZtilde[i][nx+j], 'MAX')
            objective, varsVal, solTime = thisCInst.solve()
            limUpper[iMZt[i,nx+j]] = objective[0]
            
            AlimUpLowxz[indxz, iMZt[i,nx+j]]  = 1.
            BlimUpLowxz[indxz]  = limUpper[iMZt[i,nx+j]]
            AlimUpLowxz[indxz+1, iMZt[i,nx+j]]  = -1.
            BlimUpLowxz[indxz+1]  = -limLower[iMZt[i,nx+j]]
            indxz += 2
    #Delete unnecessary:
    AlimUpLowxz = AlimUpLowxz[:indxz,:]
    BlimUpLowxz = BlimUpLowxz[:indxz]      


    #new version
    checkedVars = []
    for i in range(nX):
        for j in range(i,nX):
            thisVar = iMZt[nx+i,nx+j]
            if thisVar in checkedVars:
                continue        
            checkedVars.append(thisVar)
            #Min
            thisCInst = couenneInst()
            for k in range(nx):
                thisCInst.addVar()
            thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
            thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
            thisCInst.addObjMono(relaxationClass.powerZtilde[nx+i][nx+j], 'MIN')
            objective, varsVal, solTime = thisCInst.solve()
            limLower[iMZt[nx+i,nx+j]] = objective[0]
            
            #Max
            thisCInst = couenneInst()
            for k in range(nx):
                thisCInst.addVar()
            thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
            thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
            thisCInst.addObjMono(relaxationClass.powerZtilde[nx+i][nx+j], 'MAX')
            objective, varsVal, solTime = thisCInst.solve()
            limUpper[iMZt[nx+i,nx+j]] = objective[0]
            
            AlimUpLowZ[indZ, iMZt[nx+i,nx+j]]  = 1.
            BlimUpLowZ[indZ]  = limUpper[iMZt[nx+i,nx+j]]
            AlimUpLowZ[indZ+1, iMZt[nx+i,nx+j]]  = -1.
            BlimUpLowZ[indZ+1]  = -limLower[iMZt[nx+i,nx+j]]
            indZ += 2
    #Delete unnecessary:
    AlimUpLowZ = AlimUpLowZ[:indZ,:]
    BlimUpLowZ = BlimUpLowZ[:indZ]
    
    with open(thisName, 'wb') as f:
        pickle.dump({'AlimUpLowx':AlimUpLowx, 'BlimUpLowx':BlimUpLowx, 'AlimUpLowX':AlimUpLowX, 'BlimUpLowX':BlimUpLowX, 'AlimUpLowxz':AlimUpLowxz, 'BlimUpLowxz':BlimUpLowxz, 'AlimUpLowZ':AlimUpLowZ, 'BlimUpLowZ':BlimUpLowZ}, f)
        
    return AlimUpLowx, BlimUpLowx, AlimUpLowX, BlimUpLowX, AlimUpLowxz, BlimUpLowxz, AlimUpLowZ, BlimUpLowZ


############################################################
def NCrelaxCalcDirect( dictLin, dictPoly, **kwargs):
    #Do the actual optimization.
    #minimize the linear functional corresponding the best obtainable derivative of
    #the Lyapunov function subject to LMI constraints
    #NCrelaxCalcDirect does this in one shot by taking into account all 
    #linear constraints obtainable
    
    #numVars, degrees2ind, AcstrBase, BcstrBase, indMatrixAll, indMatrixZtilde, indMatrixXtoz, nx, firstCond, secondCond = addInputs
    #The LMI constraints corresponding to 
    #|1, x.T|
    #|x, X  | >> 0
    #and
    #|1, x.T, z.T|
    #|x,  X , Y  | >> 0
    #|z, Y.T, Z  |
    
    allCond = [relaxationClass.firstCond, relaxationClass.secondCond] 
    
    try:
        allCond.append(relaxationClass.thirdCond)
        #Third valid LMI constraint which is redundant but sometimes improves convergence
        #|X  , Y  |
        #|Y.T, Z  | >> 0
    
    except:
        pass
    
    allCondA = lmap(lambda a: a[0], allCond)
    allCondB = lmap(lambda a: a[1], allCond) 
    otherDict = {}
    otherDict.update(kwargs)
    
    if 'linearCstr' in otherDict.keys():
        #2 Set up all the constraints
            
        #New way for creating the RLTs
        #Add the constraints for state-space partitioning and get the products
        ARLTList, BRLTList = crtGeneralRLTPolyMinSortedAddLinCy(Axp=otherDict['linearCstr'][0], Bxp=otherDict['linearCstr'][1], Ax=relaxationClass.AtildeCstrBasex, Bx=relaxationClass.BtildeCstrBasex, AX=relaxationClass.AtildeCstrBaseX, BX=relaxationClass.BtildeCstrBaseX, Axz=relaxationClass.AtildeCstrBasexz, Bxz=relaxationClass.BtildeCstrBasexz)
        AallCstr = np.vstack( [relaxationClass.AtildeCstrBase]+ARLTList )
        BallCstr = np.vstack( [relaxationClass.BtildeCstrBase]+BRLTList )
    else:
        assert 0, 'nope'
        
            
    
    
    #1 get the objective
    #CVXopt minimizes. but we want to maximize
    #Objective function corresponds to the linearised version of the fourth
    #order polynomial
    objFun = np.zeros((numVars,1))
    for deg, degval in dictLin.items():
        objFun[relaxationClass.deg2ind[deg]] += degval
    for deg, degval in dictPoly.items():
        objFun[relaxationClass.deg2ind[deg]] += degval
    objFun = -objFun
    
    
    #Solve
    nQuad = nx*(nx+1)//2
    NNN=0
    
    #Approximate solve
    #Scale the objective because otherwise its fails sometimes
    scaleFac = np.max(np.abs(objFun))
    objFun = objFun/scaleFac
    
    #Get the sparse representation
    AallCstr = sparse(matrix(AallCstr))
    BallCstr = matrix(BallCstr)
    
    #TBD do we need to scale cstrs too?
    sol = solvers.sdp(matrix(objFun), Gl=AallCstr, hl=BallCstr, Gs=allCondA, hs=allCondB)
    assert sol['status']=='optimal', "Optimization failed on non-optimal points"
    
    #Test
    #Compare the relaxation result with the exact result 
    #using couenne
    #solE,posE = exactSol(dictLin, dictPoly, **kwargs)
    #if -float(sol['primal objective'])*scaleFac < solE-1e-2:
    #    print('aaaaaaaaa')
    
    
    return -float(sol['primal objective'])*scaleFac, np.array(sol['x']) 

############################################################

def NCrelaxCalcIter( dictLin, dictPoly, **kwargs):
    #Do the actual optimization.
    #minimize the linear functional corresponding the best obtainable derivative of
    #the Lyapunov function subject to LMI constraints
    #NCrelaxCalcIter uses a heuristic to determine linear constraint susceptible to be important
    #and then iterates adding violated constraints until either no new constraints
    #are violated or the found minimum is larger than 0 (equivalent to the convergence of the system)
    
    #numVars, degrees2ind, AcstrBase, BcstrBase, indMatrixAll, indMatrixZtilde, indMatrixXtoz, nx, firstCond, secondCond = addInputs
    allCond = [relaxationClass.firstCond, relaxationClass.secondCond] 
    
    try:
        allCond.append(relaxationClass.thirdCond)
    except:
        pass
    
    allCondA = lmap(lambda a: a[0], allCond)
    allCondB = lmap(lambda a: a[1], allCond) 
    otherDict = {}
    otherDict.update(kwargs)
    
    if 'linearCstr' in otherDict.keys():
        #2 Set up all the constraints
        if 0:
            AX, Axz, AZ, BX, Bxz, BZ = crtGeneralRLTPolyMinSortedAll(np.vstack((relaxationClass.AcstrBasex, otherDict['linearCstr'][0])), np.vstack((relaxationClass.BcstrBasex, otherDict['linearCstr'][1])), AX=relaxationClass.AcstrBaseX, BX=relaxationClass.BcstrBaseX, Axz=relaxationClass.AcstrBasexz, Bxz=relaxationClass.BcstrBasexz)
            AallCstr = np.vstack([relaxationClass.AcstrBase]+AX+Axz+AZ+[otherDict['linearCstr'][0]])
            BallCstr = np.vstack([relaxationClass.BcstrBase]+BX+Bxz+BZ+[otherDict['linearCstr'][1]])
            AminCstr, BminCstr, nrcor = noRedundCstrApprox(AallCstr, BallCstr)
        elif ('alpha' in otherDict.keys()) and ('polyCstr' in otherDict.keys()) and lyapReg.polyInputSep:
            assert not(otherDict['polyCstr'] is None), "polyCstr not given!"
            #The polynomials in polySet are, in general, already of order 4 -> No constraint product possible
            
            AallCstr = np.vstack( (relaxationClass.AtildeCstrBase, otherDict['polyCstr'][0]) )
            BallCstr = np.vstack( (relaxationClass.BtildeCstrBase, otherDict['polyCstr'][1]) )
            #We want a solution for x that points "the best possible" into the direction of all normals
            #So max sum_i n_i.x
            #with 
            #x.T.P.x <= alpha
            #n_i.x>0
            #With y = C.x this is equal to 
            # max sum n_i.C.y -> min sum nc_i.y which is given 
            #||y||_2 <= alpha^2
            #-n_i.C.y <= 0 -> nc_i.y <= 0 which is given
            G = -matrix([matrix(0.0,(1,nx)), matrix(np.eye(nx))])
            h = matrix([np.sqrt(otherDict['alpha']),matrix(0.0,(nx,1))])
            Glin = otherDict['normalsSet']
            try:
                hlin = matrix(-otherDict['epsDeadZone'], (otherDict['normalsSet'].shape[0],1))
            except KeyError:
                hlin = matrix(-1.0e-4, (otherDict['normalsSet'].shape[0],1))
                
            obj = matrix(np.sum(Glin,0).reshape((-1,1)))
            Glin = matrix(Glin)
            
            if 0:
                solLPstandardSol = solvers.socp(c=obj, Gl=Glin, hl=hlin, Gq=[G], hq=[h])
                assert solLPstandardSol['status']=='optimal', 'Failed to find a LP solution'
                xStandard = np.array(solLPstandardSol['x']).reshape((-1,1))
            else:
                xcvxpy = cvxpy.Variable(nx,1)
                cstr1 = np.array(Glin)*xcvxpy <= np.array(hlin)
                cstr2 = cvxpy.norm(xcvxpy) <= np.sqrt(otherDict['alpha'])
                obj1 = cvxpy.Minimize( cvxpy.max_entries( otherDict['normalsSet']*xcvxpy ))
                prob = cvxpy.Problem(obj1, [cstr1, cstr2])
                prob.solve()
                assert prob.status == 'optimal',  'Failed to find a min solution'
                xStandard = np.array(xcvxpy.value)
                xStandard.resize((nx,1))
            
            xAllStandard, xAllStandardM = getPoly(xStandard)
            assert np.all( ndot(AallCstr, xAllStandard)<=BallCstr+1e-5 ), "Standard solution is violating a constraint"
            
            AminCstr, BminCstr, nrcor = noRedundCstrApprox(AallCstr, BallCstr, c=xAllStandard)
            
        elif ('alpha' in otherDict.keys()) and ('normalsSet' in otherDict.keys()):
            #Do a socp to find "ideal" point
            #AX, Axz, AZ, BX, Bxz, BZ = crtGeneralRLTPolyMinSortedAll(np.vstack((relaxationClass.AcstrBasex, otherDict['linearCstr'][0])), np.vstack((relaxationClass.BcstrBasex, otherDict['linearCstr'][1])), AX=relaxationClass.AcstrBaseX, BX=relaxationClass.BcstrBaseX, Axz=relaxationClass.AcstrBasexz, Bxz=relaxationClass.BcstrBasexz)
            #AallCstr = np.vstack([relaxationClass.AcstrBase]+AX+Axz+AZ+[otherDict['linearCstr'][0]])
            #BallCstr = np.vstack([relaxationClass.BcstrBase]+BX+Bxz+BZ+[otherDict['linearCstr'][1]])
            
            #New way for creating the RLTs
            ARLTList, BRLTList = crtGeneralRLTPolyMinSortedAddLinCy(Axp=otherDict['linearCstr'][0], Bxp=otherDict['linearCstr'][1], Ax=relaxationClass.AtildeCstrBasex, Bx=relaxationClass.BtildeCstrBasex, AX=relaxationClass.AtildeCstrBaseX, BX=relaxationClass.BtildeCstrBaseX, Axz=relaxationClass.AtildeCstrBasexz, Bxz=relaxationClass.BtildeCstrBasexz)
            AallCstr = np.vstack( [relaxationClass.AtildeCstrBase]+ARLTList )
            BallCstr = np.vstack( [relaxationClass.BtildeCstrBase]+BRLTList )
            #We want a solution for x that points "the best possible" into the direction of all normals
            #So max sum_i n_i.x
            #with 
            #x.T.P.x <= alpha
            #n_i.x>0
            #With y = C.x this is equal to 
            # max sum n_i.C.y -> min sum nc_i.y which is given 
            #||y||_2 <= alpha^2
            #-n_i.C.y <= 0 -> nc_i.y <= 0 which is given
            G = -matrix([matrix(0.0,(1,nx)), matrix(np.eye(nx))])
            h = matrix([np.sqrt(otherDict['alpha']),matrix(0.0,(nx,1))])
            Glin = otherDict['normalsSet']
            try:
                hlin = matrix(-otherDict['epsDeadZone'], (otherDict['normalsSet'].shape[0],1))
            except KeyError:
                hlin = matrix(-1.0e-4, (otherDict['normalsSet'].shape[0],1))
                
            obj = matrix(np.sum(Glin,0).reshape((-1,1)))
            Glin = matrix(Glin)
            
            if 0:
                solLPstandardSol = solvers.socp(c=obj, Gl=Glin, hl=hlin, Gq=[G], hq=[h])
                assert solLPstandardSol['status']=='optimal', 'Failed to find a LP solution'
                xStandard = np.array(solLPstandardSol['x']).reshape((-1,1))
            else:
                xcvxpy = cvxpy.Variable(nx,1)
                cstr1 = np.array(Glin)*xcvxpy <= np.array(hlin)
                cstr2 = cvxpy.norm(xcvxpy) <= np.sqrt(otherDict['alpha'])
                obj1 = cvxpy.Minimize( cvxpy.max_entries( otherDict['normalsSet']*xcvxpy ))
                prob = cvxpy.Problem(obj1, [cstr1, cstr2])
                prob.solve()
                assert prob.status == 'optimal',  'Failed to find a min solution'
                xStandard = np.array(xcvxpy.value)
                xStandard.resize((nx,1))
            
            xAllStandard, xAllStandardM = getPoly(xStandard)
            assert np.all( ndot(AallCstr, xAllStandard)<=BallCstr+1e-5 ), "Standard solution is violating a constraint"
            
            AminCstr, BminCstr, nrcor = noRedundCstrApprox(AallCstr, BallCstr, c=xAllStandard)
            
        else:
            AallCstr, BallCstr = crtGeneralRLTPolyMin( np.vstack((relaxationClass.AcstrBase, otherDict['linearCstr'][0])), np.vstack((relaxationClass.BcstrBase, otherDict['linearCstr'][1])))
            AallCstr, BallCstr = np.vstack((relaxationClass.AcstrBase,AallCstr)), np.vstack((relaxationClass.BcstrBase,BallCstr))
            AminCstr, BminCstr, nrcor = noRedundCstrApprox(AallCstr, BallCstr)
    elif 'allCstr' in otherDict.keys():
        AallCstr, BallCstr, AminCstr, BminCstr, nrcor = otherDict['allCstr']
            
    
    
    #1 get the objective
    #CVXopt minimizes. but we want to maximize
    objFun = np.zeros((numVars,1))
    for deg, degval in dictLin.items():
        objFun[relaxationClass.deg2ind[deg]] += degval
    for deg, degval in dictPoly.items():
        objFun[relaxationClass.deg2ind[deg]] += degval
    objFun = -objFun
    
    
    #Solve
    nQuad = nx*(nx+1)//2
    NNN=0
    
    #Approximate solve
    #Scale the objective because otherwise its fails sometimes
    scaleFac = np.max(np.abs(objFun))
    objFun = objFun/scaleFac
    
    #Get the sparse representation
    AminCstr = sparse(matrix(AminCstr))
    AallCstr = sparse(matrix(AallCstr))
    BminCstr = matrix( BminCstr )
    BallCstr = matrix( BallCstr )
    while True:
        NNN+=1
        #TBD do we need to scale cstrs too??!?
        try:
            sol = solvers.sdp(matrix(objFun), Gl=AminCstr, hl=BminCstr, Gs=allCondA, hs=allCondB)
            assert sol['status'] == 'optimal'
        except:
            print("Minimal optimization failed")
            sol = solvers.sdp(matrix(objFun), Gl=AallCstr, hl=BallCstr, Gs=allCondA, hs=allCondB)
            if not(sol['status'] == 'optimal'):
                #Test if solution is valid anyways
                xSol = sol['x']
                assert np.all( np.array( AallCstr*xSol ) <= np.array(BallCstr)+1e-4 )#Allow for small violations
                Zm = toMatI( xSol, iMA )
                Zm[0,0]=1.
                e=np.linalg.eigvalsh(Zm)
                assert np.min(e) >= 1e-4
            break #No need to continue since all constraints are considered
        #Take current solution and check for violated constraints
        thisX = np.array(sol['x']).reshape((-1,1))
        #myCheckSol(thisX, [dictLin, dictPoly])
        #ARLTList, BRLTList = crtGeneralRLTPolyMinSortedAddLinCy(Axp=otherDict['linearCstr'][0], Bxp=otherDict['linearCstr'][1], Ax=relaxationClass.AtildeCstrBasex, Bx=relaxationClass.BtildeCstrBasex, AX=relaxationClass.AtildeCstrBaseX, BX=relaxationClass.BtildeCstrBaseX, Axz=relaxationClass.AtildeCstrBasexz, Bxz=relaxationClass.BtildeCstrBasexz)
        #AallCstr2 = np.vstack( [relaxationClass.AtildeCstrBase]+ARLTList )
        #BallCstr2 = np.vstack( [relaxationClass.BtildeCstrBase]+BRLTList )
        #The bound is only getting tighter by adding additional constraints
        #If the value is already negative -> exit
        print(sol['status'])
        if not(sol['status'] == 'optimal'):
            assert 0, "failed to optimize"
            print('CVX optimization failed')
        if -float(sol['primal objective'])*scaleFac < numericEps:   
            break
        #addInd = np.where( (np.dot(AallCstr, thisX) - BallCstr) > - 1e-3 )[0]
        addInd = np.where(np.array(AallCstr*sol['x']-BallCstr) > -1e-5)[0]
        #Get rid of redundant ind
        addInd = addInd[~np.in1d(addInd,nrcor)]
        #addInd = list(map(int, np.where(np.array(AallCstr*sol['x']-BallCstr) > -1e-3)[0]))
        if 0:
            print(addInd)
            checkResult(thisX)
            
        if addInd.size == 0:
            break
        #nrcor = np.unique(np.hstack((nrcor, addInd)))
        nrcor = np.hstack((nrcor, addInd))
        #nrcor = np.hstack((nrmin, addInd)) #Already applied cstr should be satisfied
        #nrcor = addInd
        nrcorL = [int(i) for i in nrcor]
        AminCstr = AallCstr[nrcorL,:]
        BminCstr = BallCstr[nrcorL]
    
    #Test
    #solE,posE = exactSol(dictLin, dictPoly, **kwargs)
    #if -float(sol['primal objective'])*scaleFac < solE-1e-2:
    #    print('aaaaaaaaa')
    
    print('exit after ' + str(NNN)+ 'with '+'{0} of {1}'.format(AminCstr.size[0], AallCstr.size[0]))
            
    return -float(sol['primal objective'])*scaleFac, np.array(sol['x']) 

###############################################################################
#Decide which calculation method to use
NCrelaxCalc=NCrelaxCalcIter
###############################################################################
#Exact solutions using couenne
#Test
def exactSol(dictLin, dictPoly, **kwargs):
    
    try:
        excludeInner = kwargs['excludeInner']
    except KeyError:
        excludeInner = None
    
    otherDict = {}
    otherDict.update(kwargs)
        
    cInst = couenneInst()
    #get variables
    for k in range(nx):
        cInst.addVar()
    #Add linear -> Planes
    cInst.addLinCstr( otherDict['linearCstr'][0][:,:nx], otherDict['linearCstr'][1] )
    #Add bounding box
    AbbC = np.vstack((  np.identity(nx), -np.identity(nx) ))
    BbbC = np.ones((2*nx,1))
    cInst.addLinCstr( AbbC, BbbC )
    #Unit sphere (for non-relaxed prob the bb is redundant)
    cInst.addQuadCstr(np.identity(nx), np.zeros((nx,1)), 1.0)
    #Exclude inner
    if not(excludeInner is None):
        cInst.addQuadCstr(-np.identity(nx), np.zeros((nx,1)), -excludeInner)
    
    allDict = {}
    for aKey in set(list(dictLin.keys()) + list(dictPoly.keys())):
        allDict[aKey] = 0.
    for aKey, aVal in dictLin.items():
        allDict[aKey] += aVal
    for aKey, aVal in dictPoly.items():
        allDict[aKey] += aVal
    cInst.addPolyObjasDict(allDict, seek='MAX')
    
    exactBoundPos, optimalSolPos, solTimePos = cInst.solve()
    print(exactBoundPos)
    print(optimalSolPos)
    
    return exactBoundPos[0], getPoly(np.array(optimalSolPos))[0]


######################################################
def getExactSolOfOrig(inPut):
    #exactSol optimizes in the transformed y space (the ellipsoid is transformed
    #into a hypersphere). This function operates in the original x space
    xDeg, zxDeg = dynSys.polyDegrees[:dynSys.polyNumbers[0]], dynSys.polyDegrees
    if len(inPut)== 13:
        thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart = inPut
        epsDeadZone = 0.
        innerAlpha = 0.
        excludeInner = None
    elif len(inPut)== 14:
        thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart, epsDeadZone = inPut
        innerAlpha = 0.
        excludeInner = None
    elif len(inPut)== 15:
        thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart, epsDeadZone,innerAlpha = inPut
        excludeInner = None    
    else:
        thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart, epsDeadZone, innerAlpha, excludeInner = inPut
    
    thisDictDeg1 = {k:0. for k in lyapReg.degreeList1}
    thisDictDeg2 = {k:0. for k in lyapReg.degreeList2}
    
    epsDeadZone = 0. if epsDeadZone is None else epsDeadZone
    #get best input given the permutation
    ustar = dynSys.inputCstr.getThisU(perm, t=thisT)
    print(ustar)
    Bustar = ndot( B, ustar )
    
    #The total convergence expression in y is
    #Vd = 2*y'.P^1/2.(f(x_r)+Ax.z(P^-1/2.y)+Bustar-xd_r) + y'.P^-1/2'.dP.P^-1/2.y + \gamma*y'.y
    #Vd = 2.x'.P.(f(x_r)+Ax.z(x)+Bustar-xd_r) + x'.dP.x + \gamma*x'.P.x
    
    #get 2*y'.P^1/2.(f(x_r)+Bustar-xd_r)
    thisVals = 2.*ndot(Pt, (f+Bustar-xd0T))
    for k in range(nx):
        #thisDictDeg1[deg2str(yDeg[k])] += 2.*(f[k]+Bustar[k])
            thisDictDeg1[deg2str(xDeg[k])] += thisVals[k]
    
    #get the imposed minimal conv. rate \gamma*x'.P.x
    thisValP = Pt*lyapReg.convergenceRate
    for  k in range(nx):
        for l in range(nx):
            thisDictDeg2[deg2str(xDeg[k]+xDeg[l])] += thisValP[k,l] 
    #get part of time-dependent shape
    #x'.dP.x 
    for i in range(nx):
        for j in range(nx):
            thisDictDeg2[deg2str(xDeg[i]+xDeg[j])] += dP[i,j]
    #get the polynomial approx of dynamics
    #2*y'.P.A_xr.z(x)
    
    PA = 2.*ndot(Pt, polyA)
    uselessCounter=0
    doPolyB = not ( polyB is None)
    if doPolyB:
        for aB, aU in zip(polyB, ustar):
            PA += 2.*np.dot( Pt, aB*aU )
    for i in range(nx):
        #Loop over y
        for j, zxDeg in enumerate(dynSys.polyDegrees):
            #Loop over z
            thisDictDeg2[deg2str(xDeg[i]+zxDeg)] += PA[i,j]
            uselessCounter+=1
    

    #Do this more efficiently
    #B.resize((nx,nu))
    CB = ndot(Pt, B)
    CB = CB / np.tile(colWiseNorm(CB), (nx, 1))
    thisB = np.zeros((nu, numVars))
    for k in range(nu):
        if perm[k] == 1:
            thisB[k,:nx] =  CB[:,k] #Equivalent to B.T[k,:]
        else:
            thisB[k,:nx] = -CB[:,k]
    
    #Call couenne
    thisB = thisB[:,:nx]; alpha = -(0 if epsDeadZone is None else epsDeadZone)*np.ones((thisB.shape[0],1))
    cInst = couenneInst()
    #get variables
    for k in range(nx):
        cInst.addVar()
    #Add linear -> Planes
    cInst.addLinCstr( thisB, alpha )

    #Ellipsoid (for non-relaxed prob the bb is redundant)
    cInst.addQuadCstr(Pt, np.zeros((nx,1)), 1.0)
    #Exclude inner
    if not(excludeInner is None):
        cInst.addQuadCstr(-Pt, np.zeros((nx,1)), -excludeInner)
    allDict = {}
    for aKey in set(list(thisDictDeg1.keys()) + list(thisDictDeg2.keys())):
        allDict[aKey] = 0.
    for aKey, aVal in thisDictDeg1.items():
        allDict[aKey] += aVal
    for aKey, aVal in thisDictDeg2.items():
        allDict[aKey] += aVal
    cInst.addPolyObjasDict(allDict, seek='MAX')
    
    exactBoundPos, optimalSolPos, solTimePos = cInst.solve()
    
    return exactBoundPos[0], getPoly(np.array(optimalSolPos))[0]
    
    



######################################################
#Functions testing the convergence
#It takes the problem from its defintion in the original space x (ellipsoid)
#to the transformed space y (hypersphere)
def convPL(inPut, calcFun=NCrelaxCalc):
    #Get some shorthands
    yDeg, zyDeg = dynSys.polyDegrees[:dynSys.polyNumbers[0]], dynSys.polyDegrees
    
    thisDictDeg1 = {k:0. for k in lyapReg.degreeList1}
    thisDictDeg2 = {k:0. for k in lyapReg.degreeList2}
    
    thisPID = os.getpid()
    
    if len(inPut)== 13:
        thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart = inPut
        epsDeadZone = 0.
        innerAlpha = 0.
        excludeInner = None
    elif len(inPut)== 14:
        thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart, epsDeadZone = inPut
        innerAlpha = 0.
        excludeInner = None
    elif len(inPut)== 15:
        thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart, epsDeadZone,innerAlpha = inPut
        excludeInner = None    
    else:
        thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart, epsDeadZone, innerAlpha, excludeInner = inPut
    
    epsDeadZone = 0. if epsDeadZone is None else epsDeadZone
    #get best input given the permutation
    ustar = dynSys.inputCstr.getThisU(perm, t=thisT)
    print(ustar)
    Bustar = ndot( B, ustar )
    
    #The total convergence expression in y is
    #Vd = 2*y'.P^1/2.(f(x_r)+Ax.z(P^-1/2.y)+Bustar-xd_r) + y'.P^-1/2'.dP.P^-1/2.y + \gamma*y'.y
    
    #get 2*y'.P^1/2.(f(x_r)+Bustar-xd_r)
    thisVals = 2.*ndot(Ct, (f+Bustar-xd0T))
    for k in range(nx):
        #thisDictDeg1[deg2str(yDeg[k])] += 2.*(f[k]+Bustar[k])
        thisDictDeg1[deg2str(yDeg[k])] += thisVals[k]
    
    #get the imposed minimal conv. rate \gamma*y'.y
    for  k in range(nx):
        thisDictDeg2[deg2str(yDeg[k]+yDeg[k])] += lyapReg.convergenceRate
    #get part of time-dependent shape
    #x'.dP.x=y'.P^-1/2'.dP.P^-1/2.y = y'.thisP.y 
    thisP = ndot(Cti.T, dP, Cti)
    for i in range(nx):
        for j in range(nx):
            thisDictDeg2[deg2str(yDeg[i]+yDeg[j])] += thisP[i,j]
    #get the polynomial approx of dynamics
    #2*y'.P^1/2.A_xr.z(P^-1/2).y)
    #for i in range(nx):
    #    for j in range(self.dynSys.polyNumbers[-1]):
    #        thisDictDeg2[deg2str(yDeg[i]+zyDeg[j])] += 2.*polyA[i,j]
    #With self.transVarsCoefs and self.transVarsDegrees z(P^-1/2).y)[i] has the form
    #z(P^-1/2).y)[i] = \sum_j \prod(self.transVarsCoefs[i][j])*monomial_in_y(self.transVarsDegrees[i][j])
    
    #test
    #zYme = degsNCoefs2PolyVec(self.dynSys.xTrans, Cti, self.transVarsCoefs, self.transVarsDegrees)
    #zY = self.dynSys.getPolyVarsTrans(C=Ct)
    #test
    
    #This actually works because each monomial in x is expanded to the corresponding sum of monomials in y
    #monoX = prod_i x_f(i) <-> monoY =  prod_i Cti[f(i),:].y -> expand to sum of monos
    #TBD this should really be changed because it produces a lot of overhead
    #Idea: Use transformed dynmics with Ct as parameter in sympy and numpify
    PA = 2.*ndot(Ct, polyA)
    transCoefsNum = [ lmap( lambda thisInd: np.prod(Cti[thisInd].squeeze()), aMonomial ) for aMonomial in lyapReg.transVarsCoefs ]#This should be changed it is pretty slow
    uselessCounter=0
    doPolyB = not ( polyB is None)
    if doPolyB:
        for aB, aU in zip(polyB, ustar):
            PA += 2.*np.dot( Ct, aB*aU )
    for i in range(nx):
        #Loop over y
        for j in range(dynSys.polyNumbers[-1]):
            #Loop over z
            for zyDeg, zyCoef in  zip( lyapReg.transVarsDegrees[j], transCoefsNum[j] ):
                #Loop over u: polyB is given, get the non-linear effects of B(x).ustar
                thisDictDeg2[deg2str(yDeg[i]+zyDeg)] += PA[i,j]*zyCoef
                uselessCounter+=1
    

    #Do this more efficiently
    #B.resize((nx,nu))
    CB = ndot(Ct, B)
    CB = CB / np.tile(colWiseNorm(CB), (nx, 1))
    thisB = np.zeros((nu, numVars))
    for k in range(nu):
        if perm[k] == 1:
            thisB[k,:nx] =  CB[:,k] #Equivalent to B.T[k,:]
        else:
            thisB[k,:nx] = -CB[:,k]
    
    
    #If this turns out to be good it needs to be done more efficiently
    if lyapReg.polyInputSep and doPolyB:
        CB = ndot(Ct, B)
        #Do the partitioning based on relaxed polynomials
        thisBpoly = np.zeros((nu, numVars))
        for k, aB in enumerate(polyB):
            #Linear part
            thisBpoly[k,:nx] =  CB[:,k]
            PaB = ndot(Ct, aB)
            for i in range(nx):
                #Loop over y
                for j in range(dynSys.polyNumbers[-1]):
                    #Loop over z
                    for zyDeg, zyCoef in  zip( lyapReg.transVarsDegrees[j], transCoefsNum[j] ):
                        thisBpoly[k, relaxationClass.deg2ind[deg2str(yDeg[i]+zyDeg)] ] += PaB[i,j]*zyCoef
            #Check sign
            if not(perm[k] == 1):
                thisBpoly[k,:] *= -1
            #Normalize
            thisBpoly[k,:] /= np.linalg.norm(thisBpoly[k,:nx])
    else:
        thisBpoly = None
                    
    #Get the cross-product of 
    #crtGeneralRLTPolyMinSortedAll(Ax, Bx, AX=None, BX=None, Axz=None, Bxz=None)
    #The constraints here are all on y; So if thisB.y <= 0 is imposed.
    #if we set thisB.y <= \eps we can test different things depending on eps:
    #eps > 0 A non-checked zone is created around the separation hyperplanes, allowing for divergent behaviour here
    #eps < 0 The different zones overlap and we do not have to switch "infinitely fast when convergin to one of the hyperplanes   
    #if relaxationClass.type == 'NCrelax':
    #    #thisConvBound = NCrelaxCalc(thisDictDeg1, thisDictDeg2, linearCstr=[thisB, np.zeros((nu,1))])
    #    if 0:
    #        thisConvBound = NCrelaxCalc(thisDictDeg1, thisDictDeg2, linearCstr=[thisB, -epsDeadZone*np.ones((nu,1))])
    #    else:
    #        #Use the "ideal" solution
    #        thisConvBound = NCrelaxCalc(thisDictDeg1, thisDictDeg2, linearCstr=[thisB, -epsDeadZone*np.ones((nu,1))], normalsSet=thisB[:,:nx], alpha = (max(epsDeadZone*np.sqrt(2), innerAlpha)+1.)/2., epsDeadZone=epsDeadZone )
    #else:
    #    assert 0, 'Not implemented'
    try:
        #Use the "ideal" solution
        #thisCalcFun = eval(calcFun)
        T=time.time()
        thisConvBound = calcFun(thisDictDeg1, thisDictDeg2, linearCstr=[thisB, -epsDeadZone*np.ones((nu,1))], normalsSet=thisB[:,:nx], alpha = (max(epsDeadZone*np.sqrt(2), 0 if innerAlpha is None else innerAlpha)+1.)/2., epsDeadZone=epsDeadZone, excludeInner=innerAlpha, polyCstr=[thisBpoly, -epsDeadZone*np.ones((nu,1))] )
        print(time.time()-T)
        if glbDBG:
            yy = thisConvBound[1][:nx].reshape((nx,1))
            myCheckSol(thisConvBound[1])
            xx = ndot(Cti, yy)
            print(ndot(xx.T, Pt, xx))
               
            fPoly = dynSys(x0T+xx, ustar, x0=x0T, mode='PP')
            fReal = dynSys(x0T+xx, ustar, mode='OO')
            dVP = 2.*ndot(xx.T,Pt,(fPoly-xd0T))+ndot(xx.T,dP,xx)
            dVR = 2.*ndot(xx.T,Pt,(fReal-xd0T))+ndot(xx.T,dP,xx)
            print("{0} : {1} : {2} : {3}".format(thisConvBound[0], dVP, thisConvBound[0]>=dVP-1.e-2, dVR))
            print("dist from plane is {0} for point x \n {1} \n y \n {2}".format( ndot(CB.T, yy), xx, yy ))
            for testK in range(0):
                thisDX = np.random.rand(nx,1)-0.5
                thisDX = thisDX/np.linalg.norm(thisDX)
                thisDX = ndot(Cti,thisDX*0.01)
                thisX = xx+thisDX
                if (ndot(thisX.T, Pt, thisX)>1. or np.any( ndot(thisB[:,:4], ndot(Ct, thisX)) > -epsDeadZone*np.ones((nu,1))-1e-4)):
                    continue
                fPolyPrime = dynSys(x0T+thisX, ustar, x0=x0T, mode='PP')
                dVPPrime = 2.*ndot(xx.T,Pt,(fPolyPrime-xd0T))+ndot(xx.T,dP,xx)
                print("{0} : {1}".format(dVP, dVPPrime))
                if dVPPrime>dVP-1e-2:
                    print('nooo')
    except:
        assert 0, "Call failed"
    
    #Test get optimal value for original prob
    #if 1:
        #solEOrig,posEOrig = getExactSolOfOrig(inPut)
        #print("Orig: {0}, Relax {1}".format(solEOrig, thisConvBound))
    
    
    return [float(thisConvBound[0]), whichPart]
    
######################################################

def convPLParallel(inQueue, outQueue, aFunc):
    #The parallelized version of convPL
    thisPID = os.getpid()
    while True:
        aInput = inQueue.get()
        if isinstance(aInput, str):
            outQueue.put([aInput, thisPID])
        else:
            outQueue.put( aFunc(aInput) )
    
    return 0

######################################################
def myCheckSol(xSolRelax, valDict=None):
    #The bound obtained by the linearization is strict if the matrix
    #|1, x.T, z.T|
    #|x,  X , Y  |
    #|z, Y.T, Z  |
    #is rank 1. The eigenvalues of
    #|1, x.T, z.T|   |1|
    #|x,  X , Y  | - |x|.|1, x.T, z.T|
    #|z, Y.T, Z  |   |z|
    #is a measure of the tightness
    #deltaZtilde, eigDeltaX, eigDeltaZ, eigDeltaZtilde = checkResult(xSolRelax)
    xSol, xSolM = getPoly(xSolRelax[:nx])
    
    xSolRelaxM = toMatInd(xSolRelax)
    print(np.linalg.eig(xSolRelaxM)[0])
    print(np.linalg.eig(xSolRelaxM[1:,1:]-xSolM[1:,1:])[0])
    
    if not (valDict is None):
        xR = xSolRelax[:nx].squeeze()
        resultRelax = 0.
        for aValDict in valDict:
            for aPower,aCoef in aValDict.items():
                resultRelax += aCoef*np.prod(np.power(xR,str2deg(aPower)))
    else:
        resultRelax=None
    
    return resultRelax
            
    
    
    
    
    