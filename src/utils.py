from coreUtils import *

import problemInvPend as p




def checkResult(xSol, iMZt, iMX2z, nx):
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


def crtAllRLTCstr(aRelaxScheme, Aadd, Badd):
    #get all rlt constraints
    AallCstr, BallCstr = crtGeneralRLTPolyMin(np.vstack((aRelaxScheme.AcstrBase, Aadd)), np.vstack((aRelaxScheme.BcstrBase, Badd)), aRelaxScheme.indMatrixAll, aRelaxScheme.indMatrixXtoz, aRelaxScheme.nX, aRelaxScheme.nX*(aRelaxScheme.nX+1)//2)
    
    #get the "most importat" ones
    AminCstr, BminCstr, nr = noRedundCstrApprox(AallCstr, BallCstr) #get an initial guess for the position
    
    #return
    return AallCstr, BallCstr, AminCstr, BminCstr, nr

#####################################################################

def crtAllRLTCstr_Parallel(AcstrBase, BcstrBase, indMatrixXtoz, nX, indMatrixAll, Aadd, Badd):
    #get all rlt constraints
    AallCstr, BallCstr = crtGeneralRLTPolyMin(np.vstack((AcstrBase, Aadd)), np.vstack((BcstrBase, Badd)), indMatrixAll, indMatrixXtoz, nX, nX*(nX+1)//2)
    
    #append the original cstrs
    AallCstr = np.vstack((AcstrBase, AallCstr))
    BallCstr = np.vstack((BcstrBase, BallCstr))
    
    #get the "most importat" ones
    AminCstr, BminCstr, nr = noRedundCstrApprox(AallCstr, BallCstr) #get an initial guess for the position
    
    #return
    return AallCstr, BallCstr, AminCstr, BminCstr, nr

#####################################################################

def crtAllRLTCstr_Parallel2(AcstrBase, BcstrBase, indMatrixXtoz, nX, indMatrixAll, Aadd, Badd):
    #get all rlt constraints
    AallCstr, BallCstr = crtGeneralRLTPolyMin(np.vstack((AcstrBase, Aadd)), np.vstack((BcstrBase, Badd)), indMatrixAll, indMatrixXtoz, nX, nX*(nX+1)//2)
    
    #append the original cstrs
    AallCstr = np.vstack((AcstrBase, AallCstr))
    BallCstr = np.vstack((BcstrBase, BallCstr))
    
    #remove all duplicata
    AallCstr, BallCstr, _ = noRedundCstrApprox2(AallCstr, BallCstr, nx=nX, iMa=indMatrixAll)
    
    #get the "most importat" ones
    AminCstr, BminCstr, nr = noRedundCstrApprox(AallCstr, BallCstr) #get an initial guess for the position
    
    #return
    return AallCstr, BallCstr, AminCstr, BminCstr, nr

#####################################################################

def crtAllRLTCstr_Parallel3(A, B, indList, indMatrixXtoz, nX, indMatrixAll):
    #get all rlt constraints
    AallCstr, BallCstr = crtGeneralRLTPolyMinSortedAll(A[indList[0],:], B[indList[0]], indMatrixAll, indMatrixXtoz, N2=nX, NX=(nX*(nX+1))//2, AX=A[indList[1],:], BX=B[indList[1]], Axz=A[indList[2],:], Bxz=B[indList[2]])
    
    #append the original cstrs
    AallCstr = np.vstack((A, AallCstr))
    BallCstr = np.vstack((B, BallCstr))
    
    #remove all duplicata
    AallCstr, BallCstr, _ = noRedundCstrApprox2(AallCstr, BallCstr, nx=nX, iMa=indMatrixAll)
    
    #get the "most importat" ones
    AminCstr, BminCstr, nr = noRedundCstrApprox(AallCstr, BallCstr) #get an initial guess for the position
    
    #return
    return AallCstr, BallCstr, AminCstr, BminCstr, nr

#####################################################################

def doSortIndices(A, nx, iMa):
    
    iX = np.unique( iMa[0,1+nx:] )
    ixz = np.unique( iMa[1:1+nx, 1+nx:] )
    iZ =  np.unique( iMa[1+nx:, 1+nx:] )
    
    indX = np.any(np.abs(A[:,iX])>1e-13 , 1)
    indxz = np.any(np.abs(A[:,ixz])>1e-13 , 1)
    indZ = np.any(np.abs(A[:,iZ])>1e-13 , 1)
    
    #indOutx = np.logical_not( np.logical_or.reduce([indX, indxz, indZ]) )
    #indOutX = np.logical_and.reduce([np.logical_not( np.logical_or.reduce([indxz, indZ]) ), indX])
    #indOutxz = np.logical_and.reduce([np.logical_not( indZ ), indxz])
    #indoutZ = indZ
    
    return np.where(np.logical_not( np.logical_or.reduce([indX, indxz, indZ]) ))[0], np.where(np.logical_and.reduce([np.logical_not( np.logical_or.reduce([indxz, indZ]) ), indX]))[0], np.where(np.logical_and.reduce([np.logical_not( indZ ), indxz]))[0], np.where(indZ)[0]

#####################################################################
def doSortMatrices(A, B, nx, iMa):
    
    iix, iiX, iixz, iiZ = doSortIndices(A, nx, iMa)
    
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
def crtGeneralRLTPolyMin(A, B, iMA, iMX2z, N2, NX):
    #It is assumed that all entries corresponding to z.z' are zero
    
    N = A.shape[1]
    L = A.shape[0]
    M = L*(L+1)//2
    Aout = np.zeros((M,N))
    Bout = np.zeros((M,1))
    
    ZInd = iMA[1+N2:,1+N2:].reshape((-1,))
    xzInd = np.unique(iMA[1:1+N2:,1+N2:].reshape((-1,)))
    XxzInd = iMA[1:1+N2:,1:].reshape((-1,)) 
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
            for i in range(N2+NX):
                for j in range(N2+NX):
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
                for ix0 in range(N2):
                    checkedVars = []
                    #ix0 covers all linear terms of one constraint
                    #Take care that no terms appear twice
                    #This is probably the worst part, do differently
                    for ix1 in range(N2):
                        for j in range(NX):
                            thisXZvar = iMA[1+ix1,1+N2+j]
                            if thisXZvar in checkedVars:
                                continue
                            checkedVars.append(thisXZvar)
                            a = iMX2z[ix0,ix1]
                            Aout[ind, iMA[1+N2+a, 1+N2+j]] += ax[iMA[0,1+ix0]]*axz[thisXZvar]
            
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

def getRLTxx(Ax, Bx, iMA, iMX2z, N2, NX ):
    Lx, N = Ax.shape
    
    ix = iMA[0,1:1+N2]
    
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
        Axx[np.ix_(indNewStore, ix)] = - Bx[k]*Ax[np.ix_(indNewCstr, ix)] - np.multiply( np.broadcast_to(Bx[indNewCstr], (nNewCstr,N2)), np.broadcast_to(Ax[k,ix], (nNewCstr,N2)) )
         
        #Get the quadratic terms a0*a1*x*x
        for i in range(N2):
            ii = iMA[0,1+i]
            #get a_0ii*x_ii*a_1ii*x_ii
            Axx[indNewStore, iMA[1+i,1+i]] += Ax[k,ii]*Ax[indNewCstr,ii]
             
        for i in range(N2): 
            for j in range(i+1,N2):
                #get a_0i*x_i*a_1j*x_j + a_0j*x_j*a_1i*x_i
                Axx[indNewStore, iMA[1+i,1+j]] += Ax[k,iMA[0,1+i]]*Ax[indNewCstr, iMA[0,1+j]] + Ax[k,iMA[0,1+j]]*Ax[indNewCstr, iMA[0,1+i]] 
         
        #Get the B with correct sign -b0*b1
        Bxx[indNewStore] = Bx[k]*Bx[indNewCstr]
        #increment
        indStore += indAdd
    
    Axx=-Axx
    
    return Axx, Bxx

#####################################################################

def crtGeneralRLTPolyMinSortedAll(Ax, Bx, iMA, iMX2z, N2, NX, AX=None, BX=None, Axz=None, Bxz=None ):
    
    #Lazy Mans way for the moment before implementing better versions
    getRLTxX = getRLTxxz = getRLTXX = lambda Axi, Bxi, AXi, BXi, *args: crtGeneralRLTPolyMin(np.vstack((Axi,AXi)), np.vstack((Bxi,BXi)), iMA, iMX2z, N2, NX)
    
    
    #Set of x \wedge x
    Axx, Bxx = getRLTxx(Ax, Bx, iMA, iMX2z, N2, NX )
    
    #Set of x \wedge X
    AxX, BxX = getRLTxX(Ax, Bx, AX, BX, iMA, iMX2z, N2, NX )
    
    #Set of x \wedge ( x \wedge x )
    Axxx, Bxxx = getRLTxX(Ax, Bx, Axx, Bxx, iMA, iMX2z, N2, NX )
    
    #Set of x \wedge xz
    Axxz, Bxxz = getRLTxxz(Ax, Bx, Axz, Bxz, iMA, iMX2z, N2, NX )
    
    #Set of x \wedge ( x \wedge X)
    Ax0xX, Bx0xX = getRLTxxz(Ax, Bx, AxX, BxX, iMA, iMX2z, N2, NX )
    
    #Set of ( x \wedge  x ) \wedge X
    Axx0X, Bxx0X = getRLTXX(Axx, Bxx, AX, BX, iMA, iMX2z, N2, NX )
    
    #Set of X \wedge X
    AXX, BXX = getRLTXX(AX, BX, AX, BX, iMA, iMX2z, N2, NX )
    
    #Set of X \wedge X
    Axxxx, Bxxxx = getRLTXX(Axx, Bxx, Axx, Bxx, iMA, iMX2z, N2, NX )
    
    return [[Axx, Bxx]], [[AxX, BxX], [Axxx, Bxxx]], [[Axxz, Bxxz], [Ax0xX, Bx0xX], [Axx0X, Bxx0X], [AXX, BXX], [Axxxx, Bxxxx]], '[Axx, Bxx], [[AxX, BxX], [Axxx, Bxxx]], [[Axxz, Bxxz], [Ax0xX, Bx0xX], [Axx0X, Bxx0X], [AXX, BXX], [Axxxx, Bxxxx]]' 

#####################################################################


def getRedundInd( inList ):
    
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
        identicalCstr = thisWorkers.map(getRedundInd, taskList)
    
    identicalCstr = list(itertools.chain.from_iterable(identicalCstr))
    
    nr = np.unique(np.array(list(map( lambda aInd: aInd[0], identicalCstr ))))
    
    An=A[nr,:]
    bn=bk[nr]
    
    return An, bn, nr

############################################
#Construct the upper and lower bounds for each element in
from couenneUtils import *

def getBounds( N2, NX, NProb, indMatrixx, indMatrixZtilde, powerxz, powerZtilde, mode = 'SDR', doRecompute = False ):
    #Since they are always the same its worth storint them
    thisName = 'RES_{0:d}_{1:d}_{2:d}_{3}.pickle'.format(N2, NX, NProb, mode)
    if not doRecompute:
        try:
            with open(thisName, 'rb') as f:
                sol = pickle.load(f)
            return sol['AlimUpLowx'], sol['BlimUpLowx'], sol['AlimUpLowX'], sol['BlimUpLowX'], sol['AlimUpLowxz'], sol['BlimUpLowxz'], sol['AlimUpLowZ'], sol['BlimUpLowZ'] 
        except:
            print('Either recomputing or new demand')
    
    ABBcouenne = np.vstack(( np.identity(N2), -np.identity(N2) ))
    PsphereCouenne = np.identity(N2)
    assert  mode in ('SDR', 'DNN'), 'either SDR or DNN'
    if mode == 'SDR':
        BBBcouenne = np.ones((2*N2,1))
        LsphereCouenne = np.zeros((N2,1))
        csphereCouenne = 1.0
        xU = 1.
        xL = -1.
        XDU = 1.0
        XDL = 0.0
        XOU = 0.5
        XOL = -0.5
    elif mode == 'DNN':
        BBBcouenne = np.vstack(( np.ones((N2,1)), np.zeros((N2,1)) ))
        LsphereCouenne = -np.ones((N2,1))
        csphereCouenne = (1.0-N2)/4.
        xU = 1.
        xL = 0.
        XDU = 1.0
        XDL = 0.0
        XOU = 0.5+0.5/2**0.5
        XOL = 0.
    
    limLower = np.zeros((NProb,))
    limUpper = np.zeros((NProb,))
    
    
    AlimUpLowx = np.zeros(( 2*N2, NProb ))
    BlimUpLowx = np.zeros(( 2*N2, 1 ))
    indx = 0
    AlimUpLowX = np.zeros(( 2*NX, NProb ))
    BlimUpLowX = np.zeros(( 2*NX, 1 ))
    indX = 0
    AlimUpLowxz = np.zeros(( 2*N2*NX, NProb ))
    BlimUpLowxz = np.zeros(( 2*N2*NX, 1 ))
    indxz = 0
    AlimUpLowZ = np.zeros(( 2*(NX*(NX+1)//2), NProb ))
    BlimUpLowZ = np.zeros(( 2*(NX*(NX+1)//2), 1 ))
    #AlimUpLowZ = np.zeros(( 2*(NProb-N2*NX-NX-N2), NProb ))
    #BlimUpLowZ = np.zeros(( 2*(NProb-N2*NX-NX-N2), 1 ))
    indZ = 0
    #x
    for k in range(N2):
        limUpper[indMatrixx[k]] = xU
        limLower[indMatrixx[k]] = xL
        AlimUpLowx[indx, indMatrixx[k]]  = 1.
        BlimUpLowx[indx]  = limUpper[indMatrixx[k]]
        AlimUpLowx[indx+1, indMatrixx[k]]  = -1.
        BlimUpLowx[indx+1]  = -limLower[indMatrixx[k]]
        indx += 2
        
    #X vec(X)=z
    for i in range(N2):
        for j in range(i,N2):
            if i==j:
                limUpper[indMatrixZtilde[i,j]] = XDU
                limLower[indMatrixZtilde[i,j]] = XDL
            else:
                limUpper[indMatrixZtilde[i,j]] = XOU
                limLower[indMatrixZtilde[i,j]] = XOL
            AlimUpLowX[indX, indMatrixZtilde[i,j]]  = 1.
            BlimUpLowX[indX]  = limUpper[indMatrixZtilde[i,j]]
            AlimUpLowX[indX+1, indMatrixZtilde[i,j]]  = -1.
            BlimUpLowX[indX+1]  = -limLower[indMatrixZtilde[i,j]]
            indX += 2
    
    #x.z'
    if 0:
        #old version with redundant
        for i in range(N2):
            for j in range(NX):
                #Min
                thisCInst = couenneInst()
                for k in range(N2):
                    thisCInst.addVar()
                thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
                thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
                thisCInst.addObjMono(powerZtilde[i][N2+j], 'MIN')
                objective, varsVal, solTime = thisCInst.solve()
                limLower[indMatrixZtilde[i,N2+j]] = objective[0]
                
                #Max
                thisCInst = couenneInst()
                for k in range(N2):
                    thisCInst.addVar()
                thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
                thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
                thisCInst.addObjMono(powerZtilde[i][N2+j], 'MAX')
                objective, varsVal, solTime = thisCInst.solve()
                limUpper[indMatrixZtilde[i,N2+j]] = objective[0]
                
                AlimUpLowxz[indxz, indMatrixZtilde[i,N2+j]]  = 1.
                BlimUpLowxz[indxz]  = limUpper[indMatrixZtilde[i,N2+j]]
                AlimUpLowxz[indxz+1, indMatrixZtilde[i,N2+j]]  = -1.
                BlimUpLowxz[indxz+1]  = -limLower[indMatrixZtilde[i,N2+j]]
                indxz += 2
    else:
        #Without redundancy in x.z
        checkedVars = []
        for i in range(N2):
            for j in range(NX):
                thisVar = indMatrixZtilde[i,N2+j]
                if thisVar in checkedVars:
                    continue        
                checkedVars.append(thisVar)
                #Min
                thisCInst = couenneInst()
                for k in range(N2):
                    thisCInst.addVar()
                thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
                thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
                thisCInst.addObjMono(powerZtilde[i][N2+j], 'MIN')
                objective, varsVal, solTime = thisCInst.solve()
                limLower[indMatrixZtilde[i,N2+j]] = objective[0]
                
                #Max
                thisCInst = couenneInst()
                for k in range(N2):
                    thisCInst.addVar()
                thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
                thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
                thisCInst.addObjMono(powerZtilde[i][N2+j], 'MAX')
                objective, varsVal, solTime = thisCInst.solve()
                limUpper[indMatrixZtilde[i,N2+j]] = objective[0]
                
                AlimUpLowxz[indxz, indMatrixZtilde[i,N2+j]]  = 1.
                BlimUpLowxz[indxz]  = limUpper[indMatrixZtilde[i,N2+j]]
                AlimUpLowxz[indxz+1, indMatrixZtilde[i,N2+j]]  = -1.
                BlimUpLowxz[indxz+1]  = -limLower[indMatrixZtilde[i,N2+j]]
                indxz += 2
        #Delete unnecessary:
        AlimUpLowxz = AlimUpLowxz[:indxz,:]
        BlimUpLowxz = BlimUpLowxz[:indxz]      
    
    
    #z.z'
    if 0:
        #old version with redundant vars
        for i in range(NX):
            for j in range(i,NX):   
                #Min
                thisCInst = couenneInst()
                for k in range(N2):
                    thisCInst.addVar()
                thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
                thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
                thisCInst.addObjMono(powerZtilde[N2+i][N2+j], 'MIN')
                objective, varsVal, solTime = thisCInst.solve()
                limLower[indMatrixZtilde[N2+i,N2+j]] = objective[0]
                
                #Max
                thisCInst = couenneInst()
                for k in range(N2):
                    thisCInst.addVar()
                thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
                thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
                thisCInst.addObjMono(powerZtilde[N2+i][N2+j], 'MAX')
                objective, varsVal, solTime = thisCInst.solve()
                limUpper[indMatrixZtilde[N2+i,N2+j]] = objective[0]
                
                AlimUpLowZ[indZ, indMatrixZtilde[N2+i,N2+j]]  = 1.
                BlimUpLowZ[indZ]  = limUpper[indMatrixZtilde[N2+i,N2+j]]
                AlimUpLowZ[indZ+1, indMatrixZtilde[N2+i,N2+j]]  = -1.
                BlimUpLowZ[indZ+1]  = -limLower[indMatrixZtilde[N2+i,N2+j]]
                indZ += 2
    else:
        #new version
        checkedVars = []
        for i in range(NX):
            for j in range(i,NX):
                thisVar = indMatrixZtilde[N2+i,N2+j]
                if thisVar in checkedVars:
                    continue        
                checkedVars.append(thisVar)
                #Min
                thisCInst = couenneInst()
                for k in range(N2):
                    thisCInst.addVar()
                thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
                thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
                thisCInst.addObjMono(powerZtilde[N2+i][N2+j], 'MIN')
                objective, varsVal, solTime = thisCInst.solve()
                limLower[indMatrixZtilde[N2+i,N2+j]] = objective[0]
                
                #Max
                thisCInst = couenneInst()
                for k in range(N2):
                    thisCInst.addVar()
                thisCInst.addQuadCstr(PsphereCouenne, LsphereCouenne, csphereCouenne)
                thisCInst.addLinCstr(ABBcouenne, BBBcouenne)
                thisCInst.addObjMono(powerZtilde[N2+i][N2+j], 'MAX')
                objective, varsVal, solTime = thisCInst.solve()
                limUpper[indMatrixZtilde[N2+i,N2+j]] = objective[0]
                
                AlimUpLowZ[indZ, indMatrixZtilde[N2+i,N2+j]]  = 1.
                BlimUpLowZ[indZ]  = limUpper[indMatrixZtilde[N2+i,N2+j]]
                AlimUpLowZ[indZ+1, indMatrixZtilde[N2+i,N2+j]]  = -1.
                BlimUpLowZ[indZ+1]  = -limLower[indMatrixZtilde[N2+i,N2+j]]
                indZ += 2
        #Delete unnecessary:
        AlimUpLowZ = AlimUpLowZ[:indZ,:]
        BlimUpLowZ = BlimUpLowZ[:indZ]
    
    with open(thisName, 'wb') as f:
        pickle.dump({'AlimUpLowx':AlimUpLowx, 'BlimUpLowx':BlimUpLowx, 'AlimUpLowX':AlimUpLowX, 'BlimUpLowX':BlimUpLowX, 'AlimUpLowxz':AlimUpLowxz, 'BlimUpLowxz':BlimUpLowxz, 'AlimUpLowZ':AlimUpLowZ, 'BlimUpLowZ':BlimUpLowZ}, f)
        
    return AlimUpLowx, BlimUpLowx, AlimUpLowX, BlimUpLowX, AlimUpLowxz, BlimUpLowxz, AlimUpLowZ, BlimUpLowZ

############################################################

def checkOrigProb(Pt, Ct, dP, aT, x0T, xd0T, perm, forig, Borig, polyAorig, x, z, ustar):
    dim  = len(x)
    
    #Get Couenne
    cInst = couenneInst()
    for k in range(dim):
        cInst.addVar()
    
    #get Couenne varNames:
    varsNames = cInst.varsName
    sympyVars = sMa(sympy.symbols(varsNames))
    #Map x to deltaX
    #deltaVars = [ '({0}-{1:.64f})'.format(varsNames[k], float(x0T[k])) for k in range(len(varsNames)) ]
    deltaVars = sympyVars - x0T
    #xString = [ deg2str(aDeg, deltaVars) for aDeg in x]
    #zString = [ deg2str(aDeg, deltaVars) for aDeg in z]
    deltaVarsx = sM1(len(x), 1)
    deltaVarsz = sM1(len(z), 1)
    
    for k,deg in enumerate(x):
        for i in range(deg.size):
            for j in range(deg[i]):
                deltaVarsx[k] *= deltaVars[i]
    for k,deg in enumerate(z):
        for i in range(deg.size):
            for j in range(deg[i]):
                deltaVarsz[k] *= deltaVars[i]
        deltaVarsz[k] = sympy.expand(deltaVarsz[k])
    
    #Set constraints
    #1 Ellipsoid
    #V = (x-x_r)'.P.(x-x_r) = x'.P.x-2x_r.P.x+x_r.P.x_r
    cInst.addQuadCstr( Pt, -2.*ndot(x0T.T, Pt), 1.0-float(ndot(x0T.T, Pt, x0T)) )#Equivalent to translated ellipsoid
    #Test whether the result can be found more easily if midpoint is excluded
    if 1:
        cInst.addQuadCstr( -Pt, 2.*ndot(x0T.T, Pt), -0.1+float(ndot(x0T.T, Pt, x0T)) )
    
    #2 Hyperplanes; (x-x_r)'*P*B*u
    #Denote Btilde = (P*B)'
    #Then Btilde[i,:].x<=Btilde[i,:].x_r -> Hyperplane going through x_r
    Btilde = ndot(Pt, Borig).T
    for k in range(Btilde.shape[0]):
        if perm[k] == 1:
            cInst.addLinCstr(Btilde[k,:].reshape((1,-1)), np.array([np.inner(Btilde[k,:], x0T.squeeze())]))
        else:
            cInst.addLinCstr(-Btilde[k,:].reshape((1,-1)), np.array([np.inner(-Btilde[k,:], x0T.squeeze())]))
    
    #Get the objective as string
    #V_d = 2(x-x_r)'.P.(f(x_r) + A_x.z_x + B_x.u^* - xd_r) + (x-x_r)'.Pd.(x-x_r)
    #Attention! Automatic conversion from np.array to sympy.Matrix!
    objExpr = 2*deltaVarsx.T*Pt*(forig + polyAorig*deltaVarsz + ndot(Borig,ustar)) + deltaVarsx.T*dP*deltaVarsx 
    #Check if the default accuracy is enough; precision seems not to work
    #objExprStr = sympy.mathematica_code(sympy.expand(objExpr[0]))
    objExprStr = str(sympy.expand(objExpr[0]))
    cInst.addObjStr(objExprStr, seek='MAX') #Seek the max <-> get worst convergence / highest divergence
    
    #Solve
    objective, varsVal, solTime = cInst.solve()
    
    return objective, np.array(varsVal).reshape((-1,1))

###############################################################################
def degsNCoefs2PolyVec(x, C, coefs, degs):
    
    transCoefsNum = [ map( lambda thisInd: np.prod(C[thisInd]), aMonomial ) for aMonomial in coefs ]
    z = sMa(np.zeros((len(coefs),1)))
    
    for k, (allCoef, allDeg) in enumerate(zip(transCoefsNum, degs)):
        for aCoef, aDeg in zip(allCoef, allDeg):
            thisExpr = 1.
            for thisX, thisDeg in zip(x, aDeg):
                if not(thisDeg==0):
                    thisExpr *= thisX**int(thisDeg)
            z[k] += aCoef*thisExpr
    
    z = sympy.expand(z)
    return z

###############################################################################


    
    
    
    




