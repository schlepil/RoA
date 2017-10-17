from utils import *
from plotUtils import *

###########################
def workingHelper(aObjAShape):
    aCalcObj, aShape = aObjAShape
    return aCalcObj.convPL(aShape)

###########################
class lqrShapeGen():
    def __init__(self, dynSys, Q=None, R=None, N=None, reshape = True ):
        self.dynSys = dynSys
        self.reshape = reshape
        if Q is None:
            self.Q = np.identity(self.dynSys.nX)
        if R is None:
            self.R = 0.01*np.identity(self.dynSys.nU)
        if N is None:
            self.N = np.zeros((self.dynSys.nX, self.dynSys.nU))
    
    def __call__(self, x0, *args, **kwargs):
        _unused,B0,A0,_unused = self.dynSys.getLin(x0)
        K, S, _unused = crtl.lqr(A0, B0, self.Q, self.R, self.N)
        if self.reshape:
            v,w = np.linalg.eigh(S)
            S = ndot(w.T, np.diag(np.sqrt(v)), w)
            #S = np.eye(S.shape[0])
            
        if 'lastShape' in kwargs.keys():
            thisSize = dp(kwargs['lastShape'][1])
        else:
            thisSize = 1.
        return [S/((np.linalg.det(S))**(1./x0.size)), thisSize]


###########################

class ellipsoidalLyap():
    def __init__(self, dynSys, refTraj, tSteps, shapeGen, initialShape, boundCalcClass, mode='PL', excludeInner=None ):
        #mode LL, QL, PL, QQ, PQ
        #Lets first implement only PL -> Input seperation by hyperplanes and third order dynamic approx
        self.dynSys = dynSys
        self.refTraj = refTraj
        self.tSteps = np.array(tSteps).squeeze()
        self.shapeGen = shapeGen
        self.allShapes = [None for _ in range(self.tSteps.shape[0])]
        self.allShapes[-1] = initialShape
        self.boundCalcClass = boundCalcClass
        self.mode = mode
        
        self.excludeInner = excludeInner
        self.boundCalcClass.excludeInner = self.excludeInner
        
        self.convergenceLim = 0.2
        self.convergenceRate = 0.5
        
        #all Permutations
        self.allInputPermutations = [0,1]
        for _ in range(self.dynSys.nU-1):
            currL = len(self.allInputPermutations)
            for l in range(currL):
                self.allInputPermutations.append(self.allInputPermutations[l].append(0))
                self.allInputPermutations.append(self.allInputPermutations[l].append(1))
        
        #Get the upto second or fourth order
        self.degreeList1 = self.dynSys.polyDegrees[:self.dynSys.polyNumbers[0]]
        if self.mode[0] == 'L':
            self.degreeList2 = self.dynSys.polyDegrees[:self.dynSys.polyNumbers[1]]
        elif self.mode[0] == 'P':
            allDegrees = dp(self.dynSys.polyDegrees)
            xDegree = self.dynSys.polyDegrees[:self.dynSys.polyNumbers[0]]
            nDegree = len(allDegrees)
            for thisN in range(nDegree):
                aDegree = allDegrees[thisN]
                for bDegree in xDegree:
                    thisDegree = aDegree+bDegree
                    if not np.any(np.min(thisDegree == allDegrees, axis=1)):
                        allDegrees.append(thisDegree)
            self.degreeList2 = allDegrees
        else:
            assert 0, 'Not implemented'
        
        self.degreeList1 = list(map(deg2str, self.degreeList1))
        self.degreeList2 = list(map(deg2str, self.degreeList2))
        
        #Helper variable to calculate the polynomial vector in transformed coordinates
        self.transVarsCoefs = [ [] for _ in self.dynSys.polyDegrees ]
        self.transVarsDegrees = [ [] for _ in self.dynSys.polyDegrees ]
        
        for i, aDeg in enumerate(self.dynSys.polyDegrees):
            #Loop over all monomials
            print(aDeg)
            for k in range(self.dynSys.nX):
                #Loop over all variables
                for n in range(aDeg[k]):
                    #Loop over the all powers of this variable
                    #1: First encounter -> Add C[k,:].y
                    if len(self.transVarsDegrees[i])==0:
                        #Get the indeices of C[k,:]
                        #self.transVarsCoefs[i] += [ [[k, kk]] for kk in range(self.dynSys.nX) ]
                        for kk in range(self.dynSys.nX): 
                            self.transVarsCoefs[i].append([[k],[kk]])
                        self.transVarsDegrees[i] += self.dynSys.polyDegrees[:self.dynSys.polyNumbers[0]]
                    #2) The list is non-empty
                    #Multiply all terms in the list with all terms C[k,j].y[j]
                    else:
                        oldCoefs = self.transVarsCoefs[i]
                        oldDegs = self.transVarsDegrees[i]
                        self.transVarsCoefs[i] = []
                        self.transVarsDegrees[i] = []
                        for j in range(self.dynSys.nX):
                            #self.transVarsCoefs[i] += [ aCoef + [[k,j]] for aCoef in oldCoefs ]
                            for thisCoef, thisDeg in zip( oldCoefs, oldDegs ):
                                self.transVarsCoefs[i].append( [thisCoef[0] + [k], thisCoef[1] + [j]] )
                                self.transVarsDegrees[i].append( self.dynSys.polyDegrees[j]+thisDeg )
            #Print
            list(map( lambda T: print(str(T[0])+' ; '+str(T[1])), zip(self.transVarsCoefs[i], self.transVarsDegrees[i]) ))
        #C = np.random.rand(2,2)
        #C = ndot(C.T,C)
        #degsNCoefs2PolyVec(self.dynSys.x, C, self.transVarsCoefs, self.transVarsDegrees)
        
        return None
            
    ###########################
    
    def getLargestFunnel(self, subSteps = 1):
        for k in range(len(self.tSteps)-2,-1,-1):
            self.getLargestRegionN(k, subSteps=subSteps)
        return 0
    
    ###########################
    
    def getLargestRegionN(self, n, subSteps = 1):
        #Check if the system converges for the shapes n, n+1
        #Get the shapes
        tTest = np.linspace(self.tSteps[n], self.tSteps[n+1], subSteps+1, endpoint=False)
        if self.allShapes[n] is None:
            self.allShapes[n] = self.shapeGen(self.refTraj(tTest[0])[0], lastShape=self.allShapes[n+1])
        
        dT = self.tSteps[n+1] - self.tSteps[n]
        Pn = self.allShapes[n][0]
        alphan = self.allShapes[n][1]
        Pn1 = self.allShapes[n+1][0]
        alphan1 = self.allShapes[n+1][1]
        
        #Check initial value
                
        #Calculate P(t) an d/dt P
        dP = 1./dT*(alphan1*Pn1-alphan*Pn)
        
        shapeTest = []
        for aT in tTest:
            #Lyapunov region for this time point
            Pt = Pn*alphan*(1.-(aT-tTest[0])/dT)+Pn1*alphan1*(aT-tTest[0])/dT
            Ct = getP2(Pt)
            Cti = np.linalg.inv(Ct)#getPm2(Pt)
            x0T, xd0T, _ = self.refTraj(aT)
            #Get all combinations of signs
            for aPerm in self.allInputPermutations:
                shapeTest.append( [Pt, Ct, Cti, dP, aT, x0T, xd0T, np.array(aPerm).astype(np.int_)] )
        
        #Solve it
        if not(glbDBG) :
            thisWorkerPool = Pool(processes=numCPU)
            if self.mode == 'PL':
                #allConvergenceRate = thisWorkerPool.map(self.convPL, shapeTest)
                allConvergenceRate = thisWorkerPool.map(workingHelper, zip( [self for _ in range(len(shapeTest))] , shapeTest ) )
                
            else:
                assert 0, 'Not implemented'
        else:
            #for dbg
            allConvergenceRate = map(self.convPL, shapeTest)
        
        minConv = min(allConvergenceRate)
        NNN=1
        if minConv < 1e-3:
            #Cut alphan in half until succesful
            while minConv < 1e-3:
                NNN+=1
                alphaOld = alphan
                alphan = alphan/2.
                
                #Calculate P(t) an d/dt P
                dP = 1./dT*(alphan1*Pn1-alphan*Pn)
                
                shapeTest = []
                for aT in tTest:
                    #Lyapunov region for this time point
                    Pt = Pn*alphan*(1.-(aT-tTest[0])/dT)+Pn1*alphan1*(aT-tTest[0])/dT
                    Ct = getP2(Pt)
                    Cti = np.linalg.inv(Ct)#getPm2(Pt)
                    x0T, xd0T, _ = self.refTraj(aT)
                    #Get all combinations of signs
                    for aPerm in self.allInputPermutations:
                        shapeTest.append( [Pt, Ct, Cti, dP, aT, x0T, xd0T, np.array(aPerm).astype(np.int_)] )
                
                #Solve it
                if not(glbDBG) :
                    thisWorkerPool = Pool(processes=numCPU)
                    if self.mode == 'PL':
                        #allConvergenceRate = thisWorkerPool.map(self.convPL, shapeTest)
                        allConvergenceRate = thisWorkerPool.map(workingHelper, zip( [self for _ in range(len(shapeTest))] , shapeTest ) )
                    else:
                        assert 0, 'Not implemented'
                else:
                    #for dbg
                    allConvergenceRate = map(self.convPL, shapeTest)
                
                minConv = min(allConvergenceRate)
            alphaSafe = alphaOld
            alphaFail = alphan
        else:
            #Cut alphan in half until succesful
            while minConv >= 1e-3:
                NNN+=1
                alphaOld = alphan
                alphan = alphan*2.
                
                #Calculate P(t) an d/dt P
                dP = 1./dT*(alphan1*Pn1-alphan*Pn)
                
                shapeTest = []
                for aT in tTest:
                    #Lyapunov region for this time point
                    Pt = Pn*alphan*(1.-(aT-tTest[0])/dT)+Pn1*alphan1*(aT-tTest[0])/dT
                    Ct = getP2(Pt)
                    Cti = np.linalg.inv(Ct)#getPm2(Pt)
                    x0T, xd0T, _ = self.refTraj(aT)
                    #Get all combinations of signs
                    for aPerm in self.allInputPermutations:
                        shapeTest.append( [Pt, Ct, Cti, dP, aT, x0T, xd0T, np.array(aPerm).astype(np.int_)] )
                
                #Solve it
                if not(glbDBG) :
                    thisWorkerPool = Pool(processes=numCPU)
                    if self.mode == 'PL':
                        #allConvergenceRate = thisWorkerPool.map(self.convPL, shapeTest)
                        allConvergenceRate = thisWorkerPool.map(workingHelper, zip( [self for _ in range(len(shapeTest))] , shapeTest ) )
                    else:
                        assert 0, 'Not implemented'
                else:
                    #for dbg
                    allConvergenceRate = map(self.convPL, shapeTest)
                
                minConv = min(allConvergenceRate)
            alphaSafe = alphan
            alphaFail = alphaOld
            
        #Either perform bisection or just take the last valid value
        #self.allShapes[n][1] = alphaSafe
        alphaU = alphaSafe
        alphaL = alphaFail
        while (2.*abs(alphaU-alphaL))/(alphaU+alphaL) > self.convergenceLim:
            NNN+=1
            alphan = (alphaU+alphaL)/2.
            
            #Calculate P(t) an d/dt P
            dP = 1./dT*(alphan1*Pn1-alphan*Pn)
            
            shapeTest = []
            for aT in tTest:
                #Lyapunov region for this time point
                Pt = Pn*alphan*(1.-(aT-tTest[0])/dT)+Pn1*alphan1*(aT-tTest[0])/dT
                Ct = getP2(Pt)
                Cti = np.linalg.inv(Ct)#getPm2(Pt)
                x0T, xd0T, _ = self.refTraj(aT)
                #Get all combinations of signs
                for aPerm in self.allInputPermutations:
                    shapeTest.append( [Pt, Ct, Cti, dP, aT, x0T, xd0T, np.array(aPerm).astype(np.int_)] )
            
            #Solve it
            if not(glbDBG) :
                thisWorkerPool = Pool(processes=numCPU)
                if self.mode == 'PL':
                    #allConvergenceRate = thisWorkerPool.map(self.convPL, shapeTest)
                    allConvergenceRate = thisWorkerPool.map(workingHelper, zip( [self for _ in range(len(shapeTest))] , shapeTest ) )
                    pass
                else:
                    assert 0, 'Not implemented'
            else:
                #for dbg
                allConvergenceRate = map(self.convPL, shapeTest)
            
            minConv = min(allConvergenceRate)
            if minConv < 1e-3:
                #Converges!
                alphaU = alphan
            else:
                #Diverges!
                alphaL = alphan
                
            
        #Think about how to change shape
        self.allShapes[n][1] = alphaU
        print('It took '+str(NNN)+' iterations')
        return 0
    
###########################
    
    def convPL(self, inList=[None, None, None, None, None, None, None, None]):
        Pt, Ct, Cti, dP, aT, x0T, xd0T, perm = inList
        
        perm.resize((perm.size,))
        
        #get optimal control
        perm[perm==0]=-1
        perm.resize((self.dynSys.nU,))
        ustar = self.dynSys.inputCstr.getThisU(perm)
        
        if 0:
            #Check orig problem
            forig, Borig, polyAorig, _unused = self.dynSys.getTaylorS3(x0T)#this takes too long if evaluated with a given transformation
            worstXorig, worstConvergenceOrig = checkOrigProb( Pt, Ct, dP, aT, x0T, xd0T, perm, forig, Borig, polyAorig, self.dynSys.polyDegrees[:self.dynSys.polyNumbers[0]], self.dynSys.polyDegrees, ustar )
            
        f, B, polyA, _unused = self.dynSys.getTaylorS3(x0T)#this takes too long if evaluated in transformed space
        
        #Clean the dict
        thisDictDeg1 = {k:0. for k in self.degreeList1}
        thisDictDeg2 = {k:0. for k in self.degreeList2}
                
        nX=self.dynSys.nX
        zyDeg = self.dynSys.polyDegrees
        yDeg = self.dynSys.polyDegrees[:self.dynSys.polyNumbers[0]]
        
        Bustar = ndot( B, ustar )
        
        #The total convergence expression in y is
        #Vd = 2*y'.P^1/2.(f(x_r)+Ax.z(P^-1/2.y)+Bustar-xd_r) + y'.P^-1/2'.dP.P^-1/2.y + \gamma*y'.y
        
        #get 2*y'.P^1/2.(f(x_r)+Bustar-xd_r)
        if 1:
            thisVals = 2.*ndot(Ct, (f+Bustar-xd0T))
            for k in range(nX):
                #thisDictDeg1[deg2str(yDeg[k])] += 2.*(f[k]+Bustar[k])
                thisDictDeg1[deg2str(yDeg[k])] += thisVals[k]
        
        #get the imposed minimal conv. rate \gamma*y'.y
        if 1:
            for  k in range(nX):
                thisDictDeg2[deg2str(yDeg[k]+yDeg[k])] += self.convergenceRate
        #get part of time-dependent shape
        #x'.dP.x=y'.P^-1/2'.dP.P^-1/2.y = y'.thisP.y 
        if 1:
            thisP = ndot(Cti.T, dP, Cti)
            for i in range(nX):
                for j in range(nX):
                    thisDictDeg2[deg2str(yDeg[i]+yDeg[j])] += thisP[i,j]
        #get the polynomial approx of dynamics
        #2*y'.P^1/2.A_xr.z(P^-1/2).y)
        #for i in range(nX):
        #    for j in range(self.dynSys.polyNumbers[-1]):
        #        thisDictDeg2[deg2str(yDeg[i]+zyDeg[j])] += 2.*polyA[i,j]
        #With self.transVarsCoefs and self.transVarsDegrees z(P^-1/2).y)[i] has the form
        #z(P^-1/2).y)[i] = \sum_j \prod(self.transVarsCoefs[i][j])*monomial_in_y(self.transVarsDegrees[i][j])
        
        if 1:
            #test
            #zYme = degsNCoefs2PolyVec(self.dynSys.xTrans, Cti, self.transVarsCoefs, self.transVarsDegrees)
            #zY = self.dynSys.getPolyVarsTrans(C=Ct)
            PA = 2.*ndot(Ct, polyA)
            transCoefsNum = [ list(map( lambda thisInd: np.prod(Cti[thisInd]), aMonomial )) for aMonomial in self.transVarsCoefs ]
            uselessCounter=0
            for i in range(nX):
                #Loop over y
                for j in range(self.dynSys.polyNumbers[-1]):
                    #Loop over z
                    for zyDeg, zyCoef in  zip( self.transVarsDegrees[j], transCoefsNum[j] ):
                        thisDictDeg2[deg2str(yDeg[i]+zyDeg)] += PA[i,j]*zyCoef
                        uselessCounter+=1
        
        
        #Do this more efficiently
        CB = ndot(Ct, B)
        thisB = np.zeros((self.dynSys.nU, self.boundCalcClass.numVars)) 
        for k in range(self.dynSys.nU):
            if perm[k] == 1:
                thisB[k,:self.dynSys.nX] =  CB[:,k] #Equivalent to B.T[k,:]
            else:
                thisB[k,:self.dynSys.nX] = -CB[:,k]
        
        #Do some basic testing for the 2D case
        if 0:
            #Check convergence at origin
            #Get the expression in the original space
            forig, Borig, polyAorig, _unused = self.dynSys.getTaylorS3(x0T)#this takes too long if evaluated with a given transformation
            thisBorig0 = np.zeros((self.dynSys.nU, self.dynSys.nX)) 
            for k in range(self.dynSys.nU):
                if perm[k] == 1:
                    thisBorig0[k,:] =  Borig[:,k] #Equivalent to B.T[k,:]
                else:
                    thisBorig0[k,:] = -Borig[:,k]
            #Vd_ustar = 2*x'.Pt.B.u_star = 2*u_star'.B'.Pt'.x = 2*u_star'.B'.Pt.x
            thisBorig = ndot(thisBorig0, Pt)
            
            
            #Vd = 2*y'.P^1/2.(f(x_r)+Ax.z(P^-1/2.y)+Bustar-xd_r) + y'.P^-1/2'.dP.P^-1/2.y + \gamma*y'.y
            y = self.dynSys.xTrans
            zY = self.dynSys.getPolyVarsTrans(C=Ct)
            
            thisBorigY = ndot(Ct, Borig).T
            thisLin0ValY = 0.
            convExprY = 2.*y.T*sMa(Ct)*(sMa(forig+Bustar-xd0T)+sMa(polyAorig)*zY) + y.T*sMa(ndot(Cti.T, dP, Cti))*y + self.convergenceRate*y.T*y
            fig=plt.figure()
            ax = fig.add_subplot(111)
            plotConv(ax, convExprY, self.dynSys.xTrans, np.zeros((self.dynSys.nX,1)), np.eye(self.dynSys.nX), 1., thisB=thisBorigY, thisA=thisLin0ValY, nPX = 15, nPY=15)
            
        
        thisConvBound = self.boundCalcClass(thisDictDeg1, thisDictDeg2, linearCstr=[thisB, np.zeros((self.dynSys.nU,1))])
        
        return float(thisConvBound[0])
        
######################################################

class NCrelax():
    def __init__(self, nX, mode='PL', excludeInner = None):
        self.nX = nX
        self.mode = mode
        
        self.excludeInner = excludeInner
        
        if mode == 'PL':
            self.degrees2ind, self.ind2degrees, self.indMatrixx, self.indMatrixX, self.indMatrixZtilde, self.indMatrixAll, self.indMatrixXtoz, self.dTraceX, self.dTraceZtilde, self.firstCond, self.secondCond, self.powerxz, self.powerZtilde = getVarsForPolyRelax(self.nX)
            self.numVars = self.firstCond[0].shape[1]
            #Store the constraints that will not change
            #Impose unit hypersphere (Condition  tr(X)<=1 <-> \sum_i x_ii**2 <= 1)
            AXU = np.array(self.dTraceX).reshape((1,-1))
            BXU = np.ones((1,1))
            #Problem: zero valued convergence at origin -> exclude inner
            #tr(X)>=.25 <-> -tr(x)<=-0.25
            if not ( self.excludeInner is None ):
                AXUi = -np.array(self.dTraceX).reshape((1,-1))
                BXUi = -self.excludeInner*np.ones((1,1))
            else:
                AXUi = np.zeros((0,AXU.shape[1]))
                BXUi = np.zeros((0,1))
            
            #All the constraints on each variable
            AlimUpLowx, BlimUpLowx, AlimUpLowX, BlimUpLowX, AlimUpLowxz, BlimUpLowxz, AlimUpLowZ, BlimUpLowZ = getBounds( self.nX, self.nX*(self.nX+1)//2, self.numVars, self.indMatrixx, self.indMatrixZtilde, self.powerxz, self.powerZtilde, mode = 'SDR',  )
            #Relations between the variables
            #impose that the quadratic term is smaller than the original
            AZijijXij = np.zeros(( self.nX, self.numVars ))
            BZijijXij = np.zeros(( self.nX, 1 ))
            for k in range(self.nX):
                #Z_ij^ij - X_ij
                AZijijXij[k, self.indMatrixX[k,k]] = -1.
                kz = self.indMatrixXtoz[k,k]
                AZijijXij[k, self.indMatrixZtilde[self.nX+kz, self.nX+kz]] = 1.
            
            #self.AcstrBase, self.BcstrBase = crtGeneralRLTPolyMin( np.vstack((AXU, AlimUpLowx, AlimUpLowX, AlimUpLowxz, AlimUpLowZ, AZijijXij)), np.vstack((BXU, BlimUpLowx, BlimUpLowX, BlimUpLowxz, BlimUpLowZ, BZijijXij)), self.indMatrixAll, self.indMatrixXtoz, self.nX, self.nX*(self.nX+1)//2)
            #test
            self.AcstrBase, self.BcstrBase = crtGeneralRLTPolyMin( np.vstack((AXU, AXUi, AlimUpLowx, AlimUpLowX, AlimUpLowxz, AlimUpLowZ, AZijijXij)), np.vstack((BXU, BXUi, BlimUpLowx, BlimUpLowX, BlimUpLowxz, BlimUpLowZ, BZijijXij)), self.indMatrixAll, self.indMatrixXtoz, self.nX, self.nX*(self.nX+1)//2)
        else:
            assert 0, 'not implemented'
        
            
        return None
    
    def __call__(self, dictLin, dictPoly, **kwargs):
        
        otherDict = {'linearCstr':[np.zeros((0,self.numVars)), np.zeros((0,1))]}
        otherDict.update(kwargs)
        #1 get the objective
        #CVXopt minimizes. but we want to maximize
        objFun = np.zeros((self.numVars,1))
        for deg, degval in dictLin.items():
            objFun[self.degrees2ind[deg]] += degval
        for deg, degval in dictPoly.items():
            objFun[self.degrees2ind[deg]] += degval
        objFun = -objFun
        
        #2 Set up all the constraints
        #This should be done with c/cython
        AallCstr, BallCstr, AminCstr, BminCstr, nrcor = crtAllRLTCstr(self, otherDict['linearCstr'][0], otherDict['linearCstr'][1])
        
        #Solve
        nQuad = self.nX*(self.nX+1)//2
        NNN=0
        while True:
            NNN+=1
            #Approximate solve
            sol = solvers.sdp(matrix(objFun), Gl=sparse(matrix( AminCstr ) ), hl=matrix( BminCstr ), Gs=[sparse(matrix(self.firstCond[0])), sparse(matrix(self.secondCond[0]))], hs=[matrix(self.firstCond[1].reshape((1+self.nX,1+self.nX))), matrix(self.secondCond[1].reshape((1+self.nX+nQuad,1+self.nX+nQuad)))] )
            #Take current solution and check for violated constraints
            thisX = np.array(sol['x']).reshape((-1,1))
            #The bound is only getting tighter by adding additional constraints
            #If the value is already negative -> exit
            if -float(sol['primal objective']) < 1e-3:
                break
            addInd = np.where( (np.dot(AallCstr, thisX) - BallCstr) > 1e-5 )[0]
            if 0:
                print(addInd)
                checkResult(thisX, self.indMatrixZtilde, self.indMatrixXtoz, self.nX)
                
            if addInd.size == 0:
                break
            nrcor = np.unique(np.hstack((nrcor, addInd)))
            #nrcor = np.hstack((nrmin, addInd)) #Already applied cstr should be satisfied
            
            AminCstr = AallCstr[nrcor,:]
            BminCstr = BallCstr[nrcor]
        print('exit after ' + str(NNN)+ 'with '+'{0} of {1}'.format(AminCstr.shape[0], AallCstr.shape[0]))
        #Test
        if 0:
            cInst = couenneInst()
            #get variables
            for k in range(self.nX):
                cInst.addVar()
            #Add linear -> Planes
            cInst.addLinCstr( otherDict['linearCstr'][0][:,:self.nX], otherDict['linearCstr'][1] )
            #Add bounding box
            AbbC = np.vstack((  np.identity(self.nX), -np.identity(self.nX) ))
            BbbC = np.ones((2*self.nX,1))
            cInst.addLinCstr( AbbC, BbbC )
            #Unit sphere (for non-relaxed prob the bb is redundant)
            cInst.addQuadCstr(np.identity(self.nX), np.zeros((self.nX,1)), 1.0)
            #Exclude inner
            if not(self.excludeInner is None):
                cInst.addQuadCstr(-np.identity(self.nX), np.zeros((self.nX,1)), -self.excludeInner)
            
            allDict = {}
            for aKey in set(list(dictLin.keys()) + list(dictPoly.keys())):
                allDict[aKey] = 0.
            for aKey, aVal in dictLin.items():
                allDict[aKey] += aVal
            for aKey, aVal in dictPoly.items():
                allDict[aKey] += aVal
            cInst.addPolyObjasDict(allDict, seek='MAX')
            
            exactBoundPos, optimalSolPos, solTimePos = cInst.solve()
            checkResult(thisX, self.indMatrixZtilde, self.indMatrixXtoz,2)
            print(exactBoundPos)
            print(optimalSolPos)
            print(np.array(optimalSolPos).reshape((2,))-thisX[0:2].squeeze())
            
            if 1.001*(-float(sol['primal objective'])+1e-3) < exactBoundPos[0]:
                print('oops')
            
            
                
        return -float(sol['primal objective']), np.array(sol['x'])    
            
        
        
        
        
        
                                   

        
        