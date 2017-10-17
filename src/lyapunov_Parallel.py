from coreUtils import *
from plotUtils import *
import time
###########################

class ellipsoidalLyap():
    def __init__(self, dynSys, refTraj, tSteps, shapeGen, initialShape, boundCalcClass, mode='PL', excludeInner=None, ellipInterPol=standardInterpol ):
        #mode LL, QL, PL, QQ, PQ
        #Lets first implement only PL -> Input seperation by hyperplanes and third order dynamic approx
        
        #How the ellipsoids are interpolated
        #0: Linear interpolation
        #1: tbd Linear interpolation of the cholesky fact
        #self.ellipInterpMode = 0
        self.ellipInterp = ellipInterPol
        
        self.calculateFunction = None
        
        #New stuff for boundary checking
        self.doSweep = None #Either none or sorted 1d array
        self.alphaSearchFact = 2.
        self.polyInputSep = False
        
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
        
        #Check around the seperating hyperplanes
        self.epsDeadZone = 0.
        
        #all Permutations
        self.allInputPermutations = [[0],[1]]
        for _ in range(self.dynSys.nu-1):
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
            for k in range(self.dynSys.nx):
                #Loop over all variables
                for n in range(aDeg[k]):
                    #Loop over the all powers of this variable
                    #1: First encounter -> Add C[k,:].y
                    if len(self.transVarsDegrees[i])==0:
                        #Get the indices of C[k,:]
                        #self.transVarsCoefs[i] += [ [[k, kk]] for kk in range(self.dynSys.nx) ]
                        for kk in range(self.dynSys.nx): 
                            self.transVarsCoefs[i].append([[k],[kk]])
                        self.transVarsDegrees[i] += self.dynSys.polyDegrees[:self.dynSys.polyNumbers[0]]
                    #2) The list is non-empty
                    #Multiply all terms in the list with all terms C[k,j].y[j]
                    else:
                        oldCoefs = self.transVarsCoefs[i]
                        oldDegs = self.transVarsDegrees[i]
                        self.transVarsCoefs[i] = []
                        self.transVarsDegrees[i] = []
                        for j in range(self.dynSys.nx):
                            #self.transVarsCoefs[i] += [ aCoef + [[k,j]] for aCoef in oldCoefs ]
                            for thisCoef, thisDeg in zip( oldCoefs, oldDegs ):
                                self.transVarsCoefs[i].append( [thisCoef[0] + [k], thisCoef[1] + [j]] )
                                self.transVarsDegrees[i].append( self.dynSys.polyDegrees[j]+thisDeg )
            #Print
            list(map( lambda T: print(str(T[0])+' ; '+str(T[1])), zip(self.transVarsCoefs[i], self.transVarsDegrees[i]) ))
        
        return None
            
    ###########################
    def getEllip(self, t):
        #returns the center shape and derivate of the ellipsoid 
        #Call this only when the funnel is already constructed
        
        #t = np.maximum( np.minimum(t, self.refTraj.tMax), self.refTraj.tMin )
        
        #get when 
        #i=0
        #while (self.tSteps[i+1] < t):
        #    i+=1
        #    if (i+2)==self.tSteps.size:
        #        break
        t=np.maximum(np.minimum(t, np.nextafter(self.tSteps[-1],self.tSteps[-1]-1.)), np.nextafter(self.tSteps[0],self.tSteps[0]+1.))
        i = np.searchsorted(self.tSteps, t)-1#np.argmax(self.tSteps>t)-1
        if i==-1:
            alphan = self.allShapes[0][1]
            alphan1 = self.allShapes[0][1]
        
            Pn = self.allShapes[0][0]
            Pn1 = self.allShapes[0][0]
        
            listEllip = self.ellipInterp(Pn, alphan, Pn1, alphan1, t, self.tSteps[0], self.tSteps[0]+1.)
        elif i==self.tSteps.size-1:
            alphan = self.allShapes[-1][1]
            alphan1 = self.allShapes[-1][1]
        
            Pn = self.allShapes[-1][0]
            Pn1 = self.allShapes[-1][0]
        
            listEllip = self.ellipInterp(Pn, alphan, Pn1, alphan1, t, self.tSteps[-1], self.tSteps[-1]+1.)
        else:
            alphan = self.allShapes[i][1]
            alphan1 = self.allShapes[i+1][1]
            
            Pn = self.allShapes[i][0]
            Pn1 = self.allShapes[i+1][0]
            
            listEllip = self.ellipInterp(Pn, alphan, Pn1, alphan1, t, self.tSteps[i], self.tSteps[i+1])
        
        #To be removed
        #if self.ellipInterpMode == 0:
        #    P, dP = standardInterpol(Pn, alphan, Pn1, alphan1, t, self.tSteps[i], self.tSteps[i+1])
        #    return [self.refTraj.xref(t), P, dP]
        #elif self.ellipInterpMode == 1:
        #    Cn = np.linalg.cholesky(Pn)
        #    Cn1 = np.linalg.cholesky(Pn1)
        #    P, dP = cholInterpol(Cn, alphan, Cn1, alphan1, t, self.tSteps[i], self.tSteps[i+1])
        #    return [self.refTraj.xref(t), P, dP]
        #else:
        #    assert 0, "The mode {0} does not correspond to sth implemented".format(self.ellipInterpMode)
        
        return  [self.refTraj.xref(t).reshape((self.dynSys.nx,1))]+listEllip
        
    ###########################
    
    def getCost(self, tVec, pt, thisDim = None):
        tVec = np.array(tVec)
        tVec.resize(tVec.size,)
        singleTime = (tVec.size==1)
        if thisDim is None:
            cost = np.zeros(pt.shape[1],)
            if singleTime:
                p0, P, _ = self.getEllip(tVec[0])
            for k in range(pt.shape[1]):
                if not singleTime:
                    t = tVec[k]
                    p0, P, _ = self.getEllip(t)
                thisV = pt[:,[k]]-p0
                cost[k] = ndot( thisV.T, P, thisV )
        else:
            cost = np.zeros((thisDim[1], pt.shape[1]))
            #Necessary when coming from odeint
            if singleTime:
                p0, P, _ = self.getEllip(t)
                C = np.linalg.cholesky(P).T
            for k in range(pt.shape[1]):
                if not singleTime:
                    t = tVec[k]
                    p0, P, _ = self.getEllip(t)
                    C = np.linalg.cholesky(P).T
                cost[:,k] = colWiseSquaredNorm( np.dot( C, pt[:,k].reshape((thisDim[1], thisDim[0])).T ) )
        if pt.shape[1] == 1:
            cost = float(cost[0])
        return cost
        
    ###########################
    
    def getInitPoints(self, t0=None, N=11, ax=[0,1]):
        if t0 is None:
            t0=self.tSteps[0]
        x0,P,_ = self.getEllip(t0)
        
        Psub = P[np.ix_(ax, ax)]
        
        init2d = getV(Psub, N)
        
        init = np.tile(x0, (1,N))
        init[ax[0],:]=init2d[0,:]
        init[ax[1],:]=init2d[1,:]
        
        init += x0
        
        return init
        
    ###########################   
    
    def getLargestFunnel(self, inputQueue=None, outputQueue=None, subSteps = 1):
        #addInputs: [numVars, degrees2ind, AcstrBase, BcstrBase, indMatrixAll, indMatrixZtilde, indMatrixXtoz, nx, firstCond, secondCond]
        try:
            with open("status.txt", "wt") as sFile:
                sFile.write( "{0}\n".format( time.strftime("%H:%M:%S") ) )
                sFile.write("Solution alpha to step {0} is \n {1} \n {2}\n {3}\n".format(len(self.tSteps)-1, self.allShapes[-1][0], self.allShapes[-1][1], np.linalg.det(self.allShapes[-1][0]*self.allShapes[-1][1])))
        except:
            with open("status.txt", "wt") as sFile:
                sFile.write( "{0}\n".format( time.strftime("%H:%M:%S") ) )
                sFile.write("Solution alpha to step {0} is \n {1} \n {2}\n {3}\n".format(len(self.tSteps)-1, self.allShapes[-1][0][0][0], self.allShapes[-1][1], np.linalg.det(self.allShapes[-1][0][0][0]*self.allShapes[-1][1])))            
        for k in range(len(self.tSteps)-2,-1,-1):
            print("Computing size of segment {0}".format(k))
            self.getLargestRegionN(k, inputQueue=inputQueue, outputQueue=outputQueue, subSteps=subSteps)
            print("Solution alpha to step {0} is {1}\n".format(k, self.allShapes[k]))
            try:
                with open("status.txt", "at") as sFile:
                    sFile.write("Solution alpha to step {0} is \n {1} \n {2}\n {3}\n".format(k, self.allShapes[k][0], self.allShapes[k][1], np.linalg.det(self.allShapes[k][0]*self.allShapes[k][1])))
                    sFile.write( "{0}\n".format( time.strftime("%H:%M:%S") ) )
            except:
                with open("status.txt", "at") as sFile:
                    sFile.write("Solution alpha to step {0} is \n {1} \n {2}\n {3}\n".format(k, self.allShapes[k][0][0][0], self.allShapes[k][1], np.linalg.det(self.allShapes[k][0][0][0]*self.allShapes[k][1])))
                    sFile.write( "{0}\n".format( time.strftime("%H:%M:%S") ) )
        return 0
    
    ###########################
    
    def getLargestRegionN(self, n, inputQueue=None, outputQueue=None, subSteps = 1):
        
        
        #Check if the system converges for the shapes n, n+1
        #Get the shapes
        tTest = np.linspace(self.tSteps[n], self.tSteps[n+1], subSteps+1, endpoint=False)
        if self.allShapes[n] is None:
            self.allShapes[n] = self.shapeGen(self.refTraj(tTest[0])[0], self.tSteps[[n,n+1]], self.refTraj, lastShape=self.allShapes[n+1])
        
        dT = self.tSteps[n+1] - self.tSteps[n]
        Pn = self.allShapes[n][0]
        alphan = self.allShapes[n][1]
        Pn1 = self.allShapes[n+1][0]
        alphan1 = self.allShapes[n+1][1]
        #To be removed
        #if self.ellipInterpMode == 1:
        #    #Get the cholesky fac
        #    Pn = np.linalg.cholesky(Pn)
        #    Pn1 = np.linalg.cholesky(Pn1)
        
        #Check initial value
        if self.doSweep is None:
            minConv = self.getThisMinConv(tTest, dT, Pn, alphan, Pn1, alphan1, inputQueue, outputQueue)
            printDBG(minConv)
            NNN=1
            if minConv <= numericEps:
                #If currently converging enlarge zone by making alpha smaller 
                while minConv <= numericEps and alphan > 1e-2:
                    NNN+=1
                    alphaOld = alphan
                    alphan = alphan/self.alphaSearchFact
                    printDBG("{0} : {1}".format(alphaOld, alphan))
                    minConv = self.getThisMinConv(tTest, dT, Pn, alphan, Pn1, alphan1, inputQueue, outputQueue)
                
                alphaSafe = alphaOld
                alphaFail = alphan
                
            
            else:
                #If not converging, reduce the size of the zone by augmenting alpha
                while minConv > numericEps:
                    NNN+=1
                    alphaOld = alphan
                    alphan = alphan*self.alphaSearchFact
                    assert alphan < 1e6, "Zone too small"
                    printDBG("{0} : {1}".format(alphan, alphaOld))
                    minConv = self.getThisMinConv(tTest, dT, Pn, alphan, Pn1, alphan1, inputQueue, outputQueue)
                
                alphaSafe = alphan
                alphaFail = alphaOld
        else:
            NNN=0
            while True:
                minConv = np.ones((len(self.doSweep,)))
                for kSweep,aFact in enumerate(self.doSweep):
                    minConv[kSweep] = self.getThisMinConv(tTest, dT, Pn, alphan*aFact, Pn1, alphan1, inputQueue, outputQueue)
                    NNN+=1
                    if minConv[kSweep]<=numericEps:
                        break
                ind = np.argmax(minConv <= numericEps)
                if (minConv[ind]<= numericEps):
                    #Actually found a solution
                    if ind == 0:
                        #The largest value was valid -> Redo sweep with lower value for alpha
                        alphan = alphan*self.doSweep[0]    
                    else:
                        alphaFail = alphan*self.doSweep[ind-1]
                        alphaSafe = alphan*self.doSweep[ind]
                        break
                else:
                    #No valid solution was found continue with smaller ellipsoids
                    alphan = alphan*self.doSweep[-1]
            
                
                
            
        #Either perform bisection or just take the last valid value
        #self.allShapes[n][1] = alphaSafe
        alphaU = alphaSafe
        alphaL = alphaFail
        while (2.*abs(alphaU-alphaL))/(alphaU+alphaL) > self.convergenceLim:
            NNN+=1
            alphan = (alphaU+alphaL)/2.
            assert alphan < 1e10, "Zone too small"
            minConv = self.getThisMinConv(tTest, dT, Pn, alphan, Pn1, alphan1, inputQueue, outputQueue)
            if minConv <= numericEps:
                #Converges!
                alphaU = alphan
            else:
                #Diverges!
                alphaL = alphan
                
            
        #Think about how to change shape
        self.allShapes[n][1] = alphaU
        print('It took '+str(NNN)+' iterations')
        return 0
    
    ###########################################################################
    
    def getThisMinConv(self, tTest, dT, Pn, alphan, Pn1, alphan1, inputQueue, outputQueue):
        #Calculate P(t) an d/dt P
        #dP = 1./dT*(alphan1*Pn1-alphan*Pn)

        shapeTest = []
        resultList = []
        for aT in tTest:
            #Lyapunov region for this time point
            #to be removed
            #Pt = Pn*alphan*(1.-(aT-tTest[0])/dT)+Pn1*alphan1*(aT-tTest[0])/dT
            #Calculate P(t) an d/dt P
            #if self.ellipInterpMode == 0:
            #    Pt, dP = standardInterpol(Pn, alphan, Pn1, alphan1, aT, tTest[0], tTest[-1])
            #elif self.ellipInterpMode == 1:
            #    Pt, dP = cholInterpol(Pn, alphan, Pn1, alphan1, aT, tTest[0], tTest[-1])
            #else:
            #    assert 0, "Not implemented"
            Pt, dP = self.ellipInterp(Pn, alphan, Pn1, alphan1, aT, tTest[0], tTest[0]+dT)
            Ct = getP2(Pt)
            Cti = np.linalg.inv(Ct)#getPm2(Pt)
            x0T, xd0T, _ = self.refTraj(aT) #Get x0T for every permutation
            #Get all combinations of signs
            for aPerm in self.allInputPermutations:
                #boundCalcType, Pt, Ct, Cti, dP, aT, x0T, xd0T, perm, nu, nX, ustar, f, B, polyA, degreeList1, degreeList2, yDeg, zyDeg, transVarsCoefs, transVarsDegrees, polyNumbers, numVars, addInPutsRelax
                #Get some stuff
                aPerm = np.array(aPerm).astype(np.int_)
                aPerm.resize((aPerm.size,))
                aPerm[aPerm==0]=-1
                aPerm.resize((self.dynSys.nu,))
                ustar = self.dynSys.inputCstr.getThisU(aPerm)
                f,B,polyA,polyB = self.dynSys.getTaylorS3Fast(x0T)
                #Test
                #polyB = None#Linear input
                #polyA[:,self.dynSys.nx:] = 0.#Linear dynamics
                #zyDeg = self.dynSys.polyDegrees
                #yDeg = self.dynSys.polyDegrees[:self.dynSys.polyNumbers[0]]
                #Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, whichPart
                #thisT, Pt, Ct, Cti, dP, x0T, xd0T, perm, f, B, polyA, polyB, whichPart, epsDeadZone, innerAlpha, excludeInner
                shapeTest.append( dp([aT, Pt, Ct, Cti, dP, x0T, xd0T, aPerm, f, B, polyA, polyB, len(shapeTest), self.epsDeadZone, max(2.0*(0 if self.epsDeadZone is None else self.epsDeadZone), (1+(0 if self.excludeInner is None else self.excludeInner) )/2.), self.excludeInner]) )
                if glbDBG:
                    resultList.append(self.calculateFunction(shapeTest[-1]))
                    
        #Solve it
        if not(glbDBG) :
            for k in range(len(shapeTest)):
                inputQueue.put(shapeTest[k])
            resultList = [None for _ in range(len(shapeTest))]
            for k in range(len(shapeTest)):
                resultList[k] = outputQueue.get()
        #else:
        #    resultList = map( self.calculateFunction, shapeTest )
        print( resultList )
        return max( map(lambda aRes: float(aRes[0]), resultList) )

###########################


class NCrelax():
    def __init__(self, nx, mode='PL', excludeInner = None):
        self.nx = nx
        self.nX = (nx*(nx+1))//2
        self.mode = mode
        self.type = 'NCrelax'
        
        self.excludeInner = excludeInner
        
        if mode == 'PL':
            self.deg2ind, self.ind2deg, self.iMx, self.iMX, self.iMZt, self.iMzZ, self.iMA, self.iMX2z, self.dTraceX, self.dTraceZtilde, self.firstCond, self.secondCond, self.thirdCond, self.powerxz, self.powerZtilde = getVarsForPolyRelax(self.nx)
            #Directly modify the type
            if len(np.where(np.abs(self.firstCond[0].reshape((-1,)))>1e-10)[0])<self.firstCond[0].size*0.2:
                self.firstCond[0] = sparse(matrix(self.firstCond[0]))
            else:
                self.firstCond[0] = matrix(self.firstCond[0])
            if len(np.where(np.abs(self.secondCond[0].reshape((-1,)))>1e-10)[0])<self.secondCond[0].size*0.2:
                self.secondCond[0] = sparse(matrix(self.secondCond[0]))
            else:
                self.secondCond[0] = matrix(self.secondCond[0])
            if len(np.where(np.abs(self.thirdCond[0].reshape((-1,)))>1e-10)[0])<self.thirdCond[0].size*0.2:
                self.thirdCond[0] = sparse(matrix(self.thirdCond[0]))
            else:
                self.thirdCond[0] = matrix(self.thirdCond[0])
            self.firstCond[1] = matrix( self.firstCond[1].reshape(1+np.array(self.iMX.shape)) )
            self.secondCond[1] = matrix( self.secondCond[1].reshape(self.iMA.shape) )
            self.thirdCond[1] = matrix( self.thirdCond[1].reshape(self.iMzZ.shape) )
                        
            self.numVars = self.firstCond[0].size[1]
            #Store the constraints that will not change
            #Impose unit hypersphere (Condition  tr(X)<=1 <-> \sum_i x_ii**2 <= 1)
            self.AXU = np.array(self.dTraceX).reshape((1,-1))
            self.BXU = np.ones((1,1))
            #Problem: zero valued convergence at origin -> exclude inner
            #tr(X)>=.25 <-> -tr(x)<=-0.25
            if not ( self.excludeInner is None ):
                self.AXUi = -np.array(self.dTraceX).reshape((1,-1))
                self.BXUi = -self.excludeInner*np.ones((1,1))
            else:
                self.AXUi = np.zeros((0,self.AXU.shape[1]))
                self.BXUi = np.zeros((0,1))

        else:
            assert 0, 'not implemented'
        
            
        return None
    
    def getCstr(self, boundCalculator, RLTCalculator):
        #All the constraints on each variable
        AlimUpLowx, BlimUpLowx, AlimUpLowX, BlimUpLowX, AlimUpLowxz, BlimUpLowxz, AlimUpLowZ, BlimUpLowZ = boundCalculator( mode = 'SDR' )
        #Relations between the variables
        #impose that the quadratic term is smaller than the original
        AZijijXij = np.zeros(( self.nx, self.numVars ))
        BZijijXij = np.zeros(( self.nx, 1 ))
        for k in range(self.nx):
            #Z_ij^ij - X_ij <= 0 
            AZijijXij[k, self.iMX[k,k]] = -1.
            kz = self.iMX2z[k,k]
            AZijijXij[k, self.iMZt[self.nx+kz, self.nx+kz]] = 1.
        
        self.AcstrBase = np.vstack((AlimUpLowx, AlimUpLowX, self.AXU, self.AXUi, AlimUpLowxz, AlimUpLowZ, AZijijXij))
        self.BcstrBase = np.vstack((BlimUpLowx, BlimUpLowX, self.BXU, self.BXUi, BlimUpLowxz, BlimUpLowZ, BZijijXij))
        
        self.AcstrBasex = AlimUpLowx
        self.BcstrBasex = BlimUpLowx
        
        self.AcstrBaseX = np.vstack((AlimUpLowX, self.AXU, self.AXUi))
        self.BcstrBaseX = np.vstack((BlimUpLowX, self.BXU, self.BXUi))
        
        self.AcstrBasexz = AlimUpLowxz
        self.BcstrBasexz = BlimUpLowxz
        
        #self.AcstrBaseZ = np.vstack((AlimUpLowx, AlimUpLowX, self.AXU, self.AXUi, AlimUpLowxz, AlimUpLowZ, AZijijXij))
        #self.BcstrBaseZ = np.vstack((BlimUpLowx, BlimUpLowX, self.BXU, self.BXUi, BlimUpLowxz, BlimUpLowZ, BZijijXij))
        self.AcstrBaseZ = np.vstack((AlimUpLowZ, AZijijXij))
        self.BcstrBaseZ = np.vstack((BlimUpLowZ, BZijijXij))
        
        
        #Precalculate enlarged base cstr
        AX, Axz, AZ, BX, Bxz, BZ = RLTCalculator(self.AcstrBasex, self.BcstrBasex, AXin=self.AcstrBaseX, BXin=self.BcstrBaseX, Axzin=self.AcstrBasexz, Bxzin=self.BcstrBasexz)
        
        self.AtildeCstrBasex = AlimUpLowx
        self.BtildeCstrBasex = BlimUpLowx
        
        self.AtildeCstrBaseX = np.vstack([self.AcstrBaseX]+AX)
        self.BtildeCstrBaseX = np.vstack([self.BcstrBaseX]+BX)
        
        self.AtildeCstrBasexz = np.vstack([self.AcstrBasexz]+Axz)
        self.BtildeCstrBasexz = np.vstack([self.BcstrBasexz]+Bxz)
        
        self.AtildeCstrBaseZ = np.vstack([self.AcstrBaseZ]+AZ)
        self.BtildeCstrBaseZ = np.vstack([self.BcstrBaseZ]+BZ)
        
        self.AtildeCstrBase = np.vstack([self.AtildeCstrBasex,self.AtildeCstrBaseX,self.AtildeCstrBasexz,self.AtildeCstrBaseZ])
        self.BtildeCstrBase = np.vstack([self.BtildeCstrBasex,self.BtildeCstrBaseX,self.BtildeCstrBasexz,self.BtildeCstrBaseZ])
        
        self.AtildeCstrBaseSM = sparse(matrix(self.AtildeCstrBase))
        
        return 0

   
        
    
    
    
    
    
                               

    
    
