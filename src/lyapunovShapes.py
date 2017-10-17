from coreUtils import *
import cvxUtils
from numpy.linalg import LinAlgError
import plotUtils as pu
expm = scipy.linalg.expm
chol = np.linalg.cholesky

###########################
class simpleGen():
    def __init__(self):
        pass
    
    def __call__(self, x0, *args, **kwargs):
        if 'lastShape' in kwargs.keys():
            thisShape, thisSize = dp(kwargs['lastShape'])
        else:
            thisShape, thisSize = np.eye(x0.size), 1.
        return [thisShape, thisSize]
###########################
#class pieceWiseCst():
#    def __init__(self):
###########################
class lqrShapeGen():
    def __init__(self, dynSys, Q=None, R=None, N=None, reshape = True ):
        self.dynSys = dynSys
        self.reshape = reshape
        if Q is None:
            self.Q = np.identity(self.dynSys.nx)
        else:
            self.Q = Q
        if R is None:
            self.R = 0.01*np.identity(self.dynSys.nu)
        else:
            self.R = R
        if N is None:
            self.N = np.zeros((self.dynSys.nx, self.dynSys.nu))
        else:
            self.N = N
    
    def __call__(self, x0, *args, **kwargs):
        #_unused,B0,A0,_unused = self.dynSys.getLinFast(x0)
        A0 = self.dynSys.getLinAFast(x0)
        B0 = self.dynSys.gEval(x0)
        B0.resize((self.dynSys.nx, self.dynSys.nu))
        self.lastK, S, _unused = crtl.lqr(A0, B0, self.Q, self.R, self.N)
        if self.reshape:
            v,w = np.linalg.eigh(S)
            S = ndot(w.T, np.diag(np.sqrt(v)), w)
            #S = ndot(w.T, np.diag(np.power(v, 1./6.)), w)
            #S = np.eye(S.shape[0])
            
        if 'lastShape' in kwargs.keys():
            thisSize = dp(kwargs['lastShape'][1])
        else:
            thisSize = 1.
        return [S/((np.linalg.det(S))**(1./x0.size)), thisSize]
    
###########################
class linearBackProp():
    def __init__(self, dynSys, interSteps= 10, dT = 2e-4, ctrlSafeFac=0.9, convMin=0.001, Q=None):
        self.dynSys = dynSys
        self.interSteps = 10
        self.dT = 2e-4
        self.ctrlSafeFac = ctrlSafeFac
        self.expFac = 1.01
        self.convMin = convMin
        self.invCond = None
        if Q is None:
            self.Q = np.zeros((self.dynSys.nx, self.dynSys.nx))
        else:
            self.Q = Q
    
    def backThisT(self, refTraj, P, t, dT):
        
        Pinit = np.copy(P)
        
        P = np.copy(P)+self.Q*dT
        if not(self.invCond is None) and False:
            e,v = np.linalg.eigh(P)
            #e = scipy.interpolate.spline(np.array([e[0],e[-1]]), np.array([e[-1]/self.invCond,e[-1]]),e,1)
            e = np.maximum(e, e[-1]/self.invCond)
            P = ndot(v, np.diag(e), v.T)
        #Do back prop
        xt = refTraj.xref(t)
        Ac = self.dynSys.getLinAFast(xt)
        Bc = self.dynSys.gEval(xt)
        #Get the best linear feedback with largest coeffs allowed
        #The optimal linear control is B.T.P
        #The largest input generated on omega is B.T.P.x=B.T.C.T.C.Ci.u=B.T.C.T.u
        while True:
            #New chol
            Cp = chol(P).T
            Kstar = ndot(Bc.T, P)
            Kscale = ndot(Bc.T,Cp.T)
            KscaleNorm = colWiseNorm(Kscale.T).reshape((self.dynSys.nu,1))
            uRef = refTraj.uref(t).reshape((self.dynSys.nu,1))
            uLim = np.minimum( np.abs(self.dynSys.inputCstr.getMinU(t)-uRef), np.abs(self.dynSys.inputCstr.getMaxU(t)-uRef) )
            Kstar = np.multiply(np.divide(Kstar,KscaleNorm), uLim)*self.ctrlSafeFac
            
            if 1:
                #test stuff
                AtildeC0=Ac
                AtildeCU = Ac-ndot(Bc,Kstar) #xd=Atilde.x
                AtildeD0 = expm(AtildeC0*dT)
                AtildeDU = expm(AtildeCU*dT)
                Pnew0 =ndot(AtildeD0.T, P, AtildeD0)
                PnewU =ndot(AtildeDU.T, P, AtildeDU)
                
                
            
            #This corresponds to forward integration
            AtildeC = Ac-ndot(Bc,Kstar) #xd=Atilde.x
            
            #But we want backward integration
            AtildeD = expm(AtildeC*dT)
            Pnew = ndot(AtildeD.T, P, AtildeD)
            Pdot = (Pinit-Pnew)/dT
            dP=ndot(P, AtildeC)
            dP=dP+dP.T+Pdot+self.convMin*P
            
            try:
                chol(-dP)#Ensuring that minus dP is positive
                break
            except LinAlgError:
                P = P*self.expFac
            
            #if np.max(np.linalg.eigh(dP)[0])<0:
            #    break
            #else:
            #    P = P*self.expFac
        if not(self.invCond is None) and True:
            e,v = np.linalg.eigh(Pnew)
            #e = scipy.interpolate.spline(np.array([e[0],e[-1]]), np.array([e[-1]/self.invCond,e[-1]]),e,1)
            e = np.maximum(e, e[-1]/self.invCond)
            Pnew = ndot(v, np.diag(e), v.T)
        
        return Pnew
        
    
    def __call__(self, x0, tSpan, refTraj, lastShape):
        Pold = lastShape[0][0][0]*lastShape[1]
        P = np.copy(Pold)
        
        tSteps = np.linspace(tSpan[0], tSpan[1], self.interSteps)
        allP = [None for _ in range(self.interSteps)]
        k=self.interSteps-1
        for t0,t1 in zip( np.flip(tSteps[:-1],0), np.flip(tSteps[1:],0) ):
            t=t1
            allP[k]=np.copy(P)
            k-=1
            while t-self.dT>t0:
                P = self.backThisT(refTraj, P, t, self.dT)
                t-=self.dT
            thisdT = t-t0
            P = self.backThisT(refTraj, P, t, thisdT)
        assert k==0
        allP[k]=np.copy(P)
        
        if 1:
            fff,aaa=plt.subplots(1,1)
            jj=pu.cmx.jet(np.linspace(0,1,len(allP)))
            for k,aP in enumerate(allP):
                print(np.linalg.eigvalsh(aP))
                pu.plotEllipse(aaa, np.zeros((2,1)), aP, 1., color=jj[k,:], faceAlpha=0.1 )
            aaa.autoscale(1)
            plt.show()
        
        return [ [allP, tSteps] , 1. ]

###########################
class timeVaryingLqrShapegen():
    def __init__(self, dynSys, Q=None, R=None, divCoeff=10., reshape=False, restart=False):
        self.dynSys = dynSys
        if Q is None:
            self.Q = np.identity(self.dynSys.nx)
        else:
            self.Q = Q
        if R is None:
            self.R = 0.01*np.identity(self.dynSys.nu)
        else:
            self.R = R
        self.divCoeff=divCoeff
        self.lastK=None
        self.reshape = reshape
        self.restart = restart
        self.interSteps = None
        self.allZ = None
        self.allT = None
        
        self.limK=True
        self.refTraj=None
        self.ctrlSafeFac=0.9
        self.retAll = False
    
    def dz(self, z, t, xx):
        #Current P matrix
        P=z.reshape((self.dynSys.nx,self.dynSys.nx))
        #Current linearised system
        A=self.dynSys.getLinAFast(xx)
        B=self.dynSys.gEval(xx)
        
        K = np.linalg.lstsq(self.R, ndot(B.T,P))[0]
        
        if self.limK:
            Cpi = np.linalg.inv(chol(P).T)
            Kstar = K
            Kscale = ndot(K, Cpi)
            KscaleNorm = colWiseNorm(Kscale.T).reshape((self.dynSys.nu,1))
            uRef = self.refTraj.uref(t).reshape((self.dynSys.nu,1))
            uLim = np.minimum( np.abs(self.dynSys.inputCstr.getMinU(t)-uRef), np.abs(self.dynSys.inputCstr.getMaxU(t)-uRef) )
            Kstar = np.multiply(np.divide(Kstar,KscaleNorm), uLim)*self.ctrlSafeFac
            K=Kstar
        
        if 1:
            dP = -(ndot(A.T,P)+ndot(P,A) - 2.*ndot(P,B,K) + self.Q)
        else:
            #Project Q into the principal axis of dP and check if Q not too large
            dP = -(ndot(A.T,P)+ndot(P,A) - 2.*ndot(P,B,K))
            e,v = np.linalg.eigh(dP)
            Qe = np.diag(ndot(v.T, self.Q, v))
            Qe = np.minimum( np.abs(e)*.5, Qe )
            dP = dP - np.diag(ndot(v, np.diag(Qe), v.T))
        
        dP = (dP+dP.T)/2.        
        return dP.reshape((self.dynSys.nx**2))
    def getDz(self, refTraj, allT, allZ):
        allXX = refTraj.xref(allT)
        allDz = np.zeros_like(allZ)
        for k in range(allZ.shape[1]):
            allDz[:,k] = self.dz(allZ[:,k], allT[k], allXX[:,[k]])
        return allDz
                                
    #self.allShapes[n] = self.shapeGen(self.refTraj(tTest[0])[0], self.tSteps[[n,n+1]], self.refTraj, lastShape=self.allShapes[n+1])
    def __call__(self, x0, tSpan, refTraj, lastShape):
        nx = self.dynSys.nx
        nu = self.dynSys.nu
        assert not (self.reshape and self.restart)
        
        if self.retAll:
            lastShape = [lastShape[0][0][0], lastShape[1]]
            
        
        if self.reshape:
            M = lastShape[0] 
            [w,v] = np.linalg.eigh(M)
            M = ndot(v.T, np.diag(np.square(w)), v)
            z0=M.reshape((nx**2,))
        elif self.restart:
            z0=(lastShape[0]*lastShape[1]).reshape((nx**2,))
        else:
            z0=lastShape[0].reshape((nx**2,))
        if self.interSteps is None:
            z = scipy.integrate.odeint(lambda thisZ, thisT: self.dz(thisZ, thisT, refTraj.xref(thisT)), z0, [tSpan[1], tSpan[0]])[-1,:]
        else:
            allT = np.linspace(tSpan[1], tSpan[0], self.interSteps)
            allZ = scipy.integrate.odeint(lambda thisZ, thisT: self.dz(thisZ, thisT, refTraj.xref(thisT)), z0, allT)
            
            z=allZ[-1,:]
            allZ = allZ.T
            #Flip to the time in order
            allT = np.flip(allT,0)
            allZ = np.fliplr(allZ)
            if not(self.allZ is None):
                self.allZ = np.hstack((allZ, self.allZ))
                self.allT = np.hstack((allT, self.allT))
            else:
                self.allZ = allZ
                self.allT = allT
        
        P = z.reshape((nx,nx))
        P = (P+P.T)/2
        B=self.dynSys.gEval(refTraj.xref(tSpan[0]))
        self.lastK = np.linalg.lstsq(self.R, ndot(B.T,P))
        try:
            assert np.all(np.isclose(P-P.T,0.)), "non symmetric"
            np.linalg.cholesky(P)
        except:
            print(P)
            print(np.linalg.eigh(P)[0])
            print(np.linalg.eigh(P)[1])
            print(P-P.T)
            assert 0
            
        if self.restart:
            retVal = 1.
        else:
            retVal = lastShape[1]
        if self.retAll:
            allDz = self.getDz(refTraj, allT, allZ)
            allZM =[ allZ[:,k].reshape((self.dynSys.nx,self.dynSys.nx)) for k in range(allZ.shape[1])]
            allDzM =[ allDz[:,k].reshape((self.dynSys.nx,self.dynSys.nx)) for k in range(allZ.shape[1])]
            return [[allZM, allDzM, allT], retVal]
        else:
            return [P, retVal]

###########################

#class KShape():
#    def __init__(self, dynSys, Kin = None, gamma=0.05, condNbr = 10):
#        self.dynSys = dynSys
#        self.gamma = gamma
#        self.condNbr = condNbr
#        if Kin is None:
#            self.K = np.zeros((self.dynSys.nu, self.dynSys.nx))
#        else:
#            self.K = Kin
#    
#    def __call__(self, x0, *args, **kwargs):




###########################
class lqrShapeGen2():
    def __init__(self, dynSys, Q=None, R=None, N=None, reshape = True ):
        self.dynSys = dynSys
        self.reshape = reshape
        if Q is None:
            self.Q = np.identity(self.dynSys.nx)
        if R is None:
            self.R = 0.01*np.identity(self.dynSys.nu)
        if N is None:
            self.N = np.zeros((self.dynSys.nx, self.dynSys.nu))
        nx = self.dynSys.nx
        indS = np.zeros((nx,nx))
        ind=0
        for k in range(nx):
            for i in range(k,nx):
                indS[k,i] = ind[i,k] = ind
                ind +=1
        indS = indS.astype(np.int_)
        self.numVars = ind+1   
        
    
    def __call__(self, x0, *args, **kwargs):
        #_unused,B0,A0,_unused = self.dynSys.getLinFast(x0)
        A0 = self.dynSys.getLinAFast(x0)
        B0 = self.dynSys.gEval(x0)
        B0.resize((self.dynSys.nx, self.dynSys.nu))
        self.lastK, S, _unused = crtl.lqr(A0, B0, self.Q, self.R, self.N)
        if self.reshape:
            self.lastK = np.sqrt(self.lastK)
            #S = np.eye(S.shape[0])
        
        #Get the optimized
        
        
        if 'lastShape' in kwargs.keys():
            thisSize = dp(kwargs['lastShape'][1])
        else:
            thisSize = 1.
        return [S/((np.linalg.det(S))**(1./x0.size)), thisSize]


###########################
class reverseIntegrationMethod():
    def __init__(self, dynSys, mode="OR", redo = 1, ellipInterpol = standardInterpol ):
        self.dynSys = dynSys
        
        self.mode = mode
        self.redo = 2
        
        self.NinitAdd = 4
        self.interSteps = 3#TBD
        
        self.ellipInterpol = ellipInterpol
        
        self.dim = self.dynSys.nx
        oldComb = [[]]
        for k in range(self.dim):
            thisComb = []
            for aC in oldComb:
                thisComb.append( dp(aC)+[1] )
                thisComb.append( dp(aC)+[-1] )
            oldComb = thisComb
        
        self.signComb = lmap(lambda a: np.array(a).reshape((self.dim,1)), oldComb)
        
        self.maxChange = [ -0.2, 0.2 ]
        
    
    def __call__(self, x0, *args, **kwargs):
        #_unused,B0,A0,_unused = self.dynSys.getLinFast(x0)
        try:
            Pn1 = kwargs['lastShape'][0]
            alphan1 = kwargs['lastShape'][1]
        except KeyError:
            Pn1 = np.identity(self.dim)
            alphan1 = 1.
        
        Pn = Pn1
        alphan = alphan1
        
        [t0, t1] = args[0]
        refTraj = args[1]
        x0 = refTraj.xref(t1)
        x0.resize((self.dim,1))
        if 0:
            e,v = np.linalg.eigh(Pn1)
            e = np.sqrt(e)
            
            allPoints = np.hstack(( np.multiply(v, np.tile(e,(self.dim,1))), -np.multiply(v, np.tile(e,(self.dim,1))) ))
            for aComb in self.signComb:
                allPoints = np.hstack(( allPoints,  np.sum(np.multiply(np.tile(np.multiply(aComb, e),(self.dim,1)), v),1).reshape((self.dim,1)) ))
                for k in range(self.NinitAdd):
                    thisComb = np.random.rand(self.dim,)
                    thisComb = thisComb/np.linalg.norm(thisComb)
                    allPoints = np.hstack(( allPoints, sum(nmult( np.tile(thisComb,(self.dim,1)), np.tile(e,(self.dim,1)), v ),1).reshape((self.dim,1)) ))
            Ninit = allPoints.shape[1]
        else:
            #Random
            allPoints =  np.hstack( lmap( lambda aPerm: np.multiply(np.random.rand(self.dim, 20), np.tile(aPerm,(1,20))), self.signComb ) )
            allPointsN = colWiseNorm(allPoints)
            allPoints = np.divide(allPoints, np.tile(allPointsN,(self.dim,1)))
            Cni1 = np.linalg.inv(np.linalg.cholesky(Pn1*alphan1).T)
            allPoints = ndot(Cni1,allPoints)
            allPoints = allPoints + x0
            Ninit = allPoints.shape[1]
            
        #I should get a smart heuristic about how to choose points
        for _ in range(self.redo):
            #Use the current approximation and backward integrate
            #Get the integration function Attention t is negative
            fInt = lambda x, t: self.dynSys.getBestF(self.ellipInterpol(Pn, alphan, Pn1, alphan1, t1+t, t0, t1)[0], refTraj.xref(t1+t), x.reshape((Ninit,self.dim)).T, mode=self.mode, u0=refTraj.uref(t1+t), t=t1+t).T.reshape((self.dim*Ninit,))
            newX = scipy.integrate.odeint(fInt, allPoints.T.reshape((self.dim*Ninit,)), [0,t0-t1])
            dXfinal = newX[-1,:].reshape((Ninit, self.dim)).T-refTraj.xref(t0)
            #Get target ellipsoid (target here but in fact its the source ellipsoid considering regular dynamics)
            Pn, alphan = cvxUtils.getLargestInnerEllip(dXfinal, maxChange = self.maxChange, Pold=Pn1*alphan1)
        
        return [Pn, alphan]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    