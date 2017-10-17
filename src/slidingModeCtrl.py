from coreUtils import *
import dynamicSystems as ds


#Sliding mode control for acrobot like in
#Design of a stable sliding-mode controller for a class of second-order underactuated systems

class slidingAcro:
    def __init__(self, dynSys, alphaI=None, betaI=None, c1I=None, c2I=None, eta=0., K=0., PI=None, secondOrderTau=None):
        
        self.dynSys = dynSys
        self.secondOrderTau = secondOrderTau
        
        self.alphaS = a = sy.symbols("alpha")
        self.betaS = b = sy.symbols("beta")
        self.c1S = c1 = sy.symbols("c1")
        self.c2S = c2 = sy.symbols("c2")
        
        self.PS = sMz(4,4)
        self.PS[0,0] = a**2*c1**2
        self.PS[0,1] = self.PS[1,0] = a**2*c1
        self.PS[0,2] = self.PS[2,0] = a*c1*b*c2
        self.PS[0,3] = self.PS[3,0] = a*c1*b
        self.PS[1,1] = a**2
        self.PS[1,2] = self.PS[2,1] = a*b*c2
        self.PS[1,3] = self.PS[3,1] = a*b
        self.PS[2,2] = b**2*c2**2
        self.PS[2,3] = self.PS[3,2] = b**2*c2
        self.PS[3,3] = b**2
        
        self.PM =[[] for _ in range(4)]
        
        if not (PI is None):
            self.Pi = 2.*PI
            
            self.alphaFromP = np.sqrt(self.Pi[1,1])
            self.betaFromP = np.sqrt(self.Pi[3,3])
            self.c1FromP = (self.Pi[0,0]/self.Pi[0,1] + np.abs(self.Pi[0,3]/(self.alphaFromP*self.betaFromP)))/2. 
            self.c2FromP = (self.Pi[2,2]/self.Pi[2,3] + np.abs(self.Pi[1,2]/(self.alphaFromP*self.betaFromP)))/2.
            
            if not np.isclose(self.Pi[1,3], 0., rtol=1e-4, atol=1e-4):
                self.signFromP = np.sign(self.Pi[1,3])
            else:
                self.signFromP = np.sign(self.Pi[3,0]/self.c1FromP)
            
            assert(self.c1FromP > 0 and self.c2FromP > 0)
            
            
        self.alpha = alphaI
        self.beta = betaI
        self.c1 = c1I
        self.c2 = c2I
        
        self.InorP = 0 #0: Use given alpha/beta ... 1: Use pars derived from P
        
        self.eta = eta
        self.K = K
        
    def getMat(self):
        return np.array( self.PS.subs( [[self.alphaS, self.alpha], [self.betaS, self.beta], [self.c1S, self.c1], [self.c2S, self.c2]] ) ).astype(np.float_)
    def getMatP(self, which=1):
        if self.signFromP > 0:
            return np.array( self.PS.subs( [[self.alphaS, self.alphaFromP], [self.betaS, self.betaFromP], [self.c1S, self.c1FromP], [self.c2S, self.c2FromP]] ) ).astype(np.float_)
        else:
            if which == 1:
                #make alpha pos
                return np.array( self.PS.subs( [[self.alphaS, self.alphaFromP], [self.betaS, -self.betaFromP], [self.c1S, self.c1FromP], [self.c2S, self.c2FromP]] ) ).astype(np.float_)
            else:
                #make beta pos
                return np.array( self.PS.subs( [[self.alphaS, -self.alphaFromP], [self.betaS, self.betaFromP], [self.c1S, self.c1FromP], [self.c2S, self.c2FromP]] ) ).astype(np.float_)
                 
    
    def __call__(self, thisX, x0, which = 1):
        #which pars to use
        if self.InorP==0:
            a,b,c1,c2 = self.alpha, self.beta, self.c1, self.c2
        elif self.InorP==1:
            if self.signFromP > 0:
                a,b,c1,c2 = self.alphaFromP, self.betaFromP, self.c1FromP, self.c2FromP
            else:
                if which == 1:
                    #make alpha pos
                    a,b,c1,c2 = self.alphaFromP, -self.betaFromP, self.c1FromP, self.c2FromP
                else:
                    #make beta pos
                    a,b,c1,c2 = -self.alphaFromP, self.betaFromP, self.c1FromP, self.c2FromP
        
        x0 = x0.reshape((4,1))
        if self.secondOrderTau is None:
            x = thisX.reshape((4,1))
            dx = x-x0
            lastS = newS = a*(c1*dx[0]+dx[1])+b*(c2*dx[2]+dx[3])
        else:
            lastS = thisX[-1]
            x = thisX[:-1].reshape((4,1))
            dx = x-x0
            newS = a*(c1*dx[0]+dx[1])+b*(c2*dx[2]+dx[3])
            #print("{0} : {1}".format(lastS, newS))
            
        #Test 
        lastS = newS = a*(c1*dx[0]+dx[1])+b*(c2*dx[2]+dx[3])
        
        #Compute the control input
        #Get input mapping
        B = self.dynSys.gEval(x)
        #And the uncontrolled system dynamics
        F = self.dynSys.fEval(x)
        
        #Get equilibrium input for one and two
        ueq1 = -(F[2]+c1*dx[1])/(B[2,0])
        ueq2 = -(F[3]+c2*dx[3])/(B[3,0])
        
        #Get eta^* and K^*
        invAB = 1./(a*B[2,0] + b*B[3,0])
        etaStar = invAB*self.eta
        KStar = invAB*self.K
        
        #Get the total u
        if self.secondOrderTau is None:
            uTot = a*B[2,0]/invAB*ueq1 + b*B[3,0]/invAB*ueq2 - etaStar*np.sign(lastS) - KStar*lastS
        else:
            uTot = a*B[2,0]/invAB*ueq1 + b*B[3,0]/invAB*ueq2 - etaStar*float(np.maximum(np.minimum(lastS/self.secondOrderTau, 1),-1)) - KStar*lastS
        #uTot = -uTot
        #Get induced dynamics
        print(uTot)
        dX = self.dynSys(x, np.array([[uTot]])).squeeze()
        
        #Get evolutions of S if second order
        if not( self.secondOrderTau is None):
            #print(-(lastS-newS)/self.secondOrderTau)
            dX = np.hstack((dX, -(lastS-newS)/self.secondOrderTau))
        
        return dX
        
         
            