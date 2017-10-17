from coreUtils import *
import os
import subprocess


#Class implementing input constraints
#Box constraints limit the inputs to a hyperrectangle


class boxCstr():
    def __init__(self, nu, limL=None, limU=None):
        self.nu = nu
        self.limLCall = False
        self.limUCall = False
        if not hasattr(limL, "__call__"):
            limL = np.array(limL)
        if not hasattr(limU, "__call__"):
            limU = np.array(limU)
        if limL is None:
            self.limL = -np.ones((self.nu,1))
            self.thisLimL = self.limL
        elif isinstance(limL, np.ndarray):
            self.limL = np.array(limL).reshape((nu,1))
            self.thisLimL = self.limL
        else:
            try:
                self.limL = lambda t: limL(t).reshape((self.nu,1))
                self.limLCall = True
                self.thisLimL = self.limL(0.0)
            except:
                print("limL is neither an array nor is it callacle")
        
            
        if limU is None:
            self.limU = np.ones((self.nu,1))
            self.thisLimU = self.limU
        elif isinstance(limU, np.ndarray):
            self.limU = np.array(limU).reshape((nu,1))
            self.thisLimU = self.limU
        else:
            try:
                self.limU = lambda t: limU(t).reshape((self.nu,1))
                self.thisLimU = self.limU(0.0)
                self.limUCall = True
            except:
                print("limL is neither an array nor is it callacle")
        
    
    #######################
    def getMinU(self, t=0):
        #Get current lower lim
        if self.limLCall:
            self.thisLimL = self.limL(t)
        return self.thisLimL
    #######################
    def getMaxU(self, t=0):
        #get current upper lim
        if self.limUCall:
            self.thisLimU = self.limU(t)
        return self.thisLimU    
    #######################
    def getThisU(self, thisInd, t=0, uRef = None):
        #Get optimal input encoded by index vector thisInd
        #1 is maximum input
        #0 is minimal input
        optCtrl = np.zeros((self.nu,1))
        thisInd = thisInd.astype(np.int_).reshape((-1,))
        boolArr = thisInd ==-1
        if self.limLCall:
            self.thisLimL = self.limL(t)
        if self.limUCall:
            self.thisLimU = self.limU(t)
        
        if np.any(boolArr):
            optCtrl[boolArr] = self.thisLimL[boolArr]
        boolArr = thisInd == 1
        if np.any(boolArr):
            optCtrl[boolArr] = self.thisLimU[boolArr]
        if not (uRef is None):
            boolArr = thisInd == 0
            if np.any(boolArr):
                optCtrl[boolArr] = uRef[boolArr]
        
        return optCtrl
    
    #######################
    def getCstr(self, t=0):
        if self.limLCall:
            self.thisLimL = self.limL(t)
        if self.limUCall:
            self.thisLimU = self.limU(t)
        return self.thisLimL, self.thisLimU
    #######################
    def __call__(self, inputNonCstr, t=0):
        #Limit given input array
        t = np.array(t)
        if t.size==1:
            if self.limLCall:
                self.thisLimL = self.limL(t)
            if self.limUCall:
                self.thisLimU = self.limU(t)
            #inputNonCstr.resize((inputNonCstr.size,))
            inputNonCstr = np.maximum(self.thisLimL, np.minimum(self.thisLimU, inputNonCstr))
        elif t.size ==inputNonCstr.shape[1]:
            if (self.limLCall or self.limUCall):
                for k,aT in enumerate(t): 
                    if self.limLCall:
                        self.thisLimL = self.limL(aT)
                    if self.limUCall:
                        self.thisLimU = self.limU(aT)
                    inputNonCstr[:,k] = np.maximum(self.thisLimL, np.minimum(self.thisLimU, inputNonCstr[:,k]))
            else:
                inputNonCstr = np.maximum(self.thisLimL, np.minimum(self.thisLimU, inputNonCstr))
        
        return inputNonCstr
        
        
        
        
#Main class implementing approximation and evaluation of general nonlinear systems
#system and input dynamics are given stored as vectors/arrays of sympy expressions
class dynamicalSys():
    
    def __init__(self, f, g, K=None, varNames=None, inputNames=None, inputCstr=None, sysName=None, so2DimsIn=None):
        #d/dt = f(x)+g(x).u
        #d/dt = f(x)+g(x).(u_ref + K(x-x_ref)
        
        assert( isinstance(f, (list, tuple, sMa)))
        assert( isinstance(g, (list, tuple, sMa)))
        assert( isinstance(K, (list, tuple, sMa)) or (K is None))
        
        #Scaling the input for realistic control
        self.innerZone = 10e-5; #will be used as tau if second order
        self.secondOrderSlide = False#Experimental not properly tested yet
        
        self.nx = len(f)
        if isinstance(g, (list, tuple)):
            self.nu = len(g[0])
        elif isinstance(g, sMa):
            self.nu = g.shape[1]
        
        assert( (K is None) or (len(K) == self.nu) )
                
        #Parse varnames
        #If special names are given for the variables paste them, else use x for the state-space and 
        #u for the input
        if (varNames is None):
            #self.x = sMa(sympy.symbols('x0:{0:d}'.format(self.nx)))
            self.x = sMz(self.nx, 1)
            for k in range(self.nx):
                exec('''x{0:d}=sympy.symbols('x{0:d}')'''.format(k))
                self.x[k] = eval('x{0:d}'.format(k))
        elif isinstance(varNames, (list, tuple)):
            self.x = sMa(varNames)
            for k in range(self.nx):
                exec('''{0}=sympy.symbols('{0}')'''.format(str(self.x[k])))
        elif isinstance(varNames, sMa):
            self.x = varNames
        else:
            assert 0
        #Get reference names
        self.xref = sMz(self.nx,1)
        for k, aVar in enumerate(self.x):
            exec('''{0}_ref=sympy.symbols('{0}_ref')'''.format(str(aVar)))
            self.xref[k] = eval('{0}_ref'.format(str(aVar)))
        
        #parse inputNames
        if (inputNames is None):
            #self.u = sMa(sympy.symbols('u0:{0:d}'.format(self.nu)))
            self.u = sMz(self.nu, 1)
            for k in range(self.nu):
                exec('''u{0:d}=sympy.symbols('u{0:d}')'''.format(k))
                self.u[k] = eval('u{0:d}'.format(k), locals())
        elif isinstance(inputNames, (list, tuple)):
            self.u = sMa(inputNames)
            for k in range(self.nx):
                exec('''{0}=sympy.symbols('{0}')'''.format(str(self.u[k])), locals())
        elif isinstance(inputNames, sMa):
            self.u = inputNames
        else:
            assert 0
        
        #Get reference names
        self.uref = sMz(self.nu,1)
        for k, aVar in enumerate(self.u):
            exec('''{0}_ref=sympy.symbols('{0}_ref')'''.format(str(aVar)))
            self.uref[k] = eval('{0}_ref'.format(str(aVar)))
        
        #Create the sympy expressions for f if given as string
        if isinstance(f, (list, tuple)):
            self.f = sMz(self.nx,1)
            for k in range(self.nx):
                self.f[k] = eval(f[k], locals(), sympy.__dict__)
        elif isinstance(f, sMa):
            self.f = f
        else:
            assert 0, 'wrong type f'
        self.ft = sympy.lambdify(self.x, self.f, 'numpy')
        self.fEval = lambda xList : self.ft(*map(np.array, xList)).reshape((self.nx, -1))
        
        #Create the sympy expressions for g if given as string
        if isinstance(g, (list, tuple)):
            self.g = sMz(self.nx, self.nu)
            for k in range(self.nx):
                for l in range(self.nu):
                    self.g[k,l] = eval(g[k][l], locals(), sympy.__dict__)
        elif isinstance(g, sMa):
            self.g = g
        else:
            assert 0, 'wrong type g'    
        self.gt = sympy.lambdify(self.x, self.g, 'numpy')
        
        #Get the corrector
        #Paste the linear gain
        self.K = sMz(self.nu, 1)
        if isinstance(K, (list, tuple)):
            for k in range(self.nx):
                self.K[k] = eval(K[k], locals(), sympy.__dict__)
        elif isinstance(K, (list, tuple)):
            self.K = K
        
        if inputCstr is None:
            inputCstr = boxCstr(self.nu)
        self.inputCstr = inputCstr
        
        #Get the polynomials
        #Get the monomials up to degree four
        #TBD this should probably be done somewhere else
        self.polyDegrees=[np.zeros((self.nx,)).astype(np.int_) for i in range(self.nx)]
        self.polyNumbers=np.zeros((3,)).astype(np.int_)
        self.polyNumbers[0] = self.nx
        #linear terms
        for k in range(self.nx):
            self.polyDegrees[k][k]=1
        #Poly degree 2
        ind = self.nx
        for i in range(self.nx):
            for j in range(i, self.nx):
                self.polyDegrees.append(self.polyDegrees[i]+self.polyDegrees[j])
                ind += 1
        self.polyNumbers[1]=ind
        #Poly degree 3
        for i in range(self.nx):
            for j in range(self.polyNumbers[0], self.polyNumbers[1]):
                #Ensure uniqueness
                thisCoeffs = self.polyDegrees[i]+self.polyDegrees[j]
                if not np.any(np.min(thisCoeffs == self.polyDegrees, axis=1)):
                    self.polyDegrees.append(thisCoeffs)
                    ind += 1
        self.polyNumbers[2]=ind
        
        #Get convenience function for evaluation
        #Get variable list
        self.xTrans = sMa(sympy.symbols('y:{0:d}'.format(self.nx)))
        self.polyVars=[[] for i in range(len(self.polyDegrees))]
        self.polyVarsTrans=[[] for i in range(len(self.polyDegrees))]
        for k in range(len(self.polyDegrees)):
            for i in range(self.nx):
                self.polyVars[k] += self.polyDegrees[k][i]*[self.x[i]]
                self.polyVarsTrans[k] += self.polyDegrees[k][i]*[self.xTrans[i]] 
        
        self.polyVarsM = sMa(list(map( sympy.prod, self.polyVars)))
        
        #get a fast polynomial function
        self.polyEval = sympy.lambdify( self.x, self.polyVarsM, 'numpy' )
        self.polyEvalS = lambda *args: self.polyEval(*args).reshape((self.polyNumbers[-1], args[0].size))
        
        #Function to evaluate polynomial
        self.getPolyValue = lambda x, x0=0.: np.array( self.polyVarsM.evalf(64, subs=dict(zip(self.x, x-x0))) ).astype(np.float_)
        self.getPolyValueTrans = lambda thisPoly, x, x0=0.: np.array( thisPoly.evalf(64, subs=dict(zip(self.xTrans, x-x0))) ).astype(np.float_)
        #Function to return symbolic polynomial in translated/rotated space
        
        
        #Get the jacobian of f and g with respect to 'x'
        #These calcs can take quite a while if the system is "large" but have to be performed only once
        #There fore we store the result
        if sysName is None:
            sysName = str(self.f)[0:50]+str(self.g)[0:50]
        self.sysName = sysName
        allItems = ['jacF', 'jacG', 'hessF', 'hessG', 'polyA', 'polyB']
        if os.path.isfile('../dynSys/'+sysName+'.pickle'):
            with open('../dynSys/'+sysName+'.pickle', 'rb') as f:
                inDict = pickle.load(f)
                for aIt in allItems:
                    exec('''self.{0}=inDict['{0}']'''.format(aIt))
        else:
            _unused, _unused, self.jacF, self.jacG = self.returnTransLin()
            #Get the hessian of f and g with respect to 'x'
            _unused, _unused, self.hessF, self.hessG = self.returnTransHess()
            #get the polynomial approx upto deg3
            _unused, _unused, self.polyA, self.polyB = self.returnTransPoly()
            
            subprocess.call(["mkdir", "-p", '../dynSys/'])
            with open('../dynSys/'+sysName+'.pickle', 'wb+') as f:
                outDict = {}
                for aIt in allItems:
                    exec('''outDict['{0}'] = self.{0}'''.format(aIt))
                pickle.dump(outDict, f)
        
        #Lambdify the approximations
        thisFunc0 = sympy.lambdify(self.x, self.jacF, 'numpy')
        self.getLinAFast = lambda xList: thisFunc0(*list(xList)).reshape((self.nx, self.nx))
        
        thisFunc1 = sympy.lambdify(self.x, self.jacG, 'numpy')
        self.getLinBFast = lambda xList: list(map( lambda aLinG: aLinG.reshape((self.nx, self.nx)),  thisFunc1(*list(xList)) ))
        
        thisFunc2 = sympy.lambdify(self.x, self.polyA, 'numpy')
        self.getPolyAFast = lambda xList: thisFunc2(*list(xList)).reshape((self.nx, self.polyNumbers[-1]))
        
        thisFunc3 = sympy.lambdify(self.x, self.polyB, 'numpy')
        self.getPolyBFast = lambda xList: lmap(lambda aPolyG: aPolyG.reshape((self.nx, self.polyNumbers[-1])), thisFunc3(*list(xList)))
        
        
        #This function is constructed such that only ONE point can be evaluated, no vectorization!
        self.getTaylorS3Fast = lambda xList: [self.fEval(xList), self.gEval(xList), self.getPolyAFast(xList), self.getPolyBFast(xList)]
        
        #Function to get the deriv
        
        ##########################
        #Define the so2 states
        if so2DimsIn is None:
            self.so2Dims = None
        else:
            self.so2Dims = np.array(so2DimsIn).astype(np.int_).squeeze()
            assert np.all(self.so2Dims<self.nx)
        
        
        return None
    
    def gEval(self, xList):
        #Get the linear input dynamics around the points stocked in xList
        if isinstance(xList, (tuple, list)):
            xList = np.array(xList)
        
        if not xList.shape == (self.nx, xList.size//self.nx):
            try:
                xList.resize((self.nx, xList.size//self.nx))
            except ValueError:
                xList = xList.reshape((self.nx, xList.size//self.nx))
        
        allG = np.zeros((self.nx, self.nu, xList.shape[1]))
        for k in range(xList.shape[1]):
            allG[:,:,k] = self.gt(*xList[:,k])
        
        if xList.shape[1] == 1:
            allG.resize((self.nx, self.nu))
        else:
            allG.resize((self.nx, self.nu, xList.shape[1]))
        
        return allG
    
    ##################################################################
    #Function to return symbolic polynomial in translated/rotated space
    #If C and x0 are None, then the polynomials are in the original space
    def getPolyVarsTrans(self, C=None, x0=None, z=None, x=None, xT=None):
        #replace x in the monomials by C^-1.y + x0
        #This function can probably be accelerated using numpify
        if C is None:
            C = sMe(self.nx)
        C = sMa(C)
        if x0 is None:
            x0 = np.zeros((self.nx,1))
        if z is None:
            z=self.polyVarsM
        if x is None:
            x=self.x
        if xT is None:#
            xT=self.xTrans
        
        Ci = C.inv()
        thisSubs = dict(zip(x, Ci*xT+x0))
        
        thisTransPoly = sympy.expand(z.subs(thisSubs))
        
        return thisTransPoly
        
        
    ##################################################################
    #Return the functions in transformed space
    def returnTransFun(self, C=None, x0 = None):
        #Attention this ONLY changes the point of application; So it does not apply a full diffeomorphism
        #it will replace x in f(x) by C^-1.y+x0 <-> y = C.(x-x0) 
        if C is None:
            C = sMe(self.nx)
        C = sMa(C)
        if x0 is None:
            x0 = np.zeros((self.nx,1))
        
        Ci = C.inv()
        thisSubs = dict(zip(self.x, Ci*self.xTrans+x0))
        
        f2 = self.f.subs(thisSubs)
        g2 = self.g.subs(thisSubs)
        
        return f2, g2
    
    ##################################################################
    #Return the linearised dynamics
    def returnTransLin(self, C=None, x0=None):
        
        f2, g2 = self.returnTransFun(C, x0)
        
        thisReSubs = dict(zip(self.xTrans, self.x))
        
        jF2 = f2.jacobian(self.xTrans).subs(thisReSubs)
        
        jG2 = [g2[:,k].jacobian(self.xTrans) for k in range(g2.shape[1])]
        
        #f2, g2 = f2.subs(thisReSubs), g2.subs(thisReSubs)
        return f2, g2, jF2, jG2
    
    ##################################################################
    #Return the hessian (all terms derived to times with respect to x) 
    def returnTransHess(self, C=None, x0=None):
        
        f2, g2 = self.returnTransFun(C, x0)
        
        thisReSubs = dict(zip(self.xTrans, self.x))
        
        #Get the corresponding hessians
        hessF2 = sTa(np.zeros((self.nx, self.nx, self.nx)))
        for k in range(self.nx):
            #very ugly change
            thisM = sympy.hessian(f2[k], self.xTrans).subs(thisReSubs)
            for ik in range(self.nx):
                for jk in range(self.nx):
                    hessF2[k,ik,jk] = thisM[ik,jk]
        
        hessG2 = sTa(np.zeros((self.nu, self.nx, self.nx)))
        for k in range(self.nu):
            #very ugly change
            thisM = sympy.hessian(g2[k], self.xTrans).subs(thisReSubs)
            for ik in range(self.nx):
                for jk in range(self.nx):
                    hessG2[k,ik,jk] = thisM[ik,jk]
        
        #f2, g2 = f2.subs(thisReSubs), g2.subs(thisReSubs)
        return f2, g2, hessF2, hessG2
    
    ##################################################################
    #Get the polynomial approximation as symbolic expressions
    def returnTransPoly(self, C=None, x0=None):
        
        f2, g2 = self.returnTransFun(C)
        
        thisReSubs = dict(zip(self.xTrans, self.x))
        
        #Get polynomial approximation
        #Replace with derivative by array
        polyA = sMz( self.nx, self.polyNumbers[-1] )
        for i in range(self.nx):    
            for j in range(self.polyNumbers[-1]):
                fac=1.
                if j >= self.polyNumbers[0]:
                    fac=0.5
                if j >= self.polyNumbers[1]:
                    fac=1./(3.*2.*1.)
                polyA[i,j] = fac*sympy.diff(f2[i], *self.polyVarsTrans[j]).subs(thisReSubs)
        #polyB = sTa( np.zeros(( self.nx, self.polyNumbers[-1], self.nu )) )
        polyB = [sMa( np.zeros(( self.nx, self.polyNumbers[-1] )) ) for k in range(self.nu)]
        for i in range(self.nx):
            for j in range(self.polyNumbers[-1]):
                for k in range(self.nu):
                    fac=1.
                    if j >= self.polyNumbers[0]:
                        fac=0.5
                    if j >= self.polyNumbers[1]:
                        fac=1./(3.*2.*1.)
                    #polyB[i,j,k] = fac*sympy.diff(g2[i,k], *self.polyVars[j]).subs(thisReSubs)
                    polyB[k][i,j] = fac*sympy.diff(g2[i,k], *self.polyVarsTrans[j]).subs(thisReSubs)
                    
        #f2, g2 = f2.subs(thisReSubs), g2.subs(thisReSubs)
        return f2, g2, polyA, polyB
    
    ##################################################################
    def getLin(self, x0, C=None, prec=64):
        
        #Attention this function returns the linearization of the function with respect to the new variables, but the output remains unchanged
        #f(x) = f(C^-1.y+x0)
        #f(x0) + Ax.(x-x0) = f(x0) + Ay.y 
        
        x0 = x0.reshape((x0.size,))
        
        if C is None:
            C = np.eye(self.nx)
            thisF, thisG, thisJacF, thisJacG = self.f, self.g, self.jacF, self.jacG
        else:
            thisF, thisG, thisJacF, thisJacG = self.returnTransLin(C)
        
        thisF, thisG = self.f, self.g
        
        thisSubsX = dict(zip( self.x, x0 ))
        thisSubsY = dict(zip( self.x, np.dot(C, x0) )) #y0=C.y0
        
        #return self.f.n(prec, thisSubs), self.g.n(prec, thisSubs), self.jacF.n(prec, thisSubs), self.jacG.n(prec, thisSubs) 
        #return np.dot(C, np.array(thisF.evalf(prec, subs=thisSubsX)).astype(np.float_)), np.dot(C, np.array(thisG.evalf(prec, subs=thisSubsX)).astype(np.float_)), np.dot(C, np.array(thisJacF.evalf(prec, subs=thisSubsY)).astype(np.float_)), np.dot(C, np.array(thisJacG.evalf(prec, subs=thisSubsY)).astype(np.float_))
        return myEvalf(thisF, prec, thisSubsX), myEvalf(thisG, prec, thisSubsX), myEvalf(thisJacF, prec, thisSubsY), lmap(partialFunc(myEvalf, prec=prec, subsDict=thisSubsY), thisJacG)
    
    ##################################################################
    
    def getTaylorS3(self, x0, C=None, prec=64):    
        
        #Attention this function returns the polynomial approximation of the function with respect to the new variables, but the output remains unchanged
        #f(x) = f(C^-1.y+x0)
        #f(x0) + Apolyx.z(x-x0) = f(x0) + Apolyy.z(y) 
                
        
        x0 = x0.reshape((x0.size,))
        if C is None:
            C = np.eye(self.nx)
            _unused, _unused, thisPolyA, thisPolyB = self.f, self.g, self.polyA, self.polyB
        else:
            _unused, _unused, thisPolyA, thisPolyB = self.returnTransPoly(C)
        
        thisF, thisG = self.f, self.g
        
        thisSubsX = dict(zip( self.x, x0 ))
        thisSubsY = dict(zip( self.x, np.dot(C, x0) ))#y0=C.y0
        
        #return self.f.n(prec, thisSubs), self.g.n(prec, thisSubs), self.polyA.n(prec, thisSubs), self.polyB.n(prec, thisSubs) 
        #return np.dot(C, np.array(thisF.evalf(prec, subs=thisSubsX)).astype(np.float_)), np.dot(C, np.array(thisG.evalf(prec, subs=thisSubsX)).astype(np.float_)), np.dot(C, np.array(thisPolyA.evalf(prec, subs=thisSubsY)).astype(np.float_)), np.dot(C, np.array(thisPolyB.evalf(prec, subs=thisSubsY)).astype(np.float_))
        #myEvalf = lambda arr: np.dot(C, np.array(arr.evalf(prec, subs=thisSubsY)).astype(np.float_))
        #thisPolyBRet = list(map(myEvalf, thisPolyB))
        #return np.dot(C, np.array(thisF.evalf(prec, subs=thisSubsX)).astype(np.float_)), np.dot(C, np.array(thisG.evalf(prec, subs=thisSubsX)).astype(np.float_)), np.dot(C, np.array(thisPolyA.evalf(prec, subs=thisSubsY)).astype(np.float_)), thisPolyBRet
        return myEvalf(thisF, prec, thisSubsX), myEvalf(thisG, prec, thisSubsX), myEvalf(thisPolyA, prec, thisSubsY), list(map(partialFunc(myEvalf, prec=prec, subsDict=thisSubsY), thisPolyB))
    ##################################################################
    
    def getIndU(self, Pt, x0, x, allInputPerms):
        #Return a vector of integers encoding the optimal input.
        #Given the center of the ellipsoid x0, its shape Pt and the considered points
        #x
        #1 <-> Maximal input 
        #0 <-> Minimal input
        nPt = x.size//self.nx
        
        x0.resize((self.nx,1))
        
        x.resize((self.nx, nPt))
        dx = x-x0
        if not(self.so2Dims is None):
            dx[self.so2Dims,:]=toSo2(dx[self.so2Dims,:])
        
        allB = self.gEval(x)
        allB.resize(self.nx,self.nu, allB.size//(self.nx*self.nu))
        uStar = cdot3(np.ascontiguousarray(np.transpose(allB, [1,0,2])), ndot(Pt, dx)) #This is not very effective since a useless copy is generated
        uStar[uStar<=0]=0
        uStar[np.logical_not(uStar<=0)]=1
        
        uStar = uStar.astype(np.int_)
        
        ind = np.zeros((uStar.shape[1]))
        for k in range(uStar.shape[1]):
            ind[k] = allInputPerms.index( list(uStar[:,k]) )
        
        return ind
    
    
    ##################################################################
    def getBestF(self, Pt, x0, x, mode='OO', u0 = None, t=0., fullOut=False):
        #Compute the dynamics for optimal input
        #Pt shape of ellipsoid
        #x0 center of the ellipsoid
        #x all considered points
        #mode choosing the dynamics
        #first char: system dynamics:
        #    O for original nonlinear sys
        #    P for polynomial approx around x0
        #    L for linear approx around x0
        #second char: input dynamics and control:
        #    O for original nonlinear input dyn with discontinuous optimal control
        #    R for original nonlinear input dyn with smoothened optimal control (parameter self.innerzone)
        #    P for polynomial approx of input dyn around x0 with discontinuous optimal control 
        #    L for linear approx of input dyn around x0 with discontinuous optimal control
        nPt = x.size//self.nx
        
        if u0 is None:
            u0 = np.zeros((self.nu,1))
        
        x0.resize((self.nx,1))
        
        if self.secondOrderSlide:
            x = x.reshape((self.nx+self.nU, nPt))
            lastS = x[self.nx:,:]
            x = x[:self.nx,:]        
        else:
            try:
                x.resize((self.nx, nPt))
            except:
                x = x.reshape((self.nx, nPt))
        
        thisN = x.shape[1]
        
        dx = x-x0
        #Take care of So2
        if not(self.so2Dims is None):
            x[self.so2Dims,:]=toSo2(x[self.so2Dims,:])
            x0[self.so2Dims,:]=toSo2(x0[self.so2Dims,:])
            dx[self.so2Dims,:]=toSo2(dx[self.so2Dims,:])
        
        
        if mode[0] == 'O':
            f = self.fEval(x)
        elif mode[0] == 'P':
            thisPolyA = self.getPolyAFast(x0)
            f = self.fEval(x0) + np.dot(thisPolyA, self.polyEvalS(*dx))
        elif mode[0] == 'L':
            thisA = self.getLinAFast(x0)
            f = self.fEval(x0) + np.dot(thisA, dx)
        else:
            assert 0
        
        
        uLimL = self.inputCstr.getMinU(t)
        uLimU = self.inputCstr.getMaxU(t)
        if mode[1] == 'O':
            allB = self.gEval(x)
            allB.resize(self.nx,self.nu, allB.size//(self.nx*self.nu))
            #x'.Pt.B.u -> u'.B'.Pt.x
            #This only works for bounding box
            uStar = cdot3(np.ascontiguousarray(np.transpose(allB, [1,0,2])), ndot(Pt, dx)) #This is not very effective since a useless copy is generated            
            for k in range(self.nu):
                ind = uStar[k,:]<=0.
                uStar[k, ind] = uLimU[k]#Converging for positive u
                uStar[k, np.logical_not(ind)] = uLimL[k]#Converging for negative u
            
            g = cdot3(allB, uStar)
        elif mode[1] == 'R':
            #This assumes u0 given
            allB = self.gEval(x)
            if nPt > 1:
                allB.resize(self.nx,self.nu, allB.size//(self.nx*self.nu))
                #allB = np.divide( allB, np.tile(np.linalg.norm(allB, axis=0),(self.nx,1,allB.size//(self.nx*self.nu))) )
                #x'.Pt.B.u -> u'.B'.Pt.x
                #This only works for bounding box
                uStar = cdot3(np.ascontiguousarray(np.transpose(allB, [1,0,2])), ndot(Pt, dx)) #This is not very effective since a useless copy is generated
            else:
                uStar = ndot(allB.T, Pt, dx)
            #deltaBorderCtrl = np.square(np.tanh(uStar/self.innerZone))
            #deltaBorderRef = 1.-np.square(np.tanh(uStar/(self.innerZone)))
            if not self.secondOrderSlide:
                deltaBorderCtrl = np.minimum(  np.abs(uStar)/self.innerZone,1. )
                deltaBorderRef = 1.-deltaBorderCtrl
                for k in range(self.nu):
                    ind = uStar[k,:]<=0.
                    uStar[k,:]=deltaBorderRef[k,:]*u0[k]
                    uStar[k, ind] += uLimU[k]*deltaBorderCtrl[k,ind]#Converging for positive u
                    uStar[k, np.logical_not(ind)] += uLimL[k]*deltaBorderCtrl[k,np.logical_not(ind)]#Converging for negative u
                    uStar[k,:] = np.minimum(uStar[k:], uLimU)
                    uStar[k,:] = np.maximum(uStar[k:], uLimL)
            else:
                dS = -(np.sign(uStar)-lastS)/self.innerZone
                for k in range(self.nu):
                    indM = lastS[k,:]<-1e-5
                    indP = lastS[k,:]>+1e-5
                    ind0 = np.logical_not( np.logical_or(indP, indM) )
                    uStar[k, indM] = uLimU[k]#Converging for positive u
                    uStar[k, indP] = uLimL[k]#Converging for negative u
                    uStar[k, ind0] = u0[k]#Fairly close
            
            if nPt>1:
                g = cdot3(allB, uStar)
            else:
                g = ndot(allB, uStar)
        elif mode[1] == 'P':
            Blin0 = self.gEval(x0)
            Blin0.resize((self.nx, self.nu, 1))
            zEval = self.polyEvalS(*dx)
            
            
            #allB[i,k] = polyB[k][i,:].z
            #allPoly = np.array(self.getPolyBFast(x0))
            #allPoly = np.ascontiguousarray( np.transpose(allPoly, [1,2,0]), dtype=np.float_ )
            allPoly = np.stack(self.getPolyBFast(x0), axis=1)#allPoly[i,k,j] = polyB[k][i,j]

            #allB = cdot3(allPoly, zEval)+np.broadcast_t(Blin0, [self.nx, self.nu, thisN])
            #Get B for each point
            #allB[i,k,n] = allPoly[i,k,:].zEval[:,n]
            allB = np.einsum('ikj,jn->ikn', allPoly, zEval) + np.broadcast_to(Blin0, (self.nx, self.nu, nPt))

            # dxp = Pt.dx
            # ustar[:,n] = (allB[:,:,n]).T.dxp[:,n]
            # ustar[i,n] = sum_j allB[j,i,n]).dxp[j,n]
            # uStar = cdot3(np.ascontiguousarray(np.transpose(allB, [1,0,2])), ndot(Pt, dx)) #This is not very effective since a useless copy is generated
            uStar = np.einsum('jin,jn->in', allB, np.dot(Pt, dx))
            for k in range(self.nu):
                ind = uStar[k,:]<=0.
                uStar[k, ind] = uLimU[k]#Converging for positive u
                uStar[k, np.logical_not(ind)] = uLimL[k]#Converging for negative u
            #g[:,n] = allB[:,:,n].ustar[:,n]
            #g[i,n] = sum_j allB[i,j,n].ustar[j,n]
            #g = np.einsum('ijn,jn->in', allB, uStar)
            g = cdot3(allB, uStar)
            
        elif mode[1] == 'L':
            Blin = self.gEval(x0)
            uStar = ndot(Blin.T, Pt, dx)
        
            for k in range(self.nu):
                ind = uStar[k,:]<=0.
                uStar[k, ind] = uLimU[k]#Converging for positive u
                uStar[k, np.logical_not(ind)] = uLimL[k]#Converging for negative u
            
            g = ndot(Blin, uStar)
        #print(g)
        #if np.any(np.linalg.norm( f+g, axis=0 ) > 1e3):
        #    print("aaa")
        if self.secondOrderSlide:
            return np.vstack((f+g, dS))
        else:
            if fullOut:
                return f+g, uStar
            else:
                return f+g

    ##################################################################
    #Introduce a constraint quadratic optimization to substitute
    #siding mode control
    def getBestFQP(self,Pt,dPt,alpha,x0,x,mode='OO', u0=None,t=0.,regEps=1.0,acceptError=True, fullOut=False, doScaleCost=False ):
        # Compute the dynamics for optimal input
        # Pt shape of ellipsoid
        # dPt derivative of the shape of ellipsoid
        # alpha convergence
        # x0 center of the ellipsoid
        # x all considered points
        # mode Considered dynamic for proper and input dynamics; L:linear, P:Polynomial, O:
        # Since we have x_dot = f(x) + g(x).u
        # and V(x) = x'.Pt.x
        # we get
        # d/dt V(x) = 2 x'.Pt.x_dot + x'.dPt.x
        # Then in order to impose the convergence we have
        # 2 x'.Pt.x_dot + x'.dPt.x <= -alpha x'.Pt.x
        # Then we separate the variables and "constants" to form the linear constraint
        # 2 x'.Pt.g(x).u <= -2 x'.Pt.f(x) - x'.dPt.x - alpha x'.Pt.x
        # Then we wish to find a compromise between control effort and convergence while
        # guaranteeing the minimal convergence
        # So
        # min     u'.u*regEps + x'.Pt.g(x).u
        # s.t.    2 x'.Pt.g(x).u <= -2 x'.Pt.f(x) - x'.dPt.x - alpha x'.Pt.x
        #         u_Min <= u <= u_Max

        nPt = x.size//self.nx

        x0.resize((self.nx,1))

        try:
            x.resize((self.nx,nPt))
        except:
            x = x.reshape((self.nx,nPt))

        # Take care of So2
        if not (self.so2Dims is None):
            x[self.so2Dims,:] = toSo2(x[self.so2Dims,:])
            x0[self.so2Dims,:] = toSo2(x0[self.so2Dims,:])

        
        dx = x - x0
        if not (self.so2Dims is None):
            dx[self.so2Dims, :] = toSo2(dx[self.so2Dims, :])

        #Evaluate input independent dynamics
        if mode[0] == 'O':
            f = self.fEval(x)
        elif mode[0] == 'P':
            thisPolyA = self.getPolyAFast(x0)
            f = self.fEval(x0)+np.dot(thisPolyA,self.polyEvalS(*dx))
        elif mode[0] == 'L':
            thisA = self.getLinAFast(x0)
            f = self.fEval(x0)+np.dot(thisA,dx)
        else:
            assert 0, "Unknown mode specification for input independent dynamics: {0}".format(mode[0])
        
        # Get current input constraints
        uLimL = self.inputCstr.getMinU(t)
        uLimU = self.inputCstr.getMaxU(t)
        # Evaluate input dependent dynamics
        if mode[1] == 'O':
            allB = self.gEval(x)
            allB.resize(self.nx,self.nu,allB.size//(self.nx*self.nu))
        elif mode[1] == 'P':
            Blin0 = self.gEval(x0)
            Blin0.resize((self.nx, self.nu, 1))
            zEval = self.polyEvalS(*dx)

            # allB[i,k] = polyB[k][i,:].z
            # allPoly = np.array(self.getPolyBFast(x0))
            # allPoly = np.ascontiguousarray( np.transpose(allPoly, [1,2,0]), dtype=np.float_ )
            allPoly = np.stack(self.getPolyBFast(x0), axis=1)  # allPoly[i,k,j] = polyB[k][i,j]

            # allB = cdot3(allPoly, zEval)+np.broadcast_t(Blin0, [self.nx, self.nu, thisN])
            # Get B for each point
            # allB[i,k,n] = allPoly[i,k,:].zEval[:,n]
            allB = np.einsum('ikj,jn->ikn', allPoly, zEval)+np.broadcast_to(Blin0, (self.nx, self.nu, nPt))
        elif mode[1] == 'L':
            Blin0 = self.gEval(x0)
            Blin0.resize((self.nx, self.nu, 1))
            allB = np.broadcast_to(Blin0, (self.nx, self.nu, nPt))
        else:
            assert 0,"Unknown mode specification for input dynamics: {0}".format(mode[1])
            
        # Get the ideal input
        #Fill up with the sliding mode controls in order to have a control law for points that can not be stabilised
        allU = np.einsum('jin,jn->in',allB,np.dot(Pt,dx))
        for k in range(self.nu):
            ind = allU[k,:] <= 0.
            allU[k,ind] = uLimU[k]  # Converging for positive u
            allU[k,np.logical_not(ind)] = uLimL[k]  # Converging for negative u
        # For optimization
        # Set up constraints
        # These matrices are typically small, so the overhead of sparse is larger than the gain
        Gl = matrix( np.vstack(( Id(self.nu), -Id(self.nu), zeros((1,self.nu)) )) )
        Hl = matrix( np.vstack((uLimU, -uLimL, 0)).astype(np.float_).squeeze() )
        # Loop over all points, cost function and additional constraint depends on point
        if fullOut:
            indSucc = zeros((nPt,)).astype(np.bool_)
            indIsTight = zeros((nPt,)).astype(np.bool_)
            anyOutIsMax = zeros((nPt,)).astype(np.bool_)
        #Test
        dY = empty((self.nx,1))
        Ct = getP2(Pt)
        By = empty((self.nx, self.nu))
        for i in range(nPt):
            # Constraint
            # Compute
            # -2 x'.Pt.f(x) - x'.dPt.x - alpha x'.Pt.x
            Hl[2*self.nu] = -2*ndot( dx[:,[i]].T, Pt, f[:,[i]]  ) - ndot( dx[:,[i]].T, dPt, dx[:,[i]] )  - 0.99*alpha*ndot( dx[:,[i]].T, Pt, dx[:,[i]] )#Added a numeric epsilon for the convergence
            #Compute 2 x'.Pt.g(x)
            convU = 2.*ndot( dx[:,[i]].T, Pt, allB[:,:,i] ).squeeze()
            #Set dependent part of convergence
            Gl[2*self.nu,:] = convU

            # Get optimization function
            convUNorm = np.linalg.norm(convU)
            if not doScaleCost:
                regEpsTilde = regEps#*convUNorm
            else:
                regEpsTilde = regEps*convUNorm
            Hl[2*self.nu] /= convUNorm
            Gl[2*self.nu,:] /= convUNorm
            # convU /= convU
            P = matrix(Id(self.nu)*regEpsTilde)
            q = matrix(convU)

            try:
                res = solvers.coneqp( P, q, Gl, Hl )
                assert (res['status'] == 'optimal') or ( (res['status'] == 'unknown') and (np.all(na(Gl*res['x'])<=na(Hl)+1e-5)) )
                allU[:,i] = np.array(res['x'])
                if fullOut:
                    indSucc[i]=np.True_
                    indIsTight[i] = abs((Gl[[2*self.nu],:]*res['x']-Hl[2*self.nu])[0,0]) < 1e-3
                    anyOutIsMax[i] = np.any( np.abs(allU[:,[i]]-uLimL)<1e-2 ) or np.any( np.abs(allU[:,[i]]-uLimU)<1e-2 )
            except:
                if acceptError:
                    import warnings
                    thisWarnStr = 'convergence can not be obtained or optimization failed, using best allowed U for point {0}'.format(i)
                    warnings.warn(thisWarnStr, UserWarning)
                else:
                    assert 0, 'Optimization failed'
        #Done computing all control inputs
        
        #Now get the resulting dynamics
        xdot = f+cdot3(allB, allU)
        
        #Done
        if fullOut:
            return xdot, indSucc, indIsTight, anyOutIsMax, allU
        else:
            return xdot

    ##################################################################
            # Introduce a constraint quadratic optimization to substitute

    def getBestFQP2(self,Pt,dPt,alpha,x0,x,mode='OO',u0=None,t=0.,regEps=1.0,acceptError=True,fullOut=False,doScaleCost=False):
        # Compute the dynamics for optimal input
        # Pt shape of ellipsoid
        # dPt derivative of the shape of ellipsoid
        # alpha convergence
        # x0 center of the ellipsoid
        # x all considered points
        # mode Considered dynamic for proper and input dynamics; L:linear, P:Polynomial, O:
        # Since we have x_dot = f(x) + g(x).u
        # and V(x) = x'.Pt.x
        # we get
        # d/dt V(x) = 2 x'.Pt.x_dot + x'.dPt.x
        # Then in order to impose the convergence we have
        # 2 x'.Pt.x_dot + x'.dPt.x <= -beta x'.Pt.x
        # Then we separate the variables and "constants" to form the linear constraint
        # 2 x'.Pt.g(x).u <= -2 x'.Pt.f(x) - x'.dPt.x - beta x'.Pt.x
        # Then we wish to find a compromise between control effort and convergence while
        # guaranteeing the minimal convergence
        # So
        # min     u'.u*regEps - beta'
        # s.t.    2 x'.Pt.g(x).u + beta' x'.Pt.x <= -2 x'.Pt.f(x) - x'.dPt.x - beta x'.Pt.x
        #         u_Min <= u <= u_Max

        nPt = x.size//self.nx

        x0.resize((self.nx,1))

        try:
            x.resize((self.nx,nPt))
        except:
            x = x.reshape((self.nx,nPt))

        # Take care of So2
        if not (self.so2Dims is None):
            x[self.so2Dims,:] = toSo2(x[self.so2Dims,:])
            x0[self.so2Dims,:] = toSo2(x0[self.so2Dims,:])

        dx = x-x0
        if not (self.so2Dims is None):
            dx[self.so2Dims,:] = toSo2(dx[self.so2Dims,:])

        # Evaluate input independent dynamics
        if mode[0] == 'O':
            f = self.fEval(x)
        elif mode[0] == 'P':
            thisPolyA = self.getPolyAFast(x0)
            f = self.fEval(x0)+np.dot(thisPolyA,self.polyEvalS(*dx))
        elif mode[0] == 'L':
            thisA = self.getLinAFast(x0)
            f = self.fEval(x0)+np.dot(thisA,dx)
        else:
            assert 0,"Unknown mode specification for input independent dynamics: {0}".format(mode[0])

        # Get current input constraints
        uLimL = self.inputCstr.getMinU(t)
        uLimU = self.inputCstr.getMaxU(t)
        # Evaluate input dependent dynamics
        if mode[1] == 'O':
            allB = self.gEval(x)
            allB.resize(self.nx,self.nu,allB.size//(self.nx*self.nu))
        elif mode[1] == 'P':
            Blin0 = self.gEval(x0)
            Blin0.resize((self.nx,self.nu,1))
            zEval = self.polyEvalS(*dx)
    
            # allB[i,k] = polyB[k][i,:].z
            # allPoly = np.array(self.getPolyBFast(x0))
            # allPoly = np.ascontiguousarray( np.transpose(allPoly, [1,2,0]), dtype=np.float_ )
            allPoly = np.stack(self.getPolyBFast(x0),axis=1)  # allPoly[i,k,j] = polyB[k][i,j]
    
            # allB = cdot3(allPoly, zEval)+np.broadcast_t(Blin0, [self.nx, self.nu, thisN])
            # Get B for each point
            # allB[i,k,n] = allPoly[i,k,:].zEval[:,n]
            allB = np.einsum('ikj,jn->ikn',allPoly,zEval)+np.broadcast_to(Blin0,(self.nx,self.nu,nPt))
        elif mode[1] == 'L':
            Blin0 = self.gEval(x0)
            Blin0.resize((self.nx,self.nu,1))
            allB = np.broadcast_to(Blin0,(self.nx,self.nu,nPt))
        else:
            assert 0,"Unknown mode specification for input dynamics: {0}".format(mode[1])

        # Get the ideal input
        # Fill up with the sliding mode controls in order to have a control law for points that can not be stabilised
        allU = np.einsum('jin,jn->in',allB,np.dot(Pt,dx))
        for k in range(self.nu):
            ind = allU[k,:] <= 0.
            allU[k,ind] = uLimU[k]  # Converging for positive u
            allU[k,np.logical_not(ind)] = uLimL[k]  # Converging for negative u
        # For optimization
        # Set up constraints
        # These matrices are typically small, so the overhead of sparse is larger than the gain
        Gl = matrix( np.hstack(( np.vstack((Id(self.nu),-Id(self.nu),zeros((2,self.nu)))), zeros((2*self.nu+2,1)) ))  )
        #For convergence
        Gl[2*self.nu+1,self.nu] = -1 #\betaAdd>0
        Hl = matrix(np.vstack((uLimU,-uLimL,0,0)).astype(np.float_).squeeze())
        # Loop over all points, cost function and additional constraint depends on point
        if fullOut:
            indSucc = zeros((nPt,)).astype(np.bool_)
            indIsTight = zeros((nPt,)).astype(np.bool_)
            anyOutIsMax = zeros((nPt,)).astype(np.bool_)
        for i in range(nPt):
            # Constraint
            # Compute
            # -2 x'.Pt.f(x) - x'.dPt.x - beta x'.Pt.x
            Vx = ndot(dx[:,[i]].T,Pt,dx[:,[i]])
            Hl[2*self.nu] = -2*ndot(dx[:,[i]].T,Pt,f[:,[i]])-ndot(dx[:,[i]].T,dPt,dx[:,[i]])-0.99*alpha*Vx  # Added a numeric epsilon for the convergence
            # Compute 2 x'.Pt.g(x)
            convU = 2.*ndot(dx[:,[i]].T,Pt,allB[:,:,i]).squeeze()
            # Set dependent part of convergence
            Gl[2*self.nu,:self.nu] = convU
            Gl[2*self.nu,self.nu] = Vx
    
            # Get optimization function
            convUNorm = np.linalg.norm(convU)
            Hl[2*self.nu] /= convUNorm
            Gl[2*self.nu,:] /= convUNorm
            # convU /= convU
            P = matrix(Id(self.nu+1)*regEps)
            P[self.nu,self.nu] = 0
            q = matrix(zeros(self.nu+1))
            q[self.nu] = -1.
    
            try:
                res = solvers.coneqp(P,q,Gl,Hl)
                assert (res['status'] == 'optimal') or (
                (res['status'] == 'unknown') and (np.all(na(Gl*res['x']) <= na(Hl)+1e-5)))
                allU[:,i] = np.array(res['x'])[:self.nu]
                if fullOut:
                    indSucc[i] = np.True_
                    indIsTight[i] = abs((Gl[[2*self.nu],:]*res['x']-Hl[2*self.nu])[0,0]) < 1e-3
                    anyOutIsMax[i] = np.any(np.abs(allU[:,[i]]-uLimL) < 1e-2) or np.any(
                        np.abs(allU[:,[i]]-uLimU) < 1e-2)
            except:
                if acceptError:
                    import warnings
                    thisWarnStr = 'convergence can not be obtained or optimization failed, using best allowed U for point {0}'.format(
                        i)
                    warnings.warn(thisWarnStr,UserWarning)
                else:
                    assert 0,'Optimization failed'
        # Done computing all control inputs

        # Now get the resulting dynamics
        xdot = f+cdot3(allB,allU)

        # Done
        if fullOut:
            return xdot,indSucc,indIsTight,anyOutIsMax,allU
        else:
            return xdot
        
    ##################################################################
    def getBestForigUlin(self, Pt, x0, x):
        x0.resize((self.nx,1))
        x.resize((self.nx, x.size//self.nx))
        #Take care of So2
        if not(self.so2Dims is None):
            x[self.so2Dims,:]=toSo2(x[self.so2Dims,:])
            x0[self.so2Dims,:]=toSo2(x0[self.so2Dims,:])
        
        _, Blin, _, _ = self.getLin(x0, prec=32)
        
        #x'.Pt.B.u -> u'.B'.Pt.x
        #This only works for bounding box
        uLimL = self.inputCstr.limL
        uLimU = self.inputCstr.limU
        uStar = ndot(Blin.T, Pt, x-x0)
        
        for k in range(self.nu):
            ind = uStar[k,:]<=0.
            uStar[k, ind] = uLimU[k]#Converging for positive u
            uStar[k, np.logical_not(ind)] = uLimL[k]#Converging for negative u
        
        return self.fEval(x) + ndot(Blin, uStar)
    
    ##################################################################
        
    def __call__(self, x, u, prec = 64, doRestrain=True, doSo2=False, t=0., mode='OO', x0 = None):
        #Compute d/dt x for a given u and a given mode
        nPt = x.size//self.nx
        
        if not x.shape == (self.nx, nPt):
            try:
                x.resize((self.nx, nPt))
            except ValueError:
                x = x.reshape((self.nx, nPt))
        #Take care of So2
        if not(self.so2Dims is None) and doSo2:
            x[self.so2Dims,:]=toSo2(x[self.so2Dims,:])
        
        if doRestrain:
            u=self.inputCstr(u,t)
        #return self.f.n(prec, thisSubs) + np.dot(self.g.n(prec, thisSubs), u)
        try:
            u.resize((self.nu, u.size//self.nu))
        except ValueError:
            u  = u.reshape((self.nu, u.size//self.nu))
                    
        assert (u.shape[1] == x.shape[1]) or u.shape[1]==1, 'incompatible dimensions'
        
        if mode[0] == 'O':
            f = self.fEval(x)
        elif mode[0] == 'P':
            if x0 is None:
                x0 = np.zeros_like(x)
            dx = x-x0
            thisPolyA = self.getPolyAFast(x0)
            f = self.fEval(x0) + np.dot(thisPolyA, self.polyEvalS(*dx))
        elif mode[0] == 'L':
            if x0 is None:
                x0 = np.zeros_like(x)
            dx = x-x0
            thisA = self.getLinAFast(x0)
            f = self.fEval(x0) + np.dot(thisA, dx)
        else:
            assert 0
        
        #Check wether cdot3 axes align
        if mode[1]=='O':
            if nPt > 1:
                res = f + cdot3(self.gEval(x), u)
            else:
                res = f + ndot(self.gEval(x), u)
        elif mode[1]=='P':
            Blin0 = self.gEval(x0)
            Blin0.resize((self.nx, self.nu))
            zEval = self.polyEvalS(*dx)
            
            
            #allB[i,k] = polyB[k][i,:].z
            allPoly = np.array(self.getPolyBFast(x0))
            
            if nPt == 1:
                res = f + ndot( ndot(allPoly.reshape((self.nx,-1)), zEval)+Blin0, u)
            else:
                allPoly = np.ascontiguousarray( np.transpose(allPoly, [1,2,0]) )
                allB = cdot3(allPoly, zEval)+np.broadcast_t(Blin0, [self.nx, self.nu, nPt])
                res = f + ndot(allB, u)
            
        elif mode[1]=='L':
            if nPt > 1:
                res = f + cdot3(self.gEval(x0), u)
            else:
                res = f + ndot(self.gEval(x0), u)
        else:
            assert 0
            
        #if np.any(np.linalg.norm( res, axis=0 ) > 1e3):
        #    print("aaa")
        return res

    ##################################################################
    
###########################

class simulateDs():
    
    def __init__(self, dynSys, refTraj, ctrlLaw):
        self.dynSys = dynSys
        self.refTraj = refTraj
        self.ctrlLaw = ctrlLaw
        
        self.dx = lambda x, t: self.dynSys( x, self.dynSys.inputCstr( self.ctrlLaw(x, t, *self.refTraj(t)) )  ).squeeze()
    
    def simulate(self, x0, T, N=100):
        
        x0 = np.array(x0).reshape((-1,))
        
        T = np.array(T).reshape((-1,))
        
        if T.shape[0] == 2:
            T = np.linspace(T[0], T[1], N)
        
        X = scipy.integrate.odeint(self.dx, x0, T)
        
        
        return T, X

##############################

class linCtrl():
    def __init__(self, K):
        self.K = K

    def __call__(self, x, t, xref, xdref, uref):
        #xref = xxduref[0]
        #xdref = xxduref[1]
        #uref = xxduref[2]
        dx = x.reshape((x.size,1)) - xref.reshape((xref.size,1))
        uref.resize((uref.size,1))
        return uref + np.dot( self.K, dx )

##############################

class npcControl():
    #To be done; Not tested
    def __init__(self, lyapReg, dynSys=None):
        self.lyapReg = lyapReg
        if dynSys is None:
            self.dynSys = lyapReg.dynSys
        else:
            self.dynSys = dynSys
        self.refTraj = lyapReg.refTraj
        self.acceptingVal = 0.1
        #get all the permuations
        self.allInputs = [-1,0,1]
        self.allPerms = []
        self.getPerms()

    def getPerms(self):
        allPerms = lmap(lambda x: [x], self.allInputs)
        for k in range(1,self.dynSys.nu):
            allPermsNew = []
            for aPerm in allPerms:
                for aInput in self.allInputs:
                    allPermsNew.append( aPerm+[aInput])
            allPerms = allPermsNew
        self.allPerms = allPerms
    
    def simulate(self, Xinit, tInit, tFinal, dTother=.2, dTmpc = 0.25):
        #Apply the optimal control if the system converges fast enough, otherwise switch to 
        #mpc-control
        
        dim = self.dynSys.nx
        Ninit = 1
        tCurrent = tInit
        Xcurrent = Xinit.reshape((dim,1))
        fOdeInt = lambda x,t: self.dynSys.getBestF(self.lyapReg.getEllip(t)[1], self.refTraj.xref(t), x.reshape((Ninit,dim)).T, mode="OR", u0 = self.refTraj.uref(t)).T.reshape((dim*Ninit,))
        #Get the derivative for faster integration
        allX = np.zeros((0,dim))
        allT = np.zeros(0,)
        while tCurrent < tFinal:
            thisTFinal = min(tFinal, dTother+tCurrent)
            nextdT = thisTFinal-tCurrent
            #Get initial lyapval
            initVal = self.lyapReg.getCost(tCurrent, Xcurrent)
            maxEndVal = initVal*np.exp(-self.lyapReg.convergenceRate*nextdT)
            #Try regular integration
            thisT = np.linspace(tCurrent, thisTFinal, 25)
            thisX = scipy.integrate.odeint(fOdeInt, Xcurrent.squeeze(), thisT)#Add Deriv
            
            if (initVal > self.acceptingVal) and (not (  self.lyapReg.getCost(thisT[-1], thisX[-1,:].reshape((dim,1)) ) < maxEndVal )):
                #Do the mpc
                thisT = np.zeros(0,)
                thisX = np.zeros((0,dim))
                while True:
                    #return True, thisX, thisT, self.allPerms[k]
                    check, thisNewX, thisNewT, _ = self.applyCtrl(Xcurrent, tCurrent, thisTFinal, interSteps=5)
                    assert check, "No mpc solution was found"
                    thisT = np.hstack((thisT, thisNewT))
                    thisX = np.vstack((thisX, thisNewX))
                    if thisNewT[-1] == thisTFinal:
                        break
            
            Xcurrent = thisX[-1,:].reshape((dim,1))
            tCurrent = thisT[-1]
            allX = np.vstack((allX, thisX))
            allT = np.hstack((allT, thisT))
        
        return allT, allX.T, self.lyapReg.getCost(allT, allX.T)
            
        
    #############################################
    def applyCtrl(self, Xinit, tInit, tFinal, interSteps = 5):
        #To be done; Not tested
        Xcurrent = Xinit.reshape(self.dynSys.nx,1)
        allSolutions = [[Xcurrent,0,[]]]
        
        tSteps = np.linspace(tInit, tFinal, interSteps)
        tStepsDiff = tSteps[1:]-tSteps[:-1]
        tStepsAll = tSteps
        tSteps = tSteps[:-1]
        
        bestSolVal = float("Inf")
        bestSol = []
        
        while len(allSolutions):
            thisSolution = allSolutions.pop()
            thisXcurrent = thisSolution[0]
            aT = tSteps[thisSolution[1]]
            dT = tStepsDiff[thisSolution[1]]
                
            A = self.dynSys.getLinAFast(Xcurrent)
            B = self.dynSys.gEval(Xcurrent)
            uMin, uMax = self.dynSys.inputCstr.getCstr(aT)
            uNom = self.refTraj.uref(aT)

            allU = [uMin, uNom, uMax]
            
            Ai = np.linalg.inv(A)
            Aint = np.matrix(scipy.linalg.expm(A*dT))
            Aintm = np.linalg.inv(Aint)
            Bint = -Aint*Ai*(Aintm-np.identity(self.dynSys.nx))*B
            

            nextXhom = np.dot(Aint,thisXcurrent)

            allX = []
            thisU = np.matrix(np.zeros((self.dynSys.nu,1)))
            for aPerm in self.allPerms:
                for k in range(self.dynSys.nu):
                    thisU[k] = allU[aPerm[k]][k] #Storing in matrix faster?
                allX.append(nextXhom + np.dot(Bint, thisU))

            nextSolNum = thisSolution[1]+1
            
            #if nextSolNum == len(tSteps):
                #This was the final step so check sol
            lyapValDiscount = self.lyapReg.getCost(tInit, Xcurrent)*np.exp(-self.lyapReg.convergenceRate*(aT+dT-tInit))
            lyapValCurrent = self.lyapReg.getCost(aT+dT, np.hstack(allX))
            lyapValCurrentCheck = lyapValCurrent < lyapValDiscount
            lyapValCurrent = lyapValCurrent/lyapValDiscount
            
            checkList = list(self.lyapReg.getEllip(aT+dT)[0:-1]) + [lyapValDiscount]
            
            for k,check  in enumerate(lyapValCurrentCheck):
                if lyapValCurrent[k] < bestSolVal:
                    bestSolVal = lyapValCurrent[k]
                    bestSol = [allX[k], nextSolNum, thisSolution[2]+[self.allPerms[k]]]
                if check:
                    print ("tested {0}".format(thisSolution[2]+self.allPerms[k]))
                    thisCheck, thisX, thisT = self.realCheck(Xcurrent, thisSolution[2]+self.allPerms[k], tStepsAll[:thisSolution[1]+1], *checkList)
                    if thisCheck:
                        return True, thisX, thisT, self.allPerms[k]
                    else:
                        allSolutions.append([allX[k], nextSolNum, thisSolution[2]+[self.allPerms[k]]])
                elif nextSolNum < interSteps-1:
                    allSolutions.append([allX[k], nextSolNum, thisSolution[2]+[self.allPerms[k]]])
                else:
                    print("discarded")
                    print(allX[k])
                    print(thisSolution[2]+self.allPerms[k])
                    pass
                
                        
                        
            #else:
            #    for aX, aPerm in zip(allX, self.allPerms):
            #        allSolutions.append([aX, nextSolNum, thisSolution[2]+[aPerm]])
                
        return False, np.zeros(0), np.zeros(0), [] #No suitable solution could be found
    ############################################
    def realCheck(self, X, aPerm, tSteps, xFinal, PFinal, maxLyapVal):
        #Part of untested part
        #Perform an actual integration of the full non-linear system
        Xcurrent = X.squeeze()
        allX = np.zeros((0,self.dynSys.nx))
        allT = np.zeros(0,)
        for k in range(len(tSteps)-1):
            t0 = tSteps[k]
            t1 = tSteps[k+1]
            thisT = np.linspace(t0,t1,50) 
            fInt = lambda x, t: self.dynSys(x, self.dynSys.inputCstr.getThisU( t, aPerm[k] ))
            
            thisX = scipy.integrate.odeint(fInt, Xcurrent, thisT)
            
            allX = np.vstack((allX, thisX))
            allT = np.hstack((allT, thisT))
            Xcurrent = thisX[-1, :].squeeze()
            
        Xcurrent.resize((self.dynSys.nx,1))
        check = ndot((Xcurrent-xFinal).T, PFinal, (Xcurrent-xFinal)) < maxLyapVal
        
        return check, allX, allT
        
        
            
            
            
        
        
        
    ############################################
    
    
    
    
        
        





        
