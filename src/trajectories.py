from coreUtils import *
from interpolators import *
import plotUtils as pu
import lyapunovShapes as lyap
import dynamicSystems as ds

from subprocess import call
import os

def prepOMPLTraj2(dynSys, basicTraj, **kwargs):
    
    opts={'dT':0.01, 'toFile':None, 'zoneFac':1., 'gain':0.001, 'intType':'LQR', 'gainVel':0.0, 'velInd':np.arange(dynSys.nx//2, dynSys.nx, 1, dtype=np.int_)}
    
    opts.update(kwargs)
    
    dynSys.innerZone *= opts['zoneFac']
    
    #Get a lqr controller
    allK = []
    allP = []
    thisLyap = lyap.lqrShapeGen(dynSys, reshape=False)
    for aT in basicTraj.t:
        P,_ = thisLyap(basicTraj.xref(aT))
        allK.append(np.copy(opts['gain']*thisLyap.lastK))
        allK[-1][:,opts['velInd']] = opts['gainVel']*allK[-1][:,opts['velInd']] 
        allP.append(np.copy(P))
        
    allT = np.array(basicTraj.t[0])
    allX = np.zeros((dynSys.nx, 1))
    allXd = np.zeros((dynSys.nx, 1))
    allU = np.zeros((dynSys.nu, 1))
    
    allX[:,0] = basicTraj.xref(basicTraj.t[0]).squeeze()
    allU[:,0] = basicTraj.uref(basicTraj.t[0]).squeeze()
    
    indCurr = 0
    
    for k in range(basicTraj.t.size-1):
        thisT = np.linspace(basicTraj.t[k], basicTraj.t[k+1], int((basicTraj.t[k+1]-basicTraj.t[k])/opts['dT'])+1, endpoint=True)
        
        if opts['intType'] == 'LQR':
            fInt = lambda thisX, thisT: dynSys( thisX.reshape((dynSys.nx,1)), basicTraj.uref(thisT) - np.dot(allK[k], replaceSo2(thisX.reshape((dynSys.nx,1))-basicTraj.xref(thisT), dynSys.so2Dims) )).squeeze()
        else:
            fInt = lambda thisX, thisT: dynSys.getBestF(allP[k], basicTraj.xref(thisT), thisX.reshape((dynSys.nx,1)), mode="OR", u0=basicTraj.uref(thisT), t=thisT).squeeze()
        
        
        intX = scipy.integrate.odeint(fInt, allX[:,indCurr].squeeze(), thisT)
        intX = intX.T
        
        #Get the other values
        refX = basicTraj.xref(thisT)
        intU = dynSys.inputCstr(basicTraj.uref(thisT) - np.dot(allK[k], replaceSo2(intX-refX, dynSys.so2Dims)), t=thisT)
        
        intXd = dynSys(intX, intU)
        
        allT = np.hstack((allT, thisT[1:]))
        allX = np.hstack((allX, intX[:,1:]))
        allXd = np.hstack((allXd, intXd[:,1:]))
        allU = np.hstack((allU, intU[:,1:]))
        indCurr = allT.size-1
        
        
    #Treat so2
    if not(dynSys.so2Dims is None) and False:
        allX[dynSys.so2Dims,:] = toSo2(allX[dynSys.so2Dims,:])
    
    if not(opts['toFile'] is None):
        with open(opts['toFile']+"/X.txt", "w") as f:
            np.savetxt(f, allX.T)
        with open(opts['toFile']+"/Xd.txt", "w") as f:
            np.savetxt(f, allXd.T)
        with open(opts['toFile']+"/U.txt", "w") as f:
            np.savetxt(f, allU.T)
        with open(opts['toFile']+"/T.txt", "w") as f:
            np.savetxt(f, allT.squeeze())
            
    dynSys.innerZone *= (1./opts['zoneFac'])
    
    return allT.squeeze(), allX, allXd, allU    

def prepOMPLTraj(dynSys, basicTraj, dT = 0.01, toFile=None):
    
    zoneFac = 1.
    
    thisTSteps = basicTraj.t
    thisLyap = lyap.lqrShapeGen(dynSys)
    #Get a lqr controller
    allK = []
    allP = []
    for aT in thisTSteps:
        P,_ = thisLyap(basicTraj.xref(aT))
        allK.append(np.copy(.0*thisLyap.lastK))
        allP.append(np.copy(P))
    allT = np.arange(thisTSteps[0], thisTSteps[-1], dT)
    allT = np.hstack((allT, thisTSteps[-1]))
    #Get a smooth trajectory with corrected inputs
    
    allX = np.zeros((allT.size, dynSys.nx))
    allXd = np.zeros((allT.size, dynSys.nx))
    allU = np.zeros((allT.size, dynSys.nu))
    
    allX[0,:] = basicTraj.xref(0.0).squeeze()
    allU[0,:] = basicTraj.uref(0.0).squeeze()
    
    #fInt = lambda thisX, thisT: dynSys( thisX.reshape((dynSys.nx,1)), basicTraj.uref(thisT)).squeeze()
    #intAllX = scipy.integrate.odeint(fInt, allX[0,:].squeeze(), allT)
    
    dynSys.innerZone *= zoneFac
    
    for k in range(1,allT.size):
        ##Get the integration function
        #(self, x, u, prec = 64, doRestrain=True)
        
        indK = np.maximum(0,np.searchsorted(thisTSteps, allT[k-1])-1)
        
        #fInt = lambda thisX, thisT: dynSys( thisX.reshape((dynSys.nx,1)), basicTraj.uref(thisT) - np.dot(allK[indK], replaceSo2(thisX.reshape((dynSys.nx,1))-basicTraj.xref(thisT), dynSys.so2Dims) )).squeeze()
        fInt = lambda thisX, thisT: dynSys( thisX.reshape((dynSys.nx,1)), basicTraj.uref(thisT) - np.dot(allK[indK], thisX.reshape((dynSys.nx,1))-basicTraj.xref(thisT) )).squeeze()
        #getBestF(self, Pt, x0, x, mode='OO', u0 = None, t=0.):
        #fInt = lambda thisX, thisT: dynSys.getBestF(allP[indK], basicTraj.xref(thisT), thisX.reshape((dynSys.nx,1)), mode="OR", u0=basicTraj.uref(thisT), t=thisT).squeeze()
        
        intX = (scipy.integrate.odeint(fInt, allX[k-1,:].squeeze(), allT[[k-1,k]])[1,:]).reshape((dynSys.nx,1))
        
        intU = basicTraj.uref(allT[k]) - np.dot(allK[indK], replaceSo2(intX-basicTraj.xref(allT[k]), dynSys.so2Dims))
        
        allX[k,:] = intX.squeeze()
        allU[k,:] = dynSys.inputCstr(intU, t=allT[k]).squeeze()
        allXd[k,:] = dynSys(intX, dynSys.inputCstr(intU, t=allT[k])).squeeze()
    
    #Treat so2
    if not(dynSys.so2Dims is None) and False:
        allX[dynSys.so2Dims,:] = toSo2(allX[dynSys.so2Dims,:])
    
    if not(toFile is None):
        with open(toFile+"/X.txt", "w") as f:
            np.savetxt(f, allX.T)
        with open(toFile+"/Xd.txt", "w") as f:
            np.savetxt(f, allXd.T)
        with open(toFile+"/U.txt", "w") as f:
            np.savetxt(f, allU.T)
        with open(toFile+"/T.txt", "w") as f:
            np.savetxt(f, allT.squeeze())
            
    dynSys.innerZone *= (1./zoneFac)
    
    return allT.squeeze(), allX.T, allXd.T, allU.T 
        
    
    
    


class analyticTraj():
    def __init__(self, nx_, nu_, xref_, uref_, t, xdref_=None, tMin=-np.inf, tMax=np.inf):
        
        #No test is performed on tmin/tmax are only here for info
        
        self.t = t
        
        self.nx = nx_
        self.nu = nu_
        
        self.tMin = tMin
        self.tMax = tMax
        
        if isinstance(xref_, (list, tuple)):
            xref = sMz( len(xref_),1 )
            for k in range( len(xref_) ):
                xref[k] = eval(xref_[k], locals(), sympy.__dict__)
        else:
            xref = eval(str(xref_), locals(), sympy.__dict__)#dp(xref)
        #lamdify
        xreft = sympy.lambdify(t, xref, 'numpy')
        self.xref = lambda t: xreft(np.array(t).squeeze()).reshape((self.nx,-1))
        
        if xdref_ is None:
            xdref = sMz( len(xref_), 1 )
            for k in range( len(xref_) ):
                xdref[k] = sympy.diff(xref_[k], t)
        else:
            if isinstance(xdref_, (list, tuple)):
                xdref = sMz( len(xdref_), 1 )
                for k in range( len(xdref_) ):
                    xdref[k] = eval(xdref_[k], locals(), sympy.__dict__)
            else:
                xdref = eval(str(xdref_), locals(), sympy.__dict__)#dp(xdref)
        #lamdify
        xdreft = sympy.lambdify(t, xdref, 'numpy')
        self.xdref = lambda t: xdreft(np.array(t).squeeze()).reshape((self.nx,-1))
        
        if isinstance(uref_, (list, tuple)):
            uref = sMz( len(uref_), 1 )
            for k in range( len(uref_) ):
                uref[k] = eval(uref_[k], locals(), sympy.__dict__)
        else:
            uref = eval(str(uref_), locals(), sympy.__dict__)#dp(uref)
        #lamdify
        ureft = sympy.lambdify(t, uref, 'numpy')
        self.uref = lambda t: ureft(t).squeeze()
        self.uref = lambda t: ureft(np.array(t).squeeze()).reshape((self.nu,-1))
        
    
    ###############
    def callOld(self, t, prec=64):
        #thisL = [self.t, t]
        #return self.xref.n(prec, thisL), self.xdref.n(prec, thisL), self.uref.n(prec, thisL)
        thisL = {self.t: t}
        return np.array(self.xref.evalf(prec, subs=thisL)).astype(np.float_), np.array(self.xdref.evalf(prec, subs=thisL)).astype(np.float_), np.array(self.uref.evalf(prec, subs=thisL)).astype(np.float_)
    ###############
    def __call__(self, t):
        return self.xref(t), self.xdref(t), self.uref(t)
    ###############
    def getPos(self, t):
        return self.xref(t)
        
##############################

class interpTraj():
    def __init__(self, x, u, t, xd=None, thisInterpolatorX = scipy.interpolate.PchipInterpolator, thisInterpolatorXd = None, thisInterpolatorU = None):
        
        if thisInterpolatorXd is None:
            thisInterpolatorXd=thisInterpolatorX
        if thisInterpolatorU is None:
            thisInterpolatorU=thisInterpolatorX
        
        self.t = t
        
        self.tMin = t[0]
        self.tMax = t[-1]
        
        self.xrefI = []
        for k in range(x.shape[0]):
            self.xrefI.append( thisInterpolatorX(t, x[k,:]) )
        
        self.xdrefI = []
        if xd is None:
            for k in range(x.shape[0]):
                self.xdrefI.append( self.xrefI[k].derivative() )
        else:
            for k in range(x.shape[0]):
                self.xdrefI.append( thisInterpolatorXd(t, xd[k,:]) )
        
        self.urefI = []
        for k in range(u.shape[0]):
            self.urefI.append( thisInterpolatorU(t, u[k,:]) )
            
        self.nx = len(self.xrefI)
        self.nu = len(self.urefI)
        
        self.xref = lambda t:self(t)[0] #This is very slow: #TBD
        self.xdref = lambda t:self(t)[1] #This is very slow: #TBD
        self.uref = lambda t:self(t)[2] #This is very slow: #TBD
        
        
    ###############
    
    def __call__(self, t):
        
        t = np.array(t)
        t.resize((t.size,))
        
        t[t<self.t[0]] = self.t[0]
        t[t>self.t[-1]] = self.t[-1]
        
        thisX = np.zeros((self.nx,t.size))
        thisXd = np.zeros((self.nx,t.size))
        thisU = np.zeros((self.nu,t.size))
        
        for k in range(self.nx):
            thisX[k,:] = self.xrefI[k](t)
            thisXd[k,:] = self.xdrefI[k](t)
        for k in range(self.nu):
            thisU[k,:] = self.urefI[k](t)
        
        return thisX, thisXd, thisU
###################################################################################################################
class OMPLtrajectory(interpTraj):
    def __init__(self, nx, nu, pathtofile, thisInterpolatorXXd=scipy.interpolate.Akima1DInterpolator ):#lambda x,y:scipy.interpolate.interp1d(x,y,kind='linear')):
        
        #Try loading
        allData = np.genfromtxt(pathtofile, dtype=np.float_)
        
        assert(allData.shape[1]==nx+nu+1)
        
        allData[0:-1,nx:nx+nu] = allData[1:,nx:nx+nu]
        allData = allData.T
        
        T=allData[-1,:].squeeze()
        T=np.cumsum(T)
        
        x = allData[:nx,:]
        xd = np.divide(np.diff(x, axis=1), np.diff(T))
        xd = np.hstack((xd, xd[:,[-1]]))
        
        super(OMPLtrajectory, self).__init__(x,allData[nx:nx+nu,:], T, xd=xd, thisInterpolatorX = thisInterpolatorXXd, thisInterpolatorXd = thisInterpolatorXXd, thisInterpolatorU = leftNeighboor)
###################################################################################################################
class OMPLtrajectory2():
    def __init__(self, dynSys, allXorPath=None, allU=None, allT=None, baseName="ompl2Traj"):
        
        self.baseName = baseName
        self.dynSys = dynSys
        
        if isinstance(allXorPath, np.ndarray):
            
            self.X = np.copy(allXorPath)
            self.U = np.copy(allU)
            self.t = np.copy(allT)
            
            self.tMin = self.t[0]
            self.tMax = self.t[-1]
            
            #self.xref = scipy.interpolate.interp1d(allT, allX, fill_value="extrapolate")
            #self.uref = scipy.interpolate.interp1d(allT, allU, fill_value="extrapolate")
            
            self.xref = scipy.interpolate.PchipInterpolator(self.t, self.X, axis=1)
            self.uref = scipy.interpolate.PchipInterpolator(self.t, self.U, axis=1)#
            
            
            self.xdref = lambda x,u,t=0.: self.dynSys(x, u, doRestrain=True, t=t)
        else:
            self.fromPath(allXorPath)
    
    def fromPath(self, aPath):
        self.X = np.load(os.path.join(aPath, self.baseName+"_X.npy"))
        self.U = np.load(os.path.join(aPath, self.baseName+"_U.npy"))
        self.t = np.load(os.path.join(aPath, self.baseName+"_T.npy"))
        
        self.tMin = self.t[0]
        self.tMax = self.t[-1]
        
        #self.xref = scipy.interpolate.interp1d(allT, allX, fill_value="extrapolate")
        #self.uref = scipy.interpolate.interp1d(allT, allU, fill_value="extrapolate")
        
        self.xref = scipy.interpolate.PchipInterpolator(self.t, self.X, axis=1)
        self.uref = scipy.interpolate.PchipInterpolator(self.t, self.U, axis=1)#
        
        
        self.xdref = lambda x,u,t=0.: self.dynSys(x, u, doRestrain=True, t=t)
    
    def saveToPath(self, aPath):
        call("mkdir -p {0}".format(aPath), shell=True)
        np.save( os.path.join(aPath, self.baseName+"_X"), self.X)
        np.save( os.path.join(aPath, self.baseName+"_U"), self.U)
        np.save( os.path.join(aPath, self.baseName+"_T"), self.t)
    
    def __call__(self, t):
        t= np.maximum(np.minimum(t, self.tMax), self.tMin); 
        t = np.array(t)
        allX = self.xref(t)
        allU = self.dynSys.inputCstr(self.uref(t))
        allXd = self.dynSys(allX, allU)
        
        return allX, allXd, allU
                    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    