## Example stabilising the acrobot in upright instable position
# Plots the resulting region of stabilizability
# and streamlines for original and approximated dynamics
from copy import deepcopy

from coreUtils import *
from plotUtils import *
import trajectories as traj
import dynamicSystems as ds
import lyapunovShapes as lyapS
import lyapunov_Parallel as lyapPar

import time

solvers.options['show_progress'] = False

#Get the dynamical system definitions
from acrobot import *

#Define the input bounds
inputCstr = ds.boxCstr(nu, -20,20)
dynSys.inputCstr = inputCstr

#Options
excludeInner=None
ellipInterpMode = 2
mtlbPath='./matlab/Acrobot/'

#Get a symbolic input that corresponds to the upright position
#The algorithm always except a "trajectory" so this is a workaround
Tlim = [0.0,0.75]
Nsteps = 2
tSteps = np.linspace(Tlim[0], Tlim[1], Nsteps)
t = sympy.symbols('t')

urefSym = sMa([0.])
xrefSym = sMa([np.pi,0.0,0.0,0.0])
xdrefSym = sMa([0.0,0.0,0.0,0.0])

ySym, uSym = dynSys.xTrans, dynSys.u

#def __init__(self, nx_, nu_, xref_, uref_, t, xdref_=None, tMin=-np.inf, tMax=np.inf):
refTraj = traj.analyticTraj(nx, nu, xrefSym, urefSym, t, xdrefSym)

#Get a shape generator
#In this case the results obtained with regular LQR are pretty good
Q =np.diag([10,10,1,1])
R=np.array([[.01]])
shapeGen = lyapS.lqrShapeGen(dynSys, reshape=False, Q=Q, R=R)

#How to calculate the convexification
relaxationClass = lyapPar.NCrelax(nx, mode='PL', excludeInner=excludeInner)
#Some shortcuts; Partly used within execUtils
iMx, iMX, iMZt, iMA, iMX2z, numVars = relaxationClass.iMx, relaxationClass.iMX, relaxationClass.iMZt, relaxationClass.iMA, relaxationClass.iMX2z, relaxationClass.numVars

#Get the "target"
xFinal = refTraj(tSteps[-1])[0]
#There are two modes:
#1 Use the LQR shape
#2 Use the shape obtained beforehand using the drake toolbox  
if 0:
    #The final ellipsoid size does not play a role because d/dt P is not taken into account
    initialShape = shapeGen(xFinal, lastShape=[np.eye(nx), 20000.])
else:
    #Load the shape obtained by matlab/drake toolbox
    try:
        Pmtlb = np.loadtxt(mtlbPath+'Pout.txt', delimiter=',')
    except:
        Pmtlb = np.loadtxt(mtlbPath+'Pout.txt')+0.1*np.identity(4)
    shapeGen = lyapS.simpleGen()
    initialShape = [Pmtlb,1.]

#Save for matlab polynomial controller to eventually compare
try:
    np.savetxt(mtlbPath+"KAcro.txt", shapeGen.lastK)
except:
    pass
fff, BBB, AAA, BGG = dynSys.getTaylorS3Fast(xFinal)
np.savetxt(mtlbPath+"fAcro.txt", fff)
np.savetxt(mtlbPath+"BAcro.txt", BBB)
np.savetxt(mtlbPath+"AAcrotilde.txt", AAA)
for k, Btilde in enumerate(BGG):
     np.savetxt(mtlbPath+"taylorAcroB_{0}.txt".format(k), Btilde)
                                        
#Get the "funnel"-class
lyapReg = lyapPar.ellipsoidalLyap(dynSys, refTraj, tSteps, shapeGen, initialShape, relaxationClass, mode=relaxationClass.mode, excludeInner=excludeInner ) 
#get the "personalized" functions
with open("execUtils.py") as f:
    code = compile(f.read(), "execUtils.py", 'exec')
    exec(code)

#Set some calculation options
#At the current state of implementation there exist no alternatives to these functions
#but they will come to existence when testing polyhedral regions and second
#order-cone constraints
thisCalcFun = NCrelaxCalc#exactSol
lyapReg.calculateFunction = lambda input: convPL(input, calcFun=thisCalcFun)
relaxationClass.getCstr(getBounds, RLTCalculator=crtGeneralRLTPolyMinSortedAllCy)

#params defining the convergence
lyapReg.convergenceLim = 0.1#0.0001#When to stop the dichotomic serach
lyapReg.convergenceRate = 5e-2#Minimally demanded exponential convergence rate
lyapReg.epsDeadZone = 0.0#when positive a zone around the hyperplanes is ignored during the minimization
                         #If overlapping a maximal dwell-time can be established (So a minimal tolerable switching rate on the sliding plane can be obtained) 
lyapReg.ellipInterp = interpList[ellipInterpMode]#Which interpolation to use; Here standard without time-dependency

#"main program"
#Till here the steps are performed on all instances if calculation is parallelized
if __name__ == '__main__':
            
    #Get the other processes
    allProcesses = []
    inputQueue = Queue()
    outputQueue = Queue()
    
    if not glbDBG:
        for k in range(cpuCount):
            allProcesses.append(Process(target=convPLParallel, args=(inputQueue, outputQueue, lyapReg.calculateFunction)))
            allProcesses[-1].start()
    
    
    lyapReg.getLargestFunnel(inputQueue, outputQueue)
    with open("../results/acroUpright.pickle","wb") as fpickle:
        pickle.dump(lyapReg.allShapes, fpickle)
    
    np.savetxt(mtlbPath+'Prelax.txt', lyapReg.allShapes[0][0]*lyapReg.allShapes[0][1])
    
    for k,[a,b] in enumerate(lyapReg.allShapes): 
        print( "{0} : {1}".format(k, b) )
        print(a)
    
    #Done with principal calcs
    for aProc in allProcesses:
        aProc.terminate()
    #Plot the funnel; Streamlines are only meaningful for systems with a two dimensional
    #state-space
    replayOpts = {'doEachStream':False, 'doEachConv':False, 'doEachDV':False}
    for indI in range(4):
        for indK in range(indI+1,4):
            plotFunnel(lyapReg, opts=replayOpts, plotAx=[indI, indK])
    
    #Replace all ellipsoids with the final ellipsoid
    #(Trick needed due to fake time vector)
    finalShape = deepcopy(lyapReg.allShapes[0])
    
    for k in range(len(lyapReg.allShapes)):
        lyapReg.allShapes[k] = deepcopy(finalShape)
    

    XX0, PP0, _ = lyapReg.getEllip(0.0)
    
    print(np.linalg.eigh(PP0))
    print(np.linalg.det(PP0))
    #plt.show()
    
    initPoints = np.random.rand(4,10)-0.5
    initPoints = np.divide(initPoints, colWiseNorm(initPoints))
    initPoints = ndot( np.linalg.cholesky(np.linalg.inv(PP0)), initPoints )
    initPoints = XX0 + initPoints
    
    
    ax=plt.gca()
    ax.autoscale(1)
    ax.plot(initPoints[0,:], initPoints[1,:], '.k')
    Xinit = replayFunnel(lyapReg, initPoints=initPoints, dynamicsMode="CPR", mode="seq")
    Xinit = replayFunnel(lyapReg, initPoints=initPoints, dynamicsMode="QPP", mode="seq", otherOptsIn={'regEpsOpt':.1})


    plt.show()

    