## Example stabilising the inverted pendulum (pendulum in instable, upright position)
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

excludeInner=0.05 #The inner ellipsoid in which convergence is not verified; Convergence is proved for \all x: excludeInner <= x.T.P.x <= 1.; If set to none There exists no inner zone
ellipInterpMode = 2
mtlbPath='./matlab/Pendulum/'

##Import prob
from invertedPendulum import *

t = sympy.symbols('t')
xrefs = sMa([np.pi,0.0]); 
xdrefs = sMa([0.0,0.0])
urefs = sMa([[0]])

#Trajectories
refTraj = traj.analyticTraj(nx, nu, xrefs, urefs, t, xdrefs)

#Get a LQR controller as shape generator 
Q = np.diag([1,1])
R = np.array([[5.]])
shapeGenStd = lyapS.lqrShapeGen(dynSys, reshape=False, Q=Q, R=R)
shapeGen = shapeGenStd

#How to calculate the conv
relaxationClass = lyapPar.NCrelax(nx, mode='PL', excludeInner=excludeInner)
#Some shortcuts
iMx, iMX, iMZt, iMA, iMX2z, numVars = relaxationClass.iMx, relaxationClass.iMX, relaxationClass.iMZt, relaxationClass.iMA, relaxationClass.iMX2z, relaxationClass.numVars

#Get the position to search a surrounding region of stabilizability
xFinal = refTraj(2.)[0]
#There are two modes:
shapeMode = 4
#1 Use the LQR shape
#2 Use the shape obtained beforehand using the drake toolbox
#>3 Predefined showcase

if shapeMode==1:
    if ellipInterpMode in [0,2]:
        initialShape = shapeGenStd(xFinal, lastShape=[np.array([[2.91, 1.44],[1.44,1.05]]), 1.])
    elif ellipInterpMode in [1,3]:
        initialShape = shapeGenStd(xFinal, lastShape=[np.array([[2.91, 1.44],[1.44,1.05]]), 1.])
    else:
        assert 0
elif shapeMode==2:
    Pmtlb = np.loadtxt(mtlbPath+'Pout.txt', delimiter=',')
    shapeGen = lyapS.simpleGen()
    initialShape = [Pmtlb,1.]
elif shapeMode==3:
    shapeGen = lyapS.simpleGen()
    initialShape = [Id(2)*20.,1.]
elif shapeMode==4:
    shapeGen = lyapS.simpleGen()
    initialShape = [na([[2.5,.7],[.7,1]])*20.,1.] #[na([[4.,.9],[.9,.7]])*20.,1.]# [na([[2.5,.7],[.7,1]])*20.,1.]
else:
    assert 0, "shapeMode unknown"

#Save for matlab polynomial controller
try:
    np.savetxt(mtlbPath+"K.txt", shapeGenStd.lastK)
except:
    pass  
fff, BBB, AAA, BGG = dynSys.getTaylorS3Fast(xFinal)
np.savetxt(mtlbPath+"f.txt", fff)
np.savetxt(mtlbPath+"B.txt", BBB)
np.savetxt(mtlbPath+"Atilde.txt", AAA)
for k, Btilde in enumerate(BGG):
     np.savetxt(mtlbPath+"taylorB_{0}.txt".format(k), Btilde)

tSteps = np.linspace(0.,6.,3)#Dummy time vector needed since algorithm expects a reference trajectory, not just a point

lyapReg = lyapPar.ellipsoidalLyap(dynSys, refTraj, tSteps, shapeGen, initialShape, relaxationClass, mode=relaxationClass.mode, excludeInner=excludeInner ) 

#get the "personalized" functions
with open("execUtils.py") as f:
    code = compile(f.read(), "execUtils.py", 'exec')
    exec(code)

#Set refereneces to desired functions
thisCalcFun = NCrelaxCalc#exactSol#NCrelaxCalc
lyapReg.calculateFunction = lambda input: convPL(input, calcFun=thisCalcFun)
relaxationClass.getCstr(getBounds, RLTCalculator=crtGeneralRLTPolyMinSortedAllCy)

#params
lyapReg.convergenceLim = 0.0001#When to stop the dichotomic serach
lyapReg.convergenceRate = 0.0001#Minimally demanded exponential convergence rate
lyapReg.epsDeadZone = 0.0#when positive a zone around the hyperplanes is ignored during the minimization
                         #If overlapping a maximal dwell-time can be established (So a minimal tolerable switching rate on the sliding plane can be obtained)
if shapeMode == 3:
    # P = Identity poses numerical problems at theta_dot = 0
    lyapReg.epsDeadZone = 0.01

if shapeMode == 4:
    lyapReg.convergenceLim = 0.01  # 0.0001#When to stop the dichotomic serach
    lyapReg.convergenceRate = 0.5  # Minimally demanded exponential convergence rate
    lyapReg.epsDeadZone = 0.0  # when positive a zone around the hyperplanes is ignored during the minimization

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
    
    with open("invPendUpright.pickle","wb") as fpickle:
        pickle.dump(lyapReg.allShapes, fpickle)

    # Replace all ellipsoids with the final ellipsoid
    # (Trick needed due to fake time vector)
    finalShape = deepcopy(lyapReg.allShapes[0])

    for k in range(len(lyapReg.allShapes)):
        lyapReg.allShapes[k] = deepcopy(finalShape)
    
    np.savetxt(mtlbPath+'Prelax.txt', lyapReg.allShapes[0][0]*lyapReg.allShapes[0][1])

    # Truncated taylor dynamics
    # Epsilon-close quadratic control
    plotFunnel(lyapReg, opts={"whichDyn":"QPP",'nX':50,'nY':50,'interSteps':5,'regEpsOpt':2.,'fullQOut':True,'equalScale':False})
    
    # Optimal sliding-mode control, switching based on hyperplanes
    plotFunnel(lyapReg,opts={"whichDyn":"CPL",'nX':50,'nY':50,'interSteps':5,'regEpsOpt':2.,'fullQOut':False,'equalScale':False})
    
    #Real nonlinear dynamics
    # Epsilon-close quadratic control
    plotFunnel(lyapReg,opts={"whichDyn":"QOO",'nX':50,'nY':50,'interSteps':5,'regEpsOpt':2.,'fullQOut':False,'equalScale':False})
    
    # Optimal sliding-mode control, switching based on hyperplanes
    plotFunnel(lyapReg,opts={"whichDyn":"COR",'nX':50,'nY':50,'interSteps':5,'regEpsOpt':2.,'fullQOut':False,'equalScale':False})
    

    #Plot qp convergence for the tSpan
    xFinal, PtFinal = lyapReg.getEllip(tSteps[0])[:2]
    initPoints = getV(PtFinal, 11, endPoint = False)
    initPoints += xFinal
    tSim = np.linspace(tSteps[0], tSteps[0]+1,5)
    # Slidingmode control
    replayFunnel(lyapReg,np.linspace(0.0,1.0,11),Ninit=11,initPoints=initPoints,dynamicsMode='CPR',mode='seq')
    # QP control
    replayFunnel(lyapReg,np.linspace(0.0,1.0,11),Ninit=11,initPoints=initPoints,dynamicsMode='QPP',mode='seq', otherOptsIn={'regEpsOpt':.5})
    
    plt.show()