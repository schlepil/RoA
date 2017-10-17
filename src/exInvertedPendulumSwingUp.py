## Example performing a swing-up motion for the inverted pendulum
# (Taking the pendulum from the stable, hanging position to the instable, upright position)
# Plots the resulting region of stabilizability over time
# and streamlines for each of the checked time-points for original and approximated dynamics

from coreUtils import *
from plotUtils import *
import trajectories as traj
import dynamicSystems as ds
import lyapunovShapes as lyapS
import lyapunov_Parallel as lyapPar

import time

from invertedPendulum import *#Get the dynamic systems

solvers.options['show_progress'] = False

excludeInner=None#The inner ellipsoid in which convergence is not verified; Convergence is proved for \all x: excludeInner <= x.T.P.x <= 1.
ellipInterpMode = 6#Linear interpolation of cholesky factorizations

doCalc = 1#Whether plotting is to be done
doPlot = 1#Whether the calculation should be performed

#Get the trajectories
basicTraj = traj.OMPLtrajectory(2,1,"../trajectories/invPendSwingUp.txt")

dynSys.inputCstr = ds.boxCstr(nu, -2.6, 2.6)
newT, newX, newXd, newU = traj.prepOMPLTraj2(dynSys, basicTraj, dT = 0.01, toFile=None)


#Get a new trajectory based on the pretreated values
refTraj = traj.OMPLtrajectory2(dynSys, newX, newU, newT)

#Reset the actuator limits to have a margin
inputCstr = ds.boxCstr(nu, -3, 3)
dynSys.inputCstr=inputCstr

#Use the standard finite horizon lqr controller
Q = np.diag([1., 1.4])
R = np.array([[0.005]])
shapeGenStd = lyapS.lqrShapeGen(dynSys, reshape=False, Q=Q, R=R)

shapeGen = lyapS.timeVaryingLqrShapegen(dynSys,Q=Q, R=R, reshape=False)
shapeGen.limK = True
shapeGen.restart = True
shapeGen.refTraj = refTraj
shapeGen.ctrlSafeFac=1.
shapeGen.retAll = True
shapeGen.interSteps = 10

#How to calculate the conv
relaxationClass = lyapPar.NCrelax(nx, mode='PL', excludeInner=excludeInner)
#Some shortcuts
iMx, iMX, iMZt, iMA, iMX2z, numVars = relaxationClass.iMx, relaxationClass.iMX, relaxationClass.iMZt, relaxationClass.iMA, relaxationClass.iMX2z, relaxationClass.numVars

#Get the position to reach
xFinal = np.zeros((2,1))
#Target zone
initialShape = shapeGenStd(xFinal, lastShape=[np.eye(nx), np.sqrt(10000.)])
#If ellipInterpMode == 6 
initialShape[0] *= initialShape[1]
initialShape = [ [[initialShape[0],initialShape[0]], [np.zeros((nx,nx)),np.zeros((nx,nx))], [basicTraj.t[-1], basicTraj.t[-1]+0.001]], 1. ]
initialShape[1] = 1.

#Check at each output point of ompl; This way everytime the reference control input change the convergence is checked
#If subSteps is set to positive value, additional points will be checked 
tSteps = basicTraj.t
lyapReg = lyapPar.ellipsoidalLyap(dynSys, refTraj, tSteps, shapeGen, initialShape, relaxationClass, mode=relaxationClass.mode, excludeInner=excludeInner ) 

#get the "personnalized" functions
with open("execUtils.py") as f:
    code = compile(f.read(), "execUtils.py", 'exec')
    exec(code)

#Now do the specialized stuff
thisCalcFun = NCrelaxCalc#exactSol#NCrelaxCalc
lyapReg.calculateFunction = lambda input: convPL(input, calcFun=thisCalcFun)
relaxationClass.getCstr(getBounds, RLTCalculator=crtGeneralRLTPolyMinSortedAllCy)

#params
lyapReg.convergenceLim = 0.01#Stop criterion for dichotomic search
lyapReg.convergenceRate = 2.#Guaranteed exponential convergence exponent
lyapReg.epsDeadZone = 0.0#Overlap or exclude a neighborhood of the sliding planes
lyapReg.ellipInterp = interpList[ellipInterpMode]

#Till here the code is also executed by the childprocesses
if __name__ == '__main__':
    
    #Get the other processes
    allProcesses = []
    inputQueue = Queue()
    outputQueue = Queue()
    
    for k in range(cpuCount):
        allProcesses.append(Process(target=convPLParallel, args=(inputQueue, outputQueue, lyapReg.calculateFunction)))
        allProcesses[-1].start()
    
    #lyapReg.calculateFunction = convPL(inputQueue, outputQueue)
    if doCalc:
        lyapReg.getLargestFunnel(inputQueue, outputQueue)
        with open("../results/invPendSwingUp.pickle","wb") as fpickle:
            pickle.dump(lyapReg.allShapes, fpickle)
    else:
        with open("../results/invPendSwingUp.pickle","rb") as fpickle:
            lyapReg.allShapes = pickle.load(fpickle)
        
    for k,[a,b] in enumerate(lyapReg.allShapes):
        try: 
            print( "{0} : {1}".format(k, b) )
            print(np.linalg.det(a*b))
        except:
            _,P,_ = lyapReg.getEllip(tSteps[k])
            print(P)
            print(np.linalg.det(P))
    
    if doPlot:
        #plt.show()
        replayOpts = {'doEachStream':True, 'doEachConv':False, 'doEachDV':False, 'diffSteps':1., "whichDyn":"QPP",'nX':20, 'nY':20}
        plotFunnel(lyapReg, opts=replayOpts)
        plt.show()
    for aProc in allProcesses:
        aProc.terminate()
    