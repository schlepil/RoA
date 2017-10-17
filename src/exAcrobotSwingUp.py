from coreUtils import *
from plotUtils import *
import trajectories as traj
import dynamicSystems as ds
import lyapunovShapes as lyapS
import lyapunov_Parallel as lyapPar

import time

solvers.options['show_progress'] = False    

from acrobot import *

#Options all
excludeInner=0.2 #The inner ellipsoid in which convergence is not verified; Convergence is proved for \all x: .2 <= x.T.P.x <= 1.
ellipInterpMode = 6
doPlot=1#Whether plotting is to be done
doCalc=1#Whether the calculation should be performed


#Trajectories
#Load interpolation Traj
try:
    #Try to load a saved interpolation
    basicTraj = traj.OMPLtrajectory(4,1,"../trajectories/acroSwingUp10.txt")
    refTraj = traj.OMPLtrajectory2(dynSys, "../trajectories/acroSwingUp10/")
except:
    #Do a finer interpolation
    #and save the result
    basicTraj = traj.OMPLtrajectory(4,1,"../trajectories/acroSwingUp10.txt")
    
    dynSys.inputCstr = ds.boxCstr(nu, -10.1, 10.1)
    newT, newX, newXd, newU = traj.prepOMPLTraj2(dynSys, basicTraj, dT = 0.001, toFile=None)
    
    
    #Get a new trajectory based on the pretreated values
    refTraj = traj.OMPLtrajectory2(dynSys, newX, newU, newT)
    refTraj.saveToPath("../trajectories/acroSwingUp10/")

inputCstr = ds.boxCstr(nu, -20., 20.)#Set the actuator limits
dynSys.inputCstr = inputCstr

#Get the shape generator
#In this case a modified version of finite horizon lqr controller
#that is modified such that input constraints are partially accounted for
shapeGenStd = lyapS.lqrShapeGen(dynSys, reshape=False, Q=np.diag([10,10,1,1]), R=np.array([[1.e-1]])) #For the final zone; Can also be imposed by hand

shapeGen = lyapS.timeVaryingLqrShapegen(dynSys,Q=np.diag([1.,1.,1.,1.]), R=np.array([[1.e-1]]), reshape=False)
shapeGen.restart = True
shapeGen.refTraj = refTraj
shapeGen.ctrlSafeFac=1.
shapeGen.retAll = True
shapeGen.interSteps = 10

#How to calculate the conv
relaxationClass = lyapPar.NCrelax(nx, mode='PL', excludeInner=excludeInner)
#Some shortcuts
iMx, iMX, iMZt, iMA, iMX2z, numVars = relaxationClass.iMx, relaxationClass.iMX, relaxationClass.iMZt, relaxationClass.iMA, relaxationClass.iMX2z, relaxationClass.numVars

#Check at each output point of ompl; This way everytime the reference control input change the convergence is checked
#If subSteps is set to positive value, additional points will be checked 
tSteps = basicTraj.t
#tSteps = tSteps[-20:]
#Get the actual end of the reference traj
xFinal = refTraj(basicTraj.t[-1])[0]
#Small target zone
initialShape = shapeGenStd(xFinal, lastShape=[np.eye(nx), 20.])
initialShape[0] *= initialShape[1]
#Get it into the expectde shpae
initialShape = [ [[initialShape[0],initialShape[0]], [np.zeros((nx,nx)),np.zeros((nx,nx))], [basicTraj.t[-1], basicTraj.t[-1]+0.001]], 1. ]
initialShape[1] = 1.

lyapReg = lyapPar.ellipsoidalLyap(dynSys, refTraj, tSteps, shapeGen, initialShape, relaxationClass, mode=relaxationClass.mode, excludeInner=excludeInner ) 

#Set the sweep
#This is an alternative to pure dichotomic search
#lyapReg.doSweep = np.linspace(.9,1.1,3)
#Set it to None for dichotomic search
lyapReg.doSweep = None

#get the "personalized" functions
with open("execUtils.py") as f:
    code = compile(f.read(), "execUtils.py", 'exec')
    exec(code)

#Set reference to the functions
thisCalcFun = NCrelaxCalc
lyapReg.calculateFunction = lambda input: convPL(input, calcFun=thisCalcFun)
relaxationClass.getCstr(getBounds, RLTCalculator=crtGeneralRLTPolyMinSortedAllCy)

#params
lyapReg.convergenceLim = 0.001#Stop criterion for dichotomic search
lyapReg.convergenceRate = 0.#Guaranteed exponential convergence exponent
lyapReg.epsDeadZone = 0.0#Overlap or exclude a neighborhood of the sliding planes
lyapReg.ellipInterp = interpList[ellipInterpMode]

#Till here the code is also executed by the childprocesses
if __name__ == '__main__':
    #Do the actual computation
    allProcesses = []
    inputQueue = Queue()
    outputQueue = Queue()
    
    for k in range(cpuCount):
        allProcesses.append(Process(target=convPLParallel, args=(inputQueue, outputQueue, lyapReg.calculateFunction)))
        allProcesses[-1].start()
    
    if doCalc:
        lyapReg.getLargestFunnel(inputQueue, outputQueue, subSteps=0)
        with open("../results/acroSwingUp10.pickle","wb") as fpickle:
            pickle.dump(lyapReg.allShapes, fpickle)
    else:
        with open("../results/acroSwingUp10.pickle","rb") as fpickle:
            lyapReg.allShapes = pickle.load(fpickle)
    
    for k,aT in enumerate(lyapReg.tSteps):
        _, thisP, _ = lyapReg.getEllip(aT) 
        print( "{0} \n {1}".format(k, thisP) )
        print(np.linalg.det(thisP))
    
    #Done with principal calcs
    for aProc in allProcesses:
        aProc.terminate()
    if doPlot:
        replayOpts = {'doEachStream':False, 'doEachConv':False, 'doEachDV':False, 'diffSteps':0.}
        for i in range(4):
            for j in range(i+1,4):
                plotFunnel(lyapReg, opts=replayOpts, plotAx = [i,j])
        plt.show()
    
    #Done