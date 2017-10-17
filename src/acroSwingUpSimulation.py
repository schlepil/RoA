from coreUtils import *
import plotUtils as pu
import trajectories as traj
import lyapunovShapes as lyapShape
import lyapunov_Parallel as lyapPar
import dynamicSystems as ds

from copy import deepcopy as dp

import matplotlib.animation as animation
import matplotlib as mpl
mpl.verbose.set_level('helpful')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#Get the system
from acrobot import *
inputCstr = ds.boxCstr(nu, -20., 20.)
dynSys.inputCstr = inputCstr
dynSys.innerZone = 1.e-2 #Smoothening zone
#Get the trajectory
try:
    basicTraj = traj.OMPLtrajectory(4,1,"/home/elfuius/ownCloud/thesis/RoA/ompl/ompl/build/acroSwingUp8.txt")
    refTraj = traj.OMPLtrajectory2(dynSys, "/home/elfuius/ownCloud/thesis/RoA/ompl/ompl/build/acroSwingUp8/")
except:
    basicTraj = traj.OMPLtrajectory(4,1,"/home/elfuius/ownCloud/thesis/RoA/ompl/ompl/build/acroSwingUp8.txt")
    
    #T = np.loadtxt("../input/acroT.txt", delimiter=',').reshape((-1,))
    #X = np.loadtxt("../input/acroX.txt", delimiter=',').reshape((4,T.size))
    #U = np.loadtxt("../input/acroU.txt", delimiter=',').reshape((1,T.size))
    #basicTraj = traj.interpTraj(X,U,T)
    
    dynSys.inputCstr = ds.boxCstr(nu, -9.1, 9.1)
    newT, newX, newXd, newU = traj.prepOMPLTraj2(dynSys, basicTraj, dT = 0.001, toFile=None)
    
    
    #Get a new trajectory based on the pretreated values
    refTraj = traj.OMPLtrajectory2(dynSys, newX, newU, newT)
    refTraj.saveToPath("/home/elfuius/ownCloud/thesis/RoA/ompl/ompl/build/acroSwingUp8/")


#Get the funnel structure
tSteps = basicTraj.t
shapeGen = lyapShape.simpleGen()#Dummy for replay
initialShape = []#Dummy for replay
relaxationClass = lyapPar.NCrelax(nx, mode='PL', excludeInner=None)
#Some shortcuts
iMx, iMX, iMZt, iMA, iMX2z, numVars = relaxationClass.iMx, relaxationClass.iMX, relaxationClass.iMZt, relaxationClass.iMA, relaxationClass.iMX2z, relaxationClass.numVars
lyapReg = lyapPar.ellipsoidalLyap(dynSys, refTraj, tSteps, shapeGen, initialShape, relaxationClass) 

#Funnel to be loaded
funnelStr = "tNLNewITightNoSafe_p2_1_1_1"
#And interpolation option (manually; currently not saved)
lyapReg.convergenceRate = 0.0
lyapReg.ellipInterp = interpList[6]#Corresponds to modified lqr

with open(funnelStr+'.pickle',"rb") as fpickle:
    lyapReg.allShapes = pickle.load(fpickle)

#Animation vars
dT = 1./30.#Delay between frames in ms <-> 30fps
listOfDesigns = []
#Disturbance
#It is possible to add a disturbace of the form
#qdd = f(q,qd)+g(q).u+[dist0;dist1]
#So the disturbance is equivalent to an acceleration offset  
dTdist = np.linspace(0,dT,4, endpoint=True)
disturbanceScale = np.array([[2.5],[1.0]])#*0.
#dynamics mode
#First char is system dynamics (O->nonlinear dynamics; P->Polynomial dynamics; L-> Linear dynamics)
#Second char is control in input dynamics (O-> nonlinear input dynamics and (discontinuous) convergence optimal control; R->nonlinear input and smoothened optimal control; P-> polynomial input dynamics and (discontinuous) convergence optimal control; L-> linear input dynamics and (discontinuous) convergence optimal control)  
dynMode = 'OR'

#Animation function
T = lyapReg.tSteps[0]
allT = [0.]
allV = [0.95]
allX = []
#Initial Pos
x0, P0, _ = lyapReg.getEllip(T) 
try:
    Xinit = np.random.rand(4,1)-0.5
    assert 0
except:
    Xinit=np.array([[1,1,1,1]]).T
Xinit = np.sqrt(allV[0])*np.divide(Xinit, colWiseNorm(Xinit))#Slightly inside the funnel; If disturbance is set to zero, the prefactor can be set to 1.
Xinit = ndot( np.linalg.cholesky(np.linalg.inv(P0)), Xinit )+x0.reshape((4,1))
allX.append(Xinit)

#fig, ax = plt.subplots(2,2)
#ax[0][0].set_title(r'Acrobot')
#ax[0][1].set_title(r'$\theta_0$/$\theta_1$')
#ax[1][0].set_title(r'$V(\boldsymbol x)$')
#ax[1][1].set_title(r'$\theta_0$/$\dot{\theta}_1$')

#Plot referenec traj
#tAll = np.linspace(tSteps[0], tSteps[-1], 10000)
#xAll = refTraj(tAll)[0]
#ax[0][1].plot(xAll[0,:], xAll[1,:], '-k')
#ax[1][1].plot(xAll[0,:], xAll[3,:], '-k') 
#ax[1][0].set_xlim( [tSteps[0], tSteps[-1]] )
#ax[1][0].set_ylim( [0.0, 1.2] )
#ax[1][0].autoscale(False)

#ax[0][0].set_xlim( [-3.5, 3.5] )
#ax[0][0].set_ylim( [-3.5, 3.5] )
#ax[0][0].autoscale(False)
#Auxilliary
dim = dynSys.nx
Ninit = 1

#def update(fig, listOfDesigns, allT, allV, allX):
    #get random disturbance
T=tSteps[0]
KK = 0
while(T<tSteps[-1]):
    
    fig, ax = plt.subplots(2,2)
    ax[0][0].set_title(r'Acrobot')
    ax[0][1].set_title(r'$\theta_0$/$\theta_1$')
    ax[1][0].set_title(r'$V_{s}(x)$')
    ax[1][1].set_title(r'$\theta_0$/$\dot{\theta}_1$')
    
    #Plot referenec traj
    tAll = np.linspace(tSteps[0], tSteps[-1], 10000)
    xAll = refTraj(tAll)[0]
    ax[0][1].plot(xAll[0,:], xAll[1,:], '-k')
    ax[1][1].plot(xAll[0,:], xAll[3,:], '-k') 
    ax[1][0].set_xlim( [tSteps[0], tSteps[-1]] )
    ax[1][0].set_ylim( [0.0, 1.2] )
    ax[1][0].autoscale(False)
    
    #ax[0][0].set_xlim( [-3.5, 3.5] )
    #ax[0][0].set_ylim( [-3.5, 3.5] )
    ax[0][0].set_aspect('equal')
    ax[0][0].set_xlim( [-3.5, 3.5] )
    ax[0][0].set_ylim( [-3.5, 3.5] )
    ax[0][0].autoscale(False)
    
    xLim=np.array(ax[0][1].get_xlim());xLim*=1.3
    yLim=np.array(ax[0][1].get_ylim());yLim*=1.3
    ax[0][1].set_xlim(xLim)
    ax[0][1].set_ylim(yLim)
    xLim=np.array(ax[1][1].get_xlim());xLim*=1.3
    yLim=np.array(ax[1][1].get_ylim());yLim*=1.3
    ax[1][1].set_xlim(xLim)
    ax[1][1].set_ylim(yLim)
    
    
    T=allT[-1]
    Xinit=allX[-1]
    thisDist = np.multiply(np.random.rand(2,dTdist.size)-.5, disturbanceScale)
    thisdTdist = T+dTdist
    #Interpolate dist
    distI = scipy.interpolate.interp1d(thisdTdist, thisDist, kind='nearest', assume_sorted = True, bounds_error=None, fill_value='extrapolate')#Piecewise constant error
    #Integration function
    fInt = lambda thisX, thisT: ( lyapReg.dynSys.getBestF(lyapReg.getEllip(thisT)[1], lyapReg.refTraj(thisT)[0], thisX.reshape((Ninit,dim)).T, mode=dynMode, u0 = lyapReg.refTraj(thisT)[2], t=thisT) + np.vstack((np.zeros((2,1)), distI([thisT])))).T.reshape((dim*Ninit,))    
    #Do the actual integration
    Xres, info = scipy.integrate.odeint(fInt, Xinit.squeeze(), [T, T+dT], full_output=True)
    print(info['message'])
    if not info['message']=='Integration successful.':
        dynSys.innerZone=1.e-1
        Xres, info = scipy.integrate.odeint(fInt, Xinit.squeeze(), [T, T+dT], full_output=True)
        print(info['message'])
        dynSys.innerZone=1.e-2
    Xinit = Xres[-1,:]
    Xinit.resize((4,1))
    T+=dT
    #Get current value
    allT.append(dp(T))
    allV.append( float(lyapReg.getCost(T, Xinit)) )
    allX.append(Xinit)
    #Delete all old drawings
    for aItem in listOfDesigns:
        del aItem
    #Do new plot
    _, Pc, _ = lyapReg.getEllip(T)
    xC = refTraj(T)[0]
    print(xC)
    #Cost value
    #listOfDesigns = []
    listOfDesigns += ax[1][0].plot(allT, allV, '-b', linewidth=2.)#Lines
    #Draw acrobot
    xAcro = getDrawPos(Xinit)
    listOfDesigns +=  ax[0][0].plot(xAcro[0,:], xAcro[1,:], '-k', linewidth=2.)
    listOfDesigns +=  ax[0][0].plot(xAcro[0,:2], xAcro[1,:2], '.k', markersize=10)
    #Draw the projected ellipsoids and the current position
    listOfDesigns.append(  pu.plotEllipse(ax[0][1], xC, Pc, 1., plotAx=[0,1], color=[0,0,1.,1.], faceAlpha=0.2) )
    listOfDesigns +=   ax[0][1].plot( [xC[0,0]], [xC[1,0]], '.k', markersize=10 )
    listOfDesigns +=   ax[0][1].plot( [Xinit[0,0]], [Xinit[1,0]], '.r', markersize=10 )
    listOfDesigns.append(  pu.plotEllipse(ax[1][1], xC, Pc, 1., plotAx=[0,3], color=[0,0,1.,1.], faceAlpha=0.2) )
    listOfDesigns +=   ax[1][1].plot( [xC[0,0]], [xC[3,0]], '.k', markersize=10 )
    listOfDesigns +=   ax[1][1].plot( [Xinit[0,0]], [Xinit[3,0]], '.r', markersize=10 )
    print(T)
    plt.savefig('allImagesDist2/im_{0:05d}.png'.format(KK), dpi=300 )
    KK+=1
    plt.close()
    #return listOfDesigns#[0], listOfDesigns[1], listOfDesigns[2], listOfDesigns[3], listOfDesigns[4], listOfDesigns[5],listOfDesigns[6],listOfDesigns[7],listOfDesigns[8]

#fUpdate = lambda ff: update(ff, listOfDesigns, allT, allV, allX)

#line_ani = animation.FuncAnimation(fig, fUpdate, 12, interval=dT*1000, blit=True)

# To save the animation, use the command: line_ani.save('lines.mp4')
#line_ani.save('lines.mp4')

#plt.show()
    
      
    
    
    
