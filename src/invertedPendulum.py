import numpy as np
import dynamicSystems as ds

##Taken from Drake - RobotLocomotion @ CSAIL
#Implements the dynamics representing the inverted pendulum

m = 1;   #% kg
l = .5;  #% m
b = 0.1; #% kg m^2 /s
lc = .5; #% m
I = .25; #%m*l^2; % kg*m^2
g = 9.81; #% m/s^2

#System dynamics
dynF = ['x1', '(-{0:.8f}*{1:.8f}*{2:.8f}*sin(x0) - {3:.8f}*x1)/{4:.8f}'.format(m,g,lc,b,I)]
#Input dynamics (linear !)
dynG = [['0.'], ['1./{0:.8f}'.format(I)]]
#Numbers
nx = 2
nX = (nx*(nx+1))//2
nu = 1


#dynSys = ds.dynamicalSys(dynF, dynG, so2DimsIn=[0])
dynSys = ds.dynamicalSys(dynF, dynG, so2DimsIn=None, sysName="invPend")#The first element of x is indeed so2 states. However this currently provide no advantage and complicates the calculation of x.T.P.x and the plotting
dynSys.innerZone = 10e-2;
xSym, ySym, uSym = dynSys.x, dynSys.xTrans, dynSys.u

#Constraints
inputCstr = ds.boxCstr(nu, -3, 3)
dynSys.inputCstr = inputCstr

#Define a function for plotting
def getDrawPos(X):
    pos = np.zeros((2,2))
    pos[0,1] =  l*np.sin(X[0])
    pos[1,1] = -l*np.cos(X[0])
    
    return pos