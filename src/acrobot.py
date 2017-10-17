from coreUtils import *
import plotUtils as pu

import dynamicSystems as ds

##Taken from Drake - RobotLocomotion @ CSAIL
#Implements the dynamics representing the acrobot
#Geometry
l1 = 1; l2 = 2;  
#Mass
m1 = 1; m2 = 1;  
#Damping
b1=.1;  b2=.1;
#Gravity center and inertial moment
lc1 = .5; lc2 = 1; 
Ic1 = .083;  Ic2 = .33;
#Gravity
g = 9.81;
 
#Constraints -> It is important the keep a certain margin between the reference 
#input and the input limits to allow for stabilitzation
uLim = [-10, 10]

#Get the dynamical system
#Original definition
#===============================================================================
#       m2l1lc2 = m2*l1*lc2;  % occurs often!
# 
#       c = cos(q(1:2,:));  s = sin(q(1:2,:));  s12 = sin(q(1,:)+q(2,:));
#       
#       h12 = I2 + m2l1lc2*c(2);
#       H = [ I1 + I2 + m2*l1^2 + 2*m2l1lc2*c(2), h12; h12, I2 ];
#       
#       C = [ -2*m2l1lc2*s(2)*qd(2), -m2l1lc2*s(2)*qd(2); m2l1lc2*s(2)*qd(1), 0 ];
#       G = g*[ m1*lc1*s(1) + m2*(l1*s(1)+lc2*s12); m2*lc2*s12 ];
#             
#       % accumulate total C and add a damping term:
#       C = C*qd + G + [b1;b2].*qd;
# 
#       B = [0; 1];
#===============================================================================
#H is the mass matrix
#The dynamics
#M.qdd + C_qdq.qd + g_q = B.tau

#Get the state-space variables
x=sMz(4,1) 
for k in range(4):
    x[k] = sympy.symbols('x{0:d}'.format(k))
    exec('x{0:d}=x[{0:d}]'.format(k))
#helper
I1 = Ic1 + m1*lc1**2
I2 = Ic2 + m2*lc2**2;

h12 = I2 + m2*l1*lc2*sympy.cos(x1);

M = sMa([[ I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*sympy.cos(x1), h12], [h12, I2]])
Mi = M.inv() #invert #This is reasonable for small matrices but has to be probably changed when increasing complexity
G = g*sMa([ m1*lc1*sympy.sin(x0) + m2*(l1*sympy.sin(x0)+lc2*sympy.sin(x0+x1)), m2*lc2*sympy.sin(x0+x1)]);
#old
#C = sMa( [[ -2*m2*l1*lc2*sympy.sin(x1)*x3, -m2*l1*lc2*sympy.sin(x1)*x3], [m2*l1*lc2*sympy.sin(x1)*x2, 0 ]] )
#xdd = -Mi*(C*x[2:,0]+G)
#new
C = sMa([ -2.*m2*l1*lc2*sin(x1)*x3*x2 -m2*l1*lc2*sin(x1)*x3*x3 + b1*x2, m2*l1*lc2*sin(x1)*x2*x2 + b2*x3 ])
xdd = -Mi*(C+G)
#System dynamics
#[x1,x2,x3,x4] = [q1, q2, qd1, qd2]
dynF = sMz(4,1)
dynF[0:2,0]=x[2:,0]
dynF[2:,0]=xdd
#Control mapping -> Non-linear here
#M.inv()*B*u
dynG = sMz(4,1)
dynG[2:,0] = Mi*sMa([0,1])

#Numbers
nx = 4
nu = 1
#Create the dict/matrices needed for numeric vars
nX = (nx*(nx+1))//2


#uLim = [min(1.25*np.min(allU),-2.), max(1.25*np.max(allU),2)]
inputCstr = ds.boxCstr(nu, uLim[0],uLim[1])
#dynSys = ds.dynamicalSys(dynF, dynG, varNames=x, inputCstr=inputCstr, so2DimsIn=[0,1])
dynSys = ds.dynamicalSys(dynF, dynG, varNames=x, inputCstr=inputCstr, sysName='acrobot', so2DimsIn=None)#The first two elements of x are indeed so2 states. However this currently provide no advantage and complicates the calculation of x.T.P.x and the plotting 
dynSys.innerZone = 10e-2;

#Define a function for plotting
def getDrawPos(X):
    pos = np.zeros((2,3))
    pos[0,1] =  l1*np.sin(X[0])
    pos[1,1] = -l1*np.cos(X[0])
    pos[0,2] = pos[0,1] + l2*np.sin(X[0]+X[1])
    pos[1,2] = pos[1,1] - l2*np.cos(X[0]+X[1])
    
    return pos