from coreUtils import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as mcoll

from mpl_toolkits.mplot3d import Axes3D
from numpy.f2py.auxfuncs import isdouble

import dynSys

dynStringDict = {0:{'Q':'QP-ctrl', 'C':'SMC'}, 1:{'L':'Lin-Sys_dyn', 'P':'Poly-Sys_dyn', 'O':'NL-Sys_dyn'}, 2:{'L':'Lin-Ipt_dyn', 'P':'Poly-Ipt_dyn', 'O':'NL-Ipt_dyn', 'R':'Lin-Ipt_dyn_zoned'}}

#Unit vector in "x" direction
uVec = np.array([[1.],[0.]])
#Simple 2d rotation matrix
def Rot(alpha):
    R = np.zeros((2,2))
    R[0,0]=R[1,1]=np.cos(alpha)
    R[1,0]=np.sin(alpha)
    R[0,1]=-R[1,0]
    return R 
#Get regularly distributed points on the surface of {x|x.T.P.x} in 2d
def getV(P = np.identity(2), n=101, endPoint=True):
    Ang = np.linspace(0,2*np.pi, n, endpoint = endPoint)
    V = np.zeros((2,Ang.size))
    C = np.linalg.inv(np.linalg.cholesky(P)).T
    for k in range(Ang.size):
        V[:,k] = np.dot(Rot(Ang[k]), uVec).squeeze()
    return ndot(C,V)


###############################################################################
def projectEllip(P,T):
        #Return a "P" representing the projection of a matrix onto a affine subspace
        return np.linalg.inv(ndot(T.T,np.linalg.inv(P),T))
###############################################################################
def plot(x,y=None,z=None):
    if z is None:
        ff = plt.figure()
        aa = ff.add_subplot(111)
        if y is None:
            aa.plot(x)
        else:
            aa.plot(x,y)
    else:
        pass
    
    return aa

#########################################################################
def getEllipse(pos,P,alpha, deltaPos=None):
    #center and shape to matplotlib patch
    v, E = eigh(P)
    if deltaPos is None:
        deltaPos=np.zeros((pos.shape))
    #tbd do calculations for offset
    orient = np.arctan2(E[1,0],E[0,0])*180.0/np.pi
    return Ellipse(xy=pos, height=2.0*np.sqrt(alpha) * 1.0/np.sqrt(v[1]), width=2.0*np.sqrt(alpha) * 1.0/np.sqrt(v[0]), angle=orient)

#########################################################################
def plotEllipse(ax, pos, P, alpha, plotAx = np.array([0,1]), deltaPos=None, color = [0.0,0.0,1.0,1.0], faceAlpha=0.5, pltStyle="proj"):
    color=np.array(dp(color)); color[-1]=color[-1]*faceAlpha; color=list(color)
    
    if pltStyle=='proj':
        if len(plotAx) == 2:
            T = np.zeros((P.shape[0],2))
            T[plotAx[0],0]=1.
            T[plotAx[1], 1]=1.
        else:
            assert (T.shape[0]==P.shape[0] and T.shape[1]==P.shape[1]), "No valid affine 2d sub-space" 
        Pt = projectEllip(P, T)
    elif pltStyle=='inter':
        Pt = P[np.ix_(plotAx, plotAx)]
    else:
        assert 0, "No valid pltStyle for ellip given"
    
    e = getEllipse(pos[plotAx], Pt, alpha)
    e.set_linewidth(1.)
    e.set_edgecolor(color[:3]+[1.])
    ax.add_patch(e)
    e.set_facecolor( color )
    return e

#########################################################################

def plotConv(ax, convExpr, xSym, pos, P, alpha=1., plotAx=np.array([0,1]), deltaPos = None, nPX = 20, nPY=20, thisB=None, thisA=None):
    ellip = getEllipse(pos, P, alpha, deltaPos)
    plotEllipse(ax, pos, P, alpha, plotAx, deltaPos, color=[1.0,1.0,1.0,1.0], faceAlpha=0.)
    
    ax.autoscale(1)
    limXY = np.array( ax.get_xlim()+ax.get_ylim() )
    
    #limXY= np.array([ellip.center[0]-ellip.width*1.2/2., ellip.center[0]+ellip.width*1.2/2., ellip.center[0]-ellip.height*1.2/2., ellip.center[0]+ellip.height*1.2/2.])
    xGrid, yGrid = np.meshgrid(np.linspace(limXY[0], limXY[1], nPX), np.linspace(limXY[2], limXY[3], nPY))
    xGrid.resize((nPX*nPY,))
    yGrid.resize((nPX*nPY,))
    zValue = np.zeros((nPX*nPY,))
    
    #Get all the convergence values, expect sympy expr, TBD: numpify it!
    thisXY = np.zeros(pos.shape)
    for k, (x,y) in enumerate( zip(xGrid, yGrid) ):
        thisXY[plotAx[0]]=x
        thisXY[plotAx[1]]=y
        zValue[k] = myEvalf(convExpr, prec=16, subsDict=dict(zip(xSym, thisXY)))
    if (not (thisB is None)) and (not (thisA is None)):
        #Print hyperplane projections
        #Not suited fo n dimensions yet
        xy = np.vstack((xGrid, yGrid))
    
    xGrid.resize((nPY, nPX))
    yGrid.resize((nPY, nPX))
    zValue.resize((nPY, nPX))
    
    thisContour = ax.contourf(xGrid,yGrid, zValue)
    thisContour0 = ax.contour(xGrid,yGrid, zValue, [0], color='k', LineWidth=10)
    
    plt.colorbar(thisContour, shrink=0.8, extend='both')
    
    if (not (thisB is None)) and (not (thisA is None)):
        for k in range(thisB.shape[0]):
            thisZBorderVal = np.dot( thisB[k,:], xy )-thisA
            thisZBorderVal.resize((nPY, nPX))
            ax.contour(xGrid,yGrid, thisZBorderVal, [0], color='r', LineWidth=10)
        
    
    plotEllipse(ax, pos, P, alpha, plotAx, deltaPos, color=[1.0,1.0,1.0,1.0], faceAlpha=0.)
    
    return 0
###############################################################################
def get2DShapeAsPoints(X, n=11):
    n=np.array(n).squeeze()
    if n.size==1:
        n=np.tile( n, X.shape[1]-1 )
    assert n.size==X.shape[1]-1, "n is either scalar or column size of X-1"
    
    nn=0
    out=np.zeros((2,np.sum(n)))
    for k, nNext in enumerate(n):
        out[0,nn:nn+nNext] = np.linspace(X[0,k], X[0,k+1], nNext)
        out[1,nn:nn+nNext] = np.linspace(X[1,k], X[1,k+1], nNext)
        nn=nn+nNext
    
    return out
    
###############################################################################

def plotFunnel(aLyapReg, plotAx = [0,1], NN=200, opts = {}):
    #Plot the entire funnel constructed around a reference trajectory
    #plot style can be either "proj"ection or "inter"section to get 2d representation of the ellipsoid
    #assert isinstance(aLyapReg.dynSys, dynSys.dynamicalSys)
    optsBase = {'doEachDV':False, 'doEachStream':True, 'doEachConv':False, 'whichDyn':'QOO', 'nX':20, 'nY':21, 'interSteps':0, "plots":True, "diffSteps":1.75, 'pltStyle':'proj', 'regEpsOpt':1., 'streamColor':'input', 'equalScale':True, 'doScaleCostQP':False,"FQP2":False,'fullQOut':False}
    
    optsBase.update(opts)
    
    opts = optsBase
    
    #Parse dynamics to title string
    dynString = dynStringDict[0][opts['whichDyn'][0]]+' '+dynStringDict[1][opts['whichDyn'][1]]+' '+dynStringDict[2][opts['whichDyn'][2]]
    
    
    nX = (aLyapReg.dynSys.nx*(aLyapReg.dynSys.nx+1))//2
    tt = np.linspace(aLyapReg.tSteps[0], aLyapReg.tSteps[-1], NN)
    
    xr = aLyapReg.refTraj(tt)[0][plotAx,:]
    
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.scatter(xr[0,:], xr[1,:], c=cmx.jet( (tt-tt[0])/(tt[-1]-tt[0]) ), edgecolor='none' )
    ax.set_title("{0} : {1}-{2}".format(optsBase['pltStyle'], plotAx[0], plotAx[1]))
    #Attention this alpha is different from the one used for plotting above
    #here
    #x'.alpha.P.x <= 1
    #above
    #x'.P.x <= alpha
    
    allTSteps = []
    for [t0, t1] in zip( aLyapReg.tSteps[0:-1], aLyapReg.tSteps[1:] ):
        allTSteps += list( np.linspace(t0, t1, num=opts['interSteps']+1, endpoint=False) )
    allTSteps += [aLyapReg.tSteps[-1]]
    allTSteps = aLyapReg.tSteps[np.linspace(0,aLyapReg.tSteps.size-1, max(int(aLyapReg.tSteps.size/(opts["diffSteps"]+1)),2), endpoint=True).astype(np.int_)]
    #for aT, (aP, aAlpha) in zip(aLyapReg.tSteps, aLyapReg.allShapes):
    trajT = np.linspace(allTSteps[0], allTSteps[-1], 1000)
    trajX = aLyapReg.refTraj.xref(trajT)
    if not(aLyapReg.dynSys.so2Dims is None):
        trajX = replaceSo2(trajX, dims=aLyapReg.dynSys.so2Dims);
    ax.plot(trajX[plotAx[0],:], trajX[plotAx[1],:], '-k')
    for aT in allTSteps:
        #thisX = aLyapReg.refTraj(aT)[0]
        #plotEllipse(ax, thisX, aP, 1./aAlpha, plotAx, color=cmx.jet((aT-tt[0])/(tt[-1]-tt[0])), faceAlpha=0.5)
        thisX,aP,_ = aLyapReg.getEllip(aT)
        #print(thisX)
        #print(aP)
        plotEllipse(ax, thisX, aP, 1., plotAx, color=cmx.jet((aT-tt[0])/(tt[-1]-tt[0])), faceAlpha=0.5, pltStyle=opts["pltStyle"])
        ax.autoscale(1)
    if opts['plots']:
        try:
            plt.savefig("../tempFig/wholeFunnel_{0}_{1}.pdf".format(plotAx[0], plotAx[1]))
        except:
            print("Could not save figure as -../tempFig/evolution.pdf-")
    
    allPlots = [{} for _ in allTSteps ]
    if opts['doEachStream'] or opts['doEachConv'] or opts['doEachDV']:
            
        #for k, (aT, (aP, aAlpha)) in enumerate(zip(aLyapReg.tSteps, aLyapReg.allShapes)):
        for k, aT in enumerate(allTSteps):
            
            thisX, thisXd = aLyapReg.refTraj(aT)[0:2]
            _,aP,dP = aLyapReg.getEllip(aT)
            
            ff = plt.figure()
            aa = ff.add_subplot(111)
            #plotEllipse(aa, thisX, aP, 1./aAlpha, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
            plotEllipse(aa, thisX, aP, 1., plotAx, color=[0.,0.,0.,1.], faceAlpha=0.)
            aa.autoscale()
            overSizedRatio = 10000.
            xLim = aa.get_xlim()
            yLim = aa.get_ylim()
            dXX = xLim[1]-xLim[0]
            xLim =[xLim[0]-dXX/overSizedRatio, xLim[1]+dXX/overSizedRatio]
            dYY = yLim[1]-yLim[0]
            yLim =[yLim[0]-dYY/overSizedRatio, yLim[1]+dYY/overSizedRatio]
            plt.close(ff)
                
            xGrid, yGrid = np.meshgrid( np.linspace(xLim[0], xLim[1], opts['nX']), np.linspace(yLim[0], yLim[1], opts['nY']) )
            xGrid.resize((opts['nX']*opts['nY'],))
            yGrid.resize((opts['nX']*opts['nY'],))
            allPlots[k]['x0'] = thisX
            allPlots[k]['xd0'] = thisXd
            thisX = np.tile(thisX, (1,opts['nX']*opts['nY']))
            thisX[plotAx[0],:] = xGrid
            thisX[plotAx[1],:] = yGrid
            allPlots[k]['x'] = thisX
            
            assert (opts['whichDyn'][0] in ['C','Q']) and (opts['whichDyn'][1] in ['O','P','L']) and (opts['whichDyn'][2] in ['O','P','L','R']), 'Could not parse dynMode'
            if opts['whichDyn'][0] == 'C':
                allPlots[k]['xd'], allPlots[k]['allU'] = aLyapReg.dynSys.getBestF(aP, allPlots[k]['x0'], thisX, mode=opts['whichDyn'][1:], t=aT, fullOut=True)
            else:
                #getBestFQP(self,Pt,dPt,alpha,x0,x,mode='OO',t=0.,regEps=0.1 )
                if not opts['FQP2']:
                    if True:
                        allPlots[k]['xd'], allPlots[k]['indSuccQ'], allPlots[k]['indTightQ'], allPlots[k]['anyOutMaxQ'], allPlots[k]['allU'] = aLyapReg.dynSys.getBestFQP(aP,dP,aLyapReg.convergenceRate,allPlots[k]['x0'], thisX,mode=opts['whichDyn'][1:],t=aT, regEps=opts['regEpsOpt'],fullOut=True, doScaleCost=opts['doScaleCostQP'])
                    else:
                        allPlots[k]['xd'] = aLyapReg.dynSys.getBestFQP(aP,dP,aLyapReg.convergenceRate,allPlots[k]['x0'],thisX,mode=opts['whichDyn'][1:],t=aT,regEps=opts['regEpsOpt'], fullOut=False, doScaleCost=opts['doScaleCostQP'])
                else:
                    if True:
                        allPlots[k]['xd'], allPlots[k]['indSuccQ'], allPlots[k]['indTightQ'], allPlots[k]['anyOutMaxQ'], allPlots[k]['allU'] = aLyapReg.dynSys.getBestFQP2(aP,dP,aLyapReg.convergenceRate,allPlots[k]['x0'], thisX,mode=opts['whichDyn'][1:],t=aT, regEps=opts['regEpsOpt'],fullOut=True, doScaleCost=opts['doScaleCostQP'])
                    else:
                        allPlots[k]['xd'] = aLyapReg.dynSys.getBestFQP2(aP,dP,aLyapReg.convergenceRate,allPlots[k]['x0'],thisX,mode=opts['whichDyn'][1:],t=aT,regEps=opts['regEpsOpt'], fullOut=False, doScaleCost=opts['doScaleCostQP'])
                        
            xGrid.resize((opts['nY'],opts['nX']))
            yGrid.resize((opts['nY'],opts['nX']))
            
            if opts['equalScale']:
                plt.axes().set_aspect('equal', 'datalim')
            
            #Get stuff for dead-zone plottingb
            #_,aB,_,_ = aLyapReg.dynSys.getTaylorS3Fast(aLyapReg.refTraj(aT)[0])
            #aPb = ndot(aP, aB)
            #allAh = aPb
            #allAh[plotAx[0],:] = aPb[allAh[plotAx[1],:]]
            #allAh[plotAx[1],:] = aPb[allAh[plotAx[0],:]]
            

            if (opts['whichDyn'][0]=='Q') and (opts['fullQOut']):
                figQO = plt.figure()
                axQO = figQO.add_subplot(111)
                allPlots[k]['figQO'] = figQO
                allPlots[k]['axQO'] = axQO
                allPlots[k]['axQO'].set_title( "Active Cstr for "+dynString )
                indSuccNonTight = np.logical_and( allPlots[k]['indSuccQ'], np.logical_not(allPlots[k]['indTightQ']) )
                indSuccTight = np.logical_and( allPlots[k]['indSuccQ'], allPlots[k]['indTightQ'] )
                axQO.plot(allPlots[k]['x'][0,allPlots[k]['anyOutMaxQ']],allPlots[k]['x'][1,allPlots[k]['anyOutMaxQ']],'or')
                axQO.plot( allPlots[k]['x'][0,indSuccNonTight], allPlots[k]['x'][1,indSuccNonTight], '.g'  )
                axQO.plot( allPlots[k]['x'][0,indSuccTight], allPlots[k]['x'][1,indSuccTight],'.b' )
                plotEllipse(axQO,allPlots[k]['x0'],aP,1.,plotAx,color=[0.,0.,0.,1.],faceAlpha=0.)
                axQO.autoscale()

            if opts['doEachStream']:
                figS = plt.figure()
                axS = figS.add_subplot(111)
                allPlots[k]['figS']=figS
                allPlots[k]['axS']=axS
                allPlots[k]['axS'].set_title( "Streamplot "+dynString)
                #plotEllipse(axS, allPlots[k]['x0'], aP, 1./aAlpha, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                plotEllipse(axS, allPlots[k]['x0'], aP, 1., plotAx, color=[0.,0.,0.,1.], faceAlpha=0.)
                if not (aLyapReg.excludeInner is None):
                    #plotEllipse(axS, allPlots[k]['x0'], aP, aLyapReg.excludeInner*1./aAlpha, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                    plotEllipse(axS, allPlots[k]['x0'], aP, aLyapReg.excludeInner, plotAx, color=[0.,0.,0.,1.], faceAlpha=0.)
                #if abs(aLyapReg.epsDeadZone) > 1e-5:
                    #Get the seperating hyperplane normal vector
                #Plot the seperating hyperplanes
                
                axS.autoscale()
                xLim = axS.get_xlim()
                yLim = axS.get_ylim()
                if opts['equalScale']:
                    plt.axes().set_aspect('equal', 'datalim')
                
                deltaXd = allPlots[k]['xd'] - thisXd
                if 1:
                    if opts['streamColor'] == 'input':
                        assert allPlots[k]['allU'].shape[0] == 1, 'Only works for one input'
                        deltaXdInt = allPlots[k]['allU'][0,:].copy()
                        deltaXdInt.resize((opts['nY'],opts['nX']))
                        aMap = cmx.winter
                    else:
                        deltaXdInt = np.sqrt(np.sum(np.square(deltaXd), 0))
                        deltaXdInt.resize((opts['nY'],opts['nX']))
                        aMap = cmx.jet
                elif 0:
                    #getIndU(self, Pt, x0, x, allInputPerms)
                    deltaXdInt = aLyapReg.dynSys.getIndU(aP, allPlots[k]['x0'], allPlots[k]['x'], aLyapReg.allInputPermutations)
                    deltaXdInt.resize((opts['nY'],opts['nX']))
                    aMap = cmx.jet
                else:
                    deltaXdInt = zeros(allPlots[k]['x'].shape[1],)
                    deltaXdInt.resize((opts['nY'],opts['nX']))
                    aMap = cmx.jet
                    
                allPlots[k]['thisStream'] = allPlots[k]['axS'].streamplot(xGrid, yGrid, deltaXd[plotAx[0],:].reshape((opts['nY'],opts['nX'])), deltaXd[plotAx[1],:].reshape((opts['nY'],opts['nX'])), color=deltaXdInt, cmap=aMap )
                if opts['plots']:
                    try:
                        plt.savefig("../tempFig/stream_{0}.pdf".format(k))
                    except:
                        print("Could not save figure as -../tempFig/stream_{0}.pdf-".format(k))
            if opts['doEachConv']:
                figC = plt.figure()
                axC = figC.add_subplot(111)
                allPlots[k]['figC']=figC
                allPlots[k]['axC']=axC
                allPlots[k]['axC'].set_title("Convergence plot "+dynString)
                if opts['equalScale']:
                    plt.axes().set_aspect('equal', 'datalim')

                #plotEllipse(axC, allPlots[k]['x0'], aP, 1./aAlpha, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                plotEllipse(axC, allPlots[k]['x0'], aP, 1., plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                if not (aLyapReg.excludeInner is None):
                    #plotEllipse(axS, allPlots[k]['x0'], aP, aLyapReg.excludeInner*1./aAlpha, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                    plotEllipse(axS, allPlots[k]['x0'], aP, aLyapReg.excludeInner, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                axC.autoscale()
                xLim = axC.get_xlim()
                yLim = axC.get_ylim()
            
                deltaX = allPlots[k]['x'] - allPlots[k]['x0']
                deltaXd = allPlots[k]['xd'] - allPlots[k]['xd0']
                #Ct = np.linalg.cholesky(aP*aAlpha)
                Ct = np.linalg.cholesky(aP)
                Vx = np.sqrt( np.sum(np.square(ndot(Ct, deltaX)),0) )
                #dV = 2*x'.Pt.xd + x'.dP.x
                print(k)
                if k == len(aLyapReg.allShapes)-1:
                    Pn, an = aLyapReg.allShapes[k-2]
                    Pn1, an1 = aLyapReg.allShapes[k-1]
                    dT = aLyapReg.tSteps[k-1] - aLyapReg.tSteps[k-1]
                else:
                    Pn, an = aLyapReg.allShapes[k]
                    Pn1, an1 = aLyapReg.allShapes[k+1]
                    dT = aLyapReg.tSteps[k+1] - aLyapReg.tSteps[k]
                #dP = 1./dT*(an1*Pn1-an*Pn)
                dCt = np.linalg.cholesky(dP)
                Vdx = np.sqrt( np.sum(np.square(ndot(dCt, deltaX)),0) )
                
                dV = 2.*np.sum( np.multiply(deltaX, ndot(aP*aAlpha, deltaXd)), 0) + Vdx + aLyapReg.convergenceRate*Vx
                
                convRatio = np.divide(dV, Vx)
                if aLyapReg.excludeInner is None:
                    ind = Vx >= 0.01
                else:
                    ind = Vx <= aLyapReg.excludeInner
                
                convRatio[ind] = 0.
                #allPlots[k]['thisConv'] = allPlots[k]['axC'].contour(xGrid, yGrid, convRatio.reshape((opts['nY'],opts['nX'])))
                #plt.colorbar(allPlots[k]['thisConv'], shrink=0.8, extend='both')
                xGrid.resize((opts['nY']*opts['nX'],))
                yGrid.resize((opts['nY']*opts['nX'],))
                convRMin = -5*aLyapReg.convergenceRate
                convRMax =  2*aLyapReg.convergenceRate
                convRatio[convRatio<convRMin]= convRMin
                convRatio[convRatio>convRMax]= convRMax
                convRation = (convRatio-convRMin)/(convRMax-convRMin)
                allPlots[k]['thisConv'] = allPlots[k]['axC'].scatter(xGrid, yGrid, c=convRation, edgecolor='none', cmap=cmx.jet)
                plt.colorbar(allPlots[k]['thisConv'], shrink=0.8, extend='both')
            if opts['doEachDV']:
                figDV = plt.figure()
                axDV = figDV.add_subplot(111)
                allPlots[k]['figDV']=figDV
                allPlots[k]['axDV']=axDV
                if opts['equalScale']:
                    plt.axes().set_aspect('equal', 'datalim')
                
                #plotEllipse(axDV, allPlots[k]['x0'], aP, 1./aAlpha, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                plotEllipse(axDV, allPlots[k]['x0'], aP, 1., plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                if not (aLyapReg.excludeInner is None):
                    #plotEllipse(axDV, allPlots[k]['x0'], aP, aLyapReg.excludeInner*1./aAlpha, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                    plotEllipse(axDV, allPlots[k]['x0'], aP, aLyapReg.excludeInner, plotAx, color=[1.,1.,1.,1.], faceAlpha=0.)
                axDV.autoscale()
                xLim = axDV.get_xlim()
                yLim = axDV.get_ylim()
                
                deltaXd = allPlots[k]['xd'] - thisXd
                allPlots[k]['thisDV'] = allPlots[k]['axDV'].quiver(xGrid, yGrid, deltaXd[plotAx[0],:].reshape((opts['nY'],opts['nX'])), deltaXd[plotAx[1],:].reshape((opts['nY'],opts['nX'])) ) 
                
    return 0
    
###############################################################################

def replayFunnel(aLyapShape, tVec=None, Nsteps=None, Ninit=5, initPoints = None, dynamicsMode = "COR", pltAx=[0,1], mode="seq", pltStyle="proj", otherOptsIn={}):
    #Simulate some initial positions and plot the results
    otherOpts = {'regEpsOpt':1., 'equalScale':True, 'doScaleCostQP':False, 'FQP2':False}
    otherOpts.update(otherOptsIn)

    # Parse dynamics to title string
    dynString = dynStringDict[0][dynamicsMode[0]]+' '+dynStringDict[1][dynamicsMode[1]]+' '+dynStringDict[2][dynamicsMode[2]]
    
    if Nsteps is None:
        Nsteps = len(aLyapShape.tSteps)
    
    if tVec is None:
        tVec = np.linspace(aLyapShape.tSteps[0], aLyapShape.tSteps[-1], Nsteps)
    Nsteps = tVec.size
    
    
    if initPoints is None:
        #Get the first ellip
        initPoints = aLyapShape.getInitPoints( tVec[0], N=Ninit, ax=pltAx)
    
    [dim, Ninit] = initPoints.shape
    #plot for "cost" evolution
    figConv = plt.figure()
    axConv = figConv.add_subplot(111)
    #Plot the allowed cost
    axConv.plot(tVec, np.exp(-aLyapShape.convergenceRate*tVec), '.-r', linewidth=2)
    
    figEvol = plt.figure()
    axEvol = figEvol.add_subplot(111)
    axEvol.set_title("Convergence for "+dynString)
    
    fAll, axAll = plt.subplots(1, aLyapShape.dynSys.nx)
    
    
    nnc = plt.get_cmap("viridis")(np.linspace(0,1,Ninit))
        
    #Do the integration
    if mode=="seq":
        Nloops = Ninit
        Ninit=1
    elif mode=="simultan":
        Nloops = 1
    else:
        assert 0

    if dynamicsMode[0] == 'C':
        fInt = lambda thisX,thisT:aLyapShape.dynSys.getBestF(aLyapShape.getEllip(thisT)[1],
                                                             aLyapShape.refTraj.xref(thisT),
                                                             thisX.reshape((Ninit,dim)).T,mode=dynamicsMode[1:],
                                                             u0=aLyapShape.refTraj.uref(thisT),t=thisT).T.reshape((dim*Ninit,))
    elif dynamicsMode[0] == 'Q':
        # getBestFQP(self,Pt,dPt,alpha,x0,x,mode='OO',t=0.,regEps=0.1)
        assert otherOpts['FQP2'] is False, 'Deprecated'
        def fInt(thisX,thisT):
            _,Pt,dPt = aLyapShape.getEllip(thisT)
            return aLyapShape.dynSys.getBestFQP(Pt,dPt,aLyapShape.convergenceRate,aLyapShape.refTraj.xref(thisT),
                                                thisX.reshape((Ninit,dim)).T,mode=dynamicsMode[1:],
                                                u0=aLyapShape.refTraj.uref(thisT),t=thisT,regEps=otherOpts['regEpsOpt'],
                                                doScaleCost=otherOpts['doScaleCostQP']).T.reshape((dim*Ninit,))
    else:
        assert 0,"dynamicsMode {0} could not be interpreted".format(dynamicsMode)

    for k in range(aLyapShape.dynSys.nx):
        axAll[k].set_title("Trajectories for "+dynString)
    
    for ii in range(Nloops):
        print("{0}:{1}".format(Ninit*ii, Ninit*(ii+1)))
        lastX = initPoints[:,Ninit*ii:Ninit*(ii+1)].T.reshape((dim*Ninit,))
        for k in range(Nsteps-1):
            
            thisTvec = np.linspace(tVec[k], tVec[k+1], 101)
            thisXref = aLyapShape.refTraj.xref(thisTvec)
            newX = scipy.integrate.odeint(fInt, lastX, thisTvec)
            lastX = newX[-1,:]
            newX = newX.T
            #get the cost
            newCost = aLyapShape.getCost(thisTvec, newX-np.tile(thisXref, (Ninit,1)), thisDim=[dim,Ninit])
            #Do the plots
            for i in range(Ninit):
                axConv.plot( thisTvec, newCost[i,:], c=nnc[i,:] )
                axEvol.plot( newX[i*dim+pltAx[0],:], newX[i*dim+pltAx[1],:], c=nnc[i,:] )
                axEvol.plot( newX[i*dim+pltAx[0],0], newX[i*dim+pltAx[1],0], 'o', c=nnc[i,:] )
                for ii in range(aLyapShape.dynSys.nx):
                    axAll[ii].plot(thisTvec, newX[i*dim+ii,:], c=nnc[i,:] )
            axEvol.plot( thisXref[pltAx[0]], thisXref[pltAx[1]], 'k', linewidth=2 )
            #Plot the ellipse
            #if ii==0:
            #    _,thisP, _ = aLyapShape.getEllip(tVec[k])
            #    plotEllipse(axEvol, thisXref[:,0], thisP, 1., pltAx, faceAlpha=0.0)
    for k in range(Nsteps-1):
        _,thisP, _ = aLyapShape.getEllip(tVec[k])
        plotEllipse(axEvol, thisXref[:,0], thisP, 1., pltAx, faceAlpha=0.2, pltStyle=pltStyle)
        
    #Final
    for i in range(Ninit):
            axEvol.plot( newX[i*dim+pltAx[0],-1], newX[i*dim+pltAx[1],-1], 'o', c=nnc[i,:] )
    #Plot the ellipse
    _,thisP, _ = aLyapShape.getEllip(tVec[k+1])
    plotEllipse(axEvol, thisXref[:,-1], thisP, 1., pltAx, faceAlpha=0.0)
    
    return initPoints, 
        
    
    
    
    
    

###############################################################################

def checkLin(defList):
    
    k,Acstr,Bcstr = defList
    
    aGoal = matrix(Acstr[k,:])
    bGoal = matrix(Bcstr[k])
    
    AcstrM = matrix(np.delete(Acstr, k, 0))
    BcstrM = matrix(np.delete(Bcstr, k, 0))
    
    sol = solvers.lp(-aGoal, G=AcstrM, h=BcstrM)
    
    if sol['status'] == 'optimal':
        return float((aGoal.T*sol['x'])[0])-bGoal[0]
    else:
        return float('NaN')
    
###############################################################################
def checkCstr(*args):
    #expects a list of (A, B) tuples: [ [A0,B0], [A1, B1],... ]
    
    thisWorkers = Pool(processes=numCPU)
    
    thisColor = cmx.jet(np.linspace(0,1,len(args)))
    
    allA = args[0][0]
    allB = args[0][1]
    nCstr = [allA.shape[0]]
    
    for k, (aA, aB) in enumerate(args):
        if k==0:
            allA = aA
            allB = aB
            nCstr = [aA.shape[0]]
            nCstrCum = [0]
            allRes = [None]
        else:
            allA = np.vstack((allA, aA))
            allB = np.vstack((allB, aB))
            nCstrCum.append(nCstrCum[-1]+nCstr[-1])
            nCstr.append(aA.shape[0])
            allRes.append(None)
    
    allANorm = np.sqrt(np.sum(np.square(allA), 1))
    allANorm.resize((allANorm.size,1))
    
    allB = np.divide(allB, allANorm)
    allA = np.divide(allA, allANorm)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for k in range(len(args)):
        thisCstr = list(range(nCstrCum[k], nCstrCum[k]+nCstr[k]))
        
        thisTasks = list(zip(thisCstr, nCstr[k]*[allA], nCstr[k]*[allB]))
        
        if glbDBG:
            allRes[k] = np.array(list(map(checkLin, thisTasks)))
        else:
            allRes[k] = np.array(list(thisWorkers.map(checkLin, thisTasks)))
        
        ax.scatter(np.array(thisCstr), allRes[k], c=thisColor[k], edgecolor='none')
    
    return allRes,ax
###############################################################################

def closestPoint(inList):
    k,A = inList
    
    thisPoint = A[k,:]
    #otherPoint = np.delete(A,k,0)
    
    distPoint = np.delete(np.sqrt(np.sum(np.square(A-thisPoint), 1)), k)
    
    return np.min(distPoint)
        
###############################################################################
def pltCstrProximity(*args):
    #expects a list of (A, B) tuples: [ [A0,B0], [A1, B1],... ]
    
    #thisWorkers = Pool(processes=numCPU)
    
    thisColor = cmx.jet(np.linspace(0,1,len(args)))
    
    allA = args[0][0]
    allB = args[0][1]
    nCstr = [allA.shape[0]]
    
    for k, (aA, aB) in enumerate(args):
        if k==0:
            allA = aA
            allB = aB
            nCstr = [aA.shape[0]]
            nCstrCum = [0]
            allRes = [None]
        else:
            allA = np.vstack((allA, aA))
            allB = np.vstack((allB, aB))
            nCstrCum.append(nCstrCum[-1]+nCstr[-1])
            nCstr.append(aA.shape[0])
            allRes.append(None)
    
    #allANorm = np.sqrt(np.sum(np.square(allA), 1))
    #allANorm.resize((allANorm.size,1))
    
    #allB = np.divide(allB, allANorm)
    #allA = np.divide(allA, allANorm)
    
    L,N = allA.shape
    
    Atilde = np.hstack(( np.ones((L,1)), allA ))
         
    obj = matrix(0.0, (N+1,1))
    obj[0] = -1.

    solLP = solvers.lp( obj, G=sparse(matrix(Atilde)), h=matrix(allB) )
    c = np.array(solLP['x'][1:]).reshape((-1,1))
    
    btilde = allB.copy() # preserve
    btilde = btilde - np.dot(allA,c) # polytope A*x <= b now includes the origin
    # obtain dual polytope vertices: Simply use min max in each direction as approx
    D = np.divide(allA , np.tile(btilde,[1, N]))
   
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for k in range(len(args)):
        thisCstr = list(range(nCstrCum[k], nCstrCum[k]+nCstr[k]))
        
        thisTasks = list(zip(thisCstr, nCstr[k]*[D]))
        
        if glbDBG:
            allRes[k] = np.array(list(map(closestinitPointsPoint, thisTasks)))
        else:
            allRes[k] = np.array(list(thisWorkers.map(closestPoint, thisTasks)))
        
        ax.scatter(np.array(thisCstr), allRes[k], c=thisColor[k], edgecolor='none')
    
    return allRes,ax
        
####################################################
def plotDeriv( aDynSys, aEllip, x, x0=None, mode="OO", dim = [0,1], ax=None, doQuiver=True ):
    
    #assert isinstance(aDynSys, )
    
    if (x0 is None):
       x0 = np.zeros((x.shape[0],1))
    
    xd = aDynSys.getBestF(aEllip, x0, x, mode)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    #ax.scatter(x[dim[0],:], x[dim[1],:], 'k', edgecolor="face")
    if doQuiver:
        ax.quiver(x[dim[0],:], x[dim[1],:], xd[dim[0],:], xd[dim[1],:])
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        nmax = 0.5*min( xlim[1]-xlim[0], ylim[1]-ylim[0] )
        nn = colWiseNorm(xd)
        xd = xd*nmax/nn
        
        for k in range(x.shape[1]):
            ax.plot( [x[dim[0],k], x[dim[0],k]+xd[dim[0],k]], [x[dim[1],k], x[dim[1],k]+xd[dim[1],k]], 'k' )
            ax.plot( [x[dim[0],k]+xd[dim[0],k]], [x[dim[1],k]+xd[dim[1],k]], 'ok' )
    
    return 0
                                                              
    
    
    
    
    
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    