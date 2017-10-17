from coreUtils import *

import cvxpy


#ALIAS
det = np.linalg.det
chol = np.linalg.cholesky
inv = np.linalg.inv

#########################################################

def smallestCircumScribedEllip(ptS, P=None, obj=None, maxChange=[None,None], Pold=None, **kwargs):
    #ptS holds the points columnwise 
    #mindCondition < lambda_max/lambda_min
    opts = {'minCondition':1.e3}
    opts.update(kwargs)
    
    Pold = (Pold.T+Pold)/2.
    
    dim = ptS.shape[0]
    
    if P is None:
        P = cvxpy.Semidef(dim,"P")
    
    cstr = [ cvxpy.quad_form(aPt, P) <= 1. for aPt in ptS.T ]
    
    eigMax = np.max(np.linalg.eigh(Pold)[0])
    
    #restrict changement
    zeta = cvxpy.Variable(1)
    if not(Pold is None) and not(maxChange[0] is None):
        cstr.append( eigMax*maxChange[0]*np.identity(dim) < P-zeta*Pold )
    if not(Pold is None) and not(maxChange[1] is None):
        cstr.append( P-zeta*Pold < eigMax*maxChange[1]*np.identity(dim) )
    
    if not(opts['minCondition'] is None):
        cstr.append( cvxpy.lambda_max(P) < opts['minCondition']*cvxpy.lambda_min(P) )
    
    if obj is None:
        obj = cvxpy.Maximize( cvxpy.log_det(P) )
    
    prob = cvxpy.Problem(obj, cstr)

    #prob.solve();
    prob.solve(solver='CVXOPT', verbose=False, kktsolver=cvxpy.ROBUST_KKTSOLVER) 
    
    assert prob.status=='optimal', "Failed to solve circumscribed ellip prob"

    Pval = np.array(P.value)
    
    return Pval

#########################################################

def getLargestInnerEllip(pt, maxChange = [None, None], Pold=None, optsIn={}):
    #Opts
    opts = {'conv':1e-2}
    opts.update(optsIn)
    
    #There are faster ways to do this, but this is very convenient
    dim,Npt=pt.shape
    
    if Pold is None:
        Pval = np.identity(dim)
    else:
        Pval = Pold
    
    oldDet = 2*det(Pval)
    
    #To avoid constant recreation of vars
    P = cvxpy.Semidef(pt.shape[0],"P")
    obj = cvxpy.Maximize( cvxpy.log_det(P) )

    while( (np.abs(oldDet-det(Pval))/oldDet)>opts['conv'] ):
        oldDet = det(Pval)
        
        CpreTrans = chol(Pval).T
        CpreTransI = inv(CpreTrans)
        
        ptPre = ndot(CpreTrans, pt)
        normPtsq = colWiseSquaredNorm(ptPre)
        ptI = np.divide( ptPre, np.tile(normPtsq,(dim,1)) )
        #Adapt old
        PoldInvA = inv(ndot(CpreTransI.T, Pold, CpreTransI))
        ptInM = np.max(colWiseSquaredKerNorm(PoldInvA,ptI))
        PoldInvA = PoldInvA/ptInM
        #Solve
        PpreInv = smallestCircumScribedEllip(ptI, P=P, obj=obj, maxChange=maxChange, Pold=PoldInvA)
        Ppre = inv(PpreInv)
        Pval = ndot(CpreTrans.T, Ppre, CpreTrans)
    
    thisAlpha = ((det(Pval))**(1./float(dim)))
    thisP = Pval/thisAlpha
    
    return [thisP, thisAlpha]