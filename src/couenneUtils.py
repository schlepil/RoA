from coreUtils import *

import subprocess
import re

import string

allVarString = list(string.ascii_lowercase)
for k in range(3):
    allVarStringOld = dp(allVarString)
    allVarString = []
    for oldString in allVarStringOld:
        for letter in string.ascii_lowercase:
            allVarString.append( oldString+letter )
del allVarStringOld #No longer needed

class couenneInst:
    def __init__(self):
        self.calcFolder = '/home/elfuius/tmpRam/'
        try:
            with open(self.calcFolder+'couenneNbr.txt', 'r+') as fId:
                self.Id = int(fId.readline())+1
                fId.seek(0)
                fId.write('{0:d}'.format(self.Id))
        except:
            with open(self.calcFolder+'couenneNbr.txt', 'w') as fId:
                self.Id = 0
                fId.write('{0:d}'.format(self.Id))
        
        self.calcFolder = '/home/elfuius/tmpRam/{0:d}/'.format(self.Id)
        
        subprocess.call(["mkdir", "-p", self.calcFolder])
        
        self.outName = 'couenneProb.txt'
        self.amplStub = 'stub.nl'
        self.solF = 'out.txt'
        self.solSTD = 'outstd.txt'
        
        self.runStr = 'cd {4}; ampl -ogstub {0}; couenne -o{1} {2} > {3};'
        
        self.progStr = ''
        
        self.objective = [None, None]
        self.solTime = None
        self.varsName = [];
        self.varsVal = [];
        self.cstrName = [];
        
        self.mode = 'MIN'
    
    def __del__(self):
        subprocess.call(["rm", "-rf", self.calcFolder])
        if self.Id > 100000:
            with open(self.calcFolder+'couenneNbr.txt', 'r+') as fId:
                fId.write('0')
    
    def addVar(self):
        self.varsName.append( allVarString[len(self.varsName)] )
        self.progStr += 'var {0};\n'.format( self.varsName[-1] )
        self.varsVal.append(None)
        return 0
    
    def addObj(self, P, L, c=0):
        assert P.shape[1] == len(self.varsName)
        
        self.cstrName.append( (allVarString[len(self.cstrName)]+'Q') )
        thisStr = 'minimize objective: '
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if np.abs(P[i,j]) > 1e-10:
                    thisStr += '+{0:.16f}*{1}*{2}'.format( P[i,j], self.varsName[i], self.varsName[j] )
                
        L = L.squeeze()
        for i in range(L.shape[0]):
            if np.abs(L[i])>1e-10:
                thisStr += '+{0:.16f}*{1}'.format( L[i], self.varsName[i] )
        
        thisStr += '+{0};\n'.format(c)
        
        self.progStr += thisStr
        return thisStr
    
    def addObjStr(self, objString, seek='MIN'):
        self.mode = seek
        
        if seek == 'MIN':
            thisStr = 'minimize objective: '+ objString + ';\n'
        else:
            thisStr = 'minimize objective: -1.*('+ objString + ');\n'
        
        self.progStr += thisStr
        return thisStr
        
    
    def addObjMono(self, coefs, seek):
        
        self.mode = seek
        
        coefs = np.array(coefs).squeeze()
        assert coefs.shape[0] == len(self.varsName)
        
        self.cstrName.append( (allVarString[len(self.cstrName)]+'Q') )
        if seek == 'MIN':
            thisStr = 'minimize objective: 1.'
        elif seek == 'MAX':
            thisStr = 'minimize objective: -1.'
        else:
            print("Fail")
            assert 0
        
        for i in range(coefs.shape[0]):
            for j in range(coefs[i]):
                thisStr += '*{0}'.format(self.varsName[i])
        
        thisStr += ';\n'
        
        self.progStr += thisStr
        return thisStr
    
    def addPolyObjasDict(self, polyDict, seek='MIN'):
        self.mode = seek
        
        self.cstrName.append( (allVarString[len(self.cstrName)]+'Poly') )
        if seek == 'MIN':
            thisStr = 'minimize objective: 1.*('
        elif seek == 'MAX':
            thisStr = 'minimize objective: -1.*('
        else:
            print("Fail")
            assert 0
        
        for degree, coef in polyDict.items():
            if isinstance(degree, str):
                degree = str2deg(degree)
                thisStr += '+{0:.64f}'.format(float(coef))
                for k in range(len(self.varsName)):
                    for i in range(degree[k]):
                        thisStr += '*{0}'.format(self.varsName[k])
        
        thisStr += ');\n'
        
        self.progStr += thisStr
        return thisStr            
    
    def addPolyObj(self, P, L, c=0):
        #Polynomial objective of the form 
        #[x; vec(x.x')].P.[x; vec(x.x')] + L'.[x; vec(x.x')]
        polyList = dp(self.varsName)
        for i in range(len(self.varsName)):
            for j in  range(i,len(self.varsName)):
                polyList.append(self.varsName[i]+'*'+self.varsName[j])
        
        assert P.shape[1] == len(polyList)
        
        self.cstrName.append( (allVarString[len(self.cstrName)]+'Q') )
        thisStr = 'minimize objective: '
        
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if np.abs(P[i,j]) > 1e-10:
                    thisStr += '+{0:.16f}*{1}*{2}'.format( P[i,j], polyList[i], polyList[j] )
        
        L = L.squeeze()
        for i in range(L.shape[0]):
            if np.abs(L[i])>1e-10:
                thisStr += '+{0:.16f}*{1}'.format( L[i], polyList[i] )
        
        thisStr += '+{0:.16f};\n'.format(c)
        
        self.progStr += thisStr
        return thisStr
    
        
        
    
    def addLinCstr(self, A, b):
        
        assert A.shape[1] == len(self.varsName)
        b = b.reshape((-1,1))
        
        for i in range(A.shape[0]):
            self.cstrName.append( (allVarString[len(self.cstrName)]+'L') )
            thisStr = 'subject to {0}: '.format( self.cstrName[-1] )
            for j in range(A.shape[1]): 
                if np.abs(A[i,j]) > 1e-10:
                    thisStr += '+{0:.16f}*{1}'.format( A[i,j], self.varsName[j])
            thisStr += '<={0:.16f};\n'.format(float(b[i]))
            self.progStr += thisStr
        return thisStr
    
    def addQuadCstr(self, P, L, c):
        assert P.shape[1] == len(self.varsName)
        
        self.cstrName.append( (allVarString[len(self.cstrName)]+'Q') )
        thisStr = 'subject to {0}: '.format( self.cstrName[-1] )
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if np.abs(P[i,j]) > 1e-10:
                    thisStr += '+{0:.16f}*{1}*{2}'.format( P[i,j], self.varsName[i], self.varsName[j] )
                
        L = L.squeeze()
        for i in range(L.shape[0]):
            if np.abs(L[i])>1e-10:
                thisStr += '+{0:.16f}*{1}'.format( L[i], self.varsName[i] )
        
        thisStr += '<={0:.16f};\n'.format(float(c))
        
        self.progStr += thisStr
        return thisStr
    
    def addPolyCstr(self, P, L, c=0.0):
        #Polynomial objective of the form 
        #[x; vec(x.x')].P.[x; vec(x.x')] + L'.[x; vec(x.x')]
        polyList = dp(self.varsName)
        for i in range(len(self.varsName)):
            for j in  range(i,len(self.varsName)):
                polyList.append(self.varsName[i]+'*'+self.varsName[j])
        
        assert P.shape[1] == len(polyList)
        
        self.cstrName.append( (allVarString[len(self.cstrName)]+'P') )
        thisStr = 'subject to {0}: '.format(self.cstrName[-1])
        
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if np.abs(P[i,j]) > 1e-10:
                    thisStr += '+{0:.16f}*{1}*{2}'.format( P[i,j], polyList[i], polyList[j] )
        
        L = L.squeeze()
        for i in range(L.shape[0]):
            if np.abs(L[i])>1e-10:
                thisStr += '+{0:.16f}*{1}'.format( L[i], polyList[i] )
        
        thisStr += '<={0:.16f};\n'.format(float(c))
        
        self.progStr += thisStr
        return thisStr
    
    
    def solve(self):
        
        with open(self.calcFolder + self.outName, 'w') as f:
            f.write(self.progStr);
        
        proc = subprocess.Popen(self.runStr.format(self.outName, self.solF, self.amplStub, self.solSTD, self.calcFolder), shell=True)
        proc.wait()
        
        for aStr in open(self.calcFolder + self.solSTD, 'r').readlines():
            lower = re.findall("Lower bound:(.*?)\n", aStr)
            if lower:
                self.objective[0] = float(lower[0])
            
            upper = re.findall("Upper bound:(.*?) \(gap", aStr)
            if upper:
                self.objective[1] = float(upper[0])
            solTime = re.findall("Total solve time:(.*?)s", aStr)
            if solTime:
                self.solTime = float(solTime[0])
        
        solList = []
        for aLine in open(self.calcFolder + self.solF, 'r').readlines():
            solList.append(aLine)
        
        for k in range(len(self.varsVal)):
            self.varsVal[k] = float(solList[len(solList)-1-len(self.varsVal)+k])
        
        if self.mode == 'MIN':
            obj = dp(self.objective)
        elif self.mode == 'MAX':
            obj =[]
            for aObj in self.objective:
                obj.append( -1.*aObj )
        else:
            assert 0
        
        return obj, dp(self.varsVal), dp(self.solTime)
