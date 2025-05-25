#Unitary Dual Descriptor Scalar class
#Author: Bin-Guang Ma; Date: 2006-6-19

# The program is provided as it is and without warranty of any kind, 
# either expressed or implied.

# Running in Python IDLE;
# Python version 2.3 or higher is required which can be downloaded
# from: http://www.python.org


from copy import copy
from math import cos,pi,fabs,ceil
from random import random
from time import clock


def get_all_permustrs(charset,l):
    base=len(charset)
    apstr=''
    for i in range(base**l):
        num=i
        for j in range(l-1,-1,-1):
            apstr+=charset[num/base**j]
            num%=base**j
    aplst=[]
    for i in range(0,len(apstr),l):
        aplst.append(apstr[i:i+l])
    return aplst


##Generate a list which consists of the expanded items of position weight function
def getpwflst(pwf):
    '''argument format: integer or tuple ('expression',itemnum,mode)
       expression: should be the form f(x,k) in which k is a parameter
       and will be replaced by item index;
       itemnum: the number of items for expanded series
       mode: should be 'order' (or 0), or 'random' (or 1)
    '''
    if isinstance(pwf,int) and pwf>0:
        pwflst=[]
        for i in range(2,pwf+2):
            exec 'def bf%s(x): f=cos(2*pi*x/%s); return f'%(str(i),str(i))
            pwflst.append([i,eval('bf%s'%str(i))])
    elif isinstance(pwf,tuple):# form ('expresion(i,x)',order,mode)
        pwflst=[]
        if pwf[2] in ('order',0):
            for i in range(2,pwf[1]+2):
                exec 'def bf%s(x): f='%str(i)+pwf[0].replace('k',str(i))+'; return f'
                pwflst.append([i,eval('bf%s'%str(i))])
        if pwf[2] in ('random',1):
            for i in range(2,pwf[1]+2):
                exec 'def bf%s(x): f='%str(i)+pwf[0].replace('k',str(i))+'; return f'
                pwflst.append([pwf[1]*random(),eval('bf%s'%str(i))])
    return pwflst
        
def upscale(seqs,cwm,pwf,step=1,dist=1):
    '''For multi-scaled analysis'''
    rank=len(cwm.keys()[0])    
    flst=[]
    for s in seqs:
        L=len(s)
        L-=int(ceil(float(rank)/step)*step)
        f=0.0
        if callable(pwf):
            for k in range(0,L,step):
                f=cwm[s[k:k+rank]]*pwf(k/step)
                flst.append(f/L/float)
        elif isinstance(pwf,str):
            pwf=lambda x: eval(pwf)
            for k in range(0,L,step):
                f=cwm[s[k:k+rank]]*pwf(k/step)
                flst.append(f/L/float)
        elif isinstance(pwf,list) and isinstance(pwf[0],tuple):
                Ik=0.0
                for i in range(order):
                    Ik+=pwf[i][0]*pwf[i][1](k/step)
                f+=cwm[s[k:k+rank]]*Ik
                flst.append(f/(L/float(step)))
    return flst


class uDDs:
    'The unitary Dual Descriptor scalar class'
    def __init__(self,cwm={},pwf=9):
        self.rank=1
        self.order=9
        self.charset=''
        self.defgetcwf='order'
        self.defgetpwc='order'
        self.defprdfun='cos'
        self.data=[]
        
        #construct composition weight map
        if isinstance(cwm,dict):
            self.cwm=cwm
            self.charset=self.cwm.keys()
            self.charset.sort()
            if self.charset!=[]:
                if len(self.charset)>=2:
                    self.rank=len(self.charset[1])
                else :
                    print 'there should be at least two items in cwm'
                    exit()
        elif isinstance(cwm,tuple):#form (charset,rank,mode)
            self.cwm={}
            if cwm[1]==1:
                if cwm[2] in(0,'order'):
                    for i in range(len(cwm[0])):
                        self.cwm[cwm[0][i]]=i+1
                elif cwm[2] in (1,'random'):
                    for i in cwm[0]:
                        self.cwm[i]=random()*len(cwm[0])
            elif isinstance(cwm[1],int) and cwm[1]>1:
                self.rank=cwm[1]
                aplst=get_all_permustrs(cwm[0],cwm[1])
                if cwm[2] in (0,'order'):                    
                    for i in range(len(aplst)):
                        self.cwm[aplst[i]]=i+1
                elif cwm[2] in (1,'random'):
                    for i in aplst:
                        self.cwm[i]=random()*len(aplst)
            self.charset=self.cwm.keys()
            self.charset.sort()
            if self.charset!=[]:
                if len(self.charset)>=2:
                    self.rank=len(self.charset[1])
                else :
                    print 'there should be at least two items in CWM'
                    exit()
        else :
            print 'improper arguments for CWM'
            print 'should be a tuple or dictionary'

        #construct position weight function
        if isinstance(pwf,list):# form [[coef,basefun],...]
            self.pwf=pwf
            self.order=len(pwf)
        elif isinstance(pwf, int) and pwf > 0:
            self.order=pwf
            self.pwf=getpwflst(pwf)            
        elif isinstance(pwf,tuple):# form ('expresion(i,x)',order,mode)
            self.order=pwf[1]
            self.defgetpwc=pwf[2]
            self.pwf=getpwflst(pwf)
        else :
            print 'improper argument for PWF'
            print 'should be a list of the form [[coef,basefun],...]'
            print 'or a integer to designate the order of series'
            print 'or a tuple of the form (expression,order,mode)'

    def __del__(self):
        pass
        
    def study(self,seqs,step=1,mode=1):
        '''Implementation of alternate training Process
           Arguments: seqs, the character sequences used for training which
           can be one sequence or multiple sequences;
           step, the step for pattern description, default value is 1;
           mode, the training mode, totally 5 training modes 
        '''
        #construct the data set for training
        self.data=[]
        if isinstance(seqs,str):
            self.data.append((1.0,seqs))
        elif isinstance(seqs,list) and seqs!=[] and isinstance(seqs[0],str):
            for i in seqs:
                self.data.append((1.0,i))
        elif isinstance(seqs,list) and seqs!=[] and isinstance(seqs[0],tuple):
            for i in seqs:
                self.data.append((i[0],i[1]))
        elif isinstance(seqs,tuple):
            if isinstance(seqs[1],str):
                self.data.append((seqs[0],seqs[1]))
            elif isinstance(seqs[1],list):
                for i in seqs[1]:
                    self.data.append((seqs[0],i))                
        else :
            print 'improper data for training descriptor'
            print 'should be a string of a tuple of (target, string)'
            print 'or a list of tuples: [(target,string),...]'
        #construct CWM when it's empty
        if self.cwm=={}:
            strs=''.join(map(lambda x:x[1],self.data)) #future problem
            for i in strs:
                self.cwm[i]=1
            self.charset=self.cwm.keys()
            self.charset.sort()
            if self.defgetcwf=='order':                
                i=1
                for j in self.charset:
                    self.cwm[j]=i
                    i+=1                
            elif self.defgetcwf=='random':
                for j in self.charset:
                    self.cwm[j]=random()*len(self.charset)
        #reconstruct PWF if default config is updated
        if self.defprdfun!='cos':
            self.pwf=getpwflst(('x%k',self.order,self.defgetpwc))
        if self.defgetpwc!='order':            
            if self.defprdfun=='cos':
                self.pwf=getpwflst(('cos(2*pi*x/k)',self.order,'random'))
            elif self.defprdfun=='mod':
                self.pwf=getpwflst(('x%k',self.order,'random'))
        #generate position weight function value table to quicken running speed
        self.pwf_table()
        #study begins
        D0=10.0**308 #DBL_MAX
        #choose study mode
        if mode in ('once_from_c', 1):
            print 'The initial value of Pattern Deviation Function is:'
            print self.get_D(self.data,step)
            self.minD_get_p(self.data,step)
            print 'after once study, the value of PDF is:'
            print self.get_D(self.data,step)
        elif mode in ('once_from_p',2):
            print 'The initial value of Pattern Deviation Function is:'
            print self.get_D(self.data,step)
            self.minD_get_c(self.data,step)
            print 'after once study, the value of PDF is:'
            print self.get_D(self.data,step)
        elif mode in ('alte_from_c',3):
            print 'The initial value of Pattern Deviation Function is:'
            D=self.get_D(self.data,step)
            print D
            print 'The initial parameters of Dual Descriptor'
            self.show('parameter')
            print 'now begin alternant study'
            print '.........................'
            i=0
            start=clock()
            while D0>D :
                D0=D
                oldcwm=copy(self.cwm)
		oldpwf=copy(self.pwf)
		i+=1
		print 'after the %d th round of study, the parameter values are:'%i
		self.minD_get_p(self.data,step)
		self.minD_get_c(self.data,step)
		print 'after the %d th round of study, the PDF value is:'%i
		D=self.get_D(self.data,step)
		print D
	    self.cwm=copy(oldcwm)
	    self.pwf=copy(oldpwf)
	    print'after the %d th round of study, the minimum D value is:'%(i-1)
	    print 'D=',D0
	    end=clock()
	    print 'the time used: %3.3f seconds'%(end-start)
	elif mode in ('alte_from_p',4):
            print 'The initial value of Pattern Deviation Function is:'
            D=self.get_D(self.data,step)
            print D
            print 'The initial parameters of Dual Descriptor'
            self.show('parameter')
            print 'now begin alternant study'
            print '.........................'
            i=0
            start=clock()
            while D0>D :
                D0=D
                oldcwm=copy(self.cwm)
		oldpwf=copy(self.pwf)
		i+=1
		print 'after the %d th round of study, the parameter values are:'%i
		self.minD_get_c(self.data,step)
		self.minD_get_p(self.data,step)		
		print 'after the %d th round of study, the PDF value is:'%i
		D=self.get_D(self.data,step)
		print D
	    self.cwm=copy(oldcwm)
	    self.pwf=copy(oldpwf)
	    print'after the %d th study, the minimum D value is:'%(i-1)
	    print 'D=',D0
	    end=clock()
	    print 'the time used: %3.3f seconds'%(end-start)
	elif isinstance(mode,str) and  mode.isdigit():
            print 'The initial value of Pattern Deviation Function is:'
            D=self.get_D(self.data,step)
            print D
            print 'The initial parameters of Dual Descriptor'
            self.show('parameter')
            print 'now begin alternant study'
            print '.........................'
            i=0
            start=clock()
            for j in range(int(mode)):
		i+=1
		print 'after the %d th round of study, the parameter values are:'%i
		self.minD_get_p(self.data,step)
		self.minD_get_c(self.data,step)
		print 'after the %d th round of study, the PDF value is:'%i
		D=self.get_D(self.data,step)
		print D
	    end=clock()
	    print 'the time used: %3.3f seconds'%(end-start)
	else : print 'No such study mode!'	
            
    def get_d(self,aseq,step):
        c=aseq[0]
        #c = target, the value of target pattern
        L=len(aseq[1])
        L-=int(ceil(float(self.rank)/step)*step)
        d=0.0
        for k in range(0,L,step):
            Ik=0.0
            for i in range(self.order):
                Ik+=self.pwf[i][0]*self.pwf[i][1][k/step%(i+2)]
            d+=pow(self.cwm[aseq[1][k:k+self.rank]]*Ik-c,2.0)
	return d/(L/float(step))

    def get_d_normed(self,aseq,step):
        c=aseq[0]
        L=len(aseq[1])
        L-=int(ceil(float(self.rank)/step)*step)
        d=0.0
        for k in range(0,L,step):
            Ik=0.0
            for i in range(self.order):
                Ik+=self.pwf[i][0]*self.pwf[i][1][k/step%(i+2)]
            d+=pow(self.cwm[aseq[1][k:k+self.rank]]*Ik-c,2.0)
	return d/(L/float(step))/float(c*c)
    
    def get_d_list(self,seqs,step):
        dlst=[]
        for i in seqs:
            dlst.append(self.get_d(i,step))
        return dlst

    def get_D(self,seqs,step):
        D=0.0
        for i in seqs:            
            D+=self.get_d(i,step)
        return D/len(seqs)

    def get_pdf_value(self,seqs,step):        
        if isinstance(seqs,str):
            L=len(seqs)
            L-=int(ceil(float(self.rank)/step)*step)
            f=0.0
            for k in range(0,L,step):
                Ik=0.0
                for i in range(self.order):
                    Ik+=self.pwf[i][0]*self.pwf[i][1][k/step%(i+2)]
                f+=self.cwm[seqs[k:k+self.rank]]*Ik                
            return f/(L/float(step))
        elif isinstance(seqs,list):
            flst=[]
            for s in seqs:
                L=len(s)
                L-=int(ceil(float(self.rank)/step)*step)
                f=0.0
                for k in range(0,L,step):
                    Ik=0.0
                    for i in range(self.order):
                        Ik+=self.pwf[i][0]*self.pwf[i][1][k/step%(i+2)]
                    f+=self.cwm[s[k:k+self.rank]]*Ik
                flst.append(f/(L/float(step)))
            return flst
        else : print 'improper arguments for seqs.'

    def minD_get_p(self,seqs,step):
        '''Get the coefficients of position weight function (PWC)
           by minimizing the value D of pattern deviation function (PDF)
        '''
        u=range(self.order*self.order)
        v=range(self.order)
        for i in u:
            u[i]=0.0
        for i in v:
            v[i]=0.0
        for s in seqs:
            c=s[0]
            L=len(s[1])
            L-=int(ceil(float(self.rank)/step)*step)
            for k in range(0,L,step):                
                xk=self.cwm[s[1][k:k+self.rank]]
                for j1 in range(self.order):
                    for j2 in range(self.order):
                        u[j1*self.order+j2]+=xk*self.pwf[j1][1][k/step%(j1+2)]*xk*self.pwf[j2][1][k/step%(j2+2)]
		    v[j1]+=c*xk*self.pwf[j1][1][k/step%(j1+2)]
	self.Agaus(u,v)
	for i in range(self.order):
            self.pwf[i][0]=v[i]
            print 'a%d=%1.10f'%(i+1,self.pwf[i][0])
            
    def minD_get_c(self,seqs,step):
        '''Get the composition weight factors (CWF) of the
           composition weight map (CWM) by minimizing D       
        '''
        Idict=copy(self.cwm)
        for i in Idict:
            Idict[i]=[1.0,1.0] #
	for s in seqs:
            c=s[0]
            L=len(s[1])
            L-=int(ceil(float(self.rank)/step)*step)
            for k in range(0,L,step):
                Ik=0.0
                for j in range(self.order):
                    Ik+=self.pwf[j][0]*self.pwf[j][1][k/step%(j+2)]
                Idict[s[1][k:k+self.rank]][0]+=c*Ik
                Idict[s[1][k:k+self.rank]][1]+=Ik*Ik
        for i in self.charset:
            self.cwm[i]=Idict[i][0]/Idict[i][1]
            print 'x%s=%1.10f'%(i,self.cwm[i])

            
    def show(self,what='parameter'):
        '''Show parameter values of the DD'''
        if what in ('parameter',0):            
            print 'DD Parameters are:'
            for i in self.charset:
                print 'x%s=%1.10f'%(i,self.cwm[i])
            for i in range(self.order):
                print 'a%d=%1.10f'%(i+1,self.pwf[i][0])
        elif what in ('state',1):
            print 'State variables are:'
            print 'rank=',self.rank
            print 'order=',self.order
            print 'defgetcwf=',self.defgetcwf
            print 'defgetpwc=',self.defgetpwc
            print 'defprdfun=',self.defprdfun
            print 'charset=',self.charset
        elif what in ('data',2):
            print 'Data are:'
            print self.data
        elif what in ('all',3):
            print 'State variables are:'
            print 'rank=',self.rank
            print 'order=',self.order
            print 'defgetcwf=',self.defgetcwf
            print 'defgetpwc=',self.defgetpwc
            print 'defprdfun=',self.defprdfun
            print 'charset=',self.charset
            print 'DD Parameters are:'
            for i in self.charset:
                print 'x%s=%1.10f'%(i,self.cwm[i])
            for i in range(self.order):
                print 'a%d=%1.10f'%(i+1,self.pwf[i][0])
            print 'Data are:'
            print self.data
        else : print 'improper arguments'

    def help(self):
        print 'Help for unitary Dual Descriptor scalar class.'
        print 'Author: Bin-Guang Ma; Date: 2006-6-19;'
        print 'All right reserved'
        print 'Members:...................'
        print 'properties:'
        print 'rank: the rank for DD which means how many char in a combination'
        print 'order: the number of expanded items of position weight funciton'
        print 'charset: the characters in composition weight map'
        print ''
      

    def get_fwpw(self,aseq,step):
        wfdict=copy(self.cwm)
        for i in wfdict:
            wfdict[i]=0.0            
        L=len(aseq)
	for k in range(0,L,step):
            Ik=0
            for i in range(self.order):
                Ik+=self.pwf[i][0]*self.pwf[i][1][k/step%i]
	    wfdict[aseq[k:k+self.rank]]+=Ik
	return wfdict
    
    def get_dvs(self,aseq,step):
	#use normalized composition weight factor
	cwfvalues=self.cwm.values()
	mincwf=min(cwfvalues)
	maxcwf=max(cwfvalues)
	sumcwf=sum(cwfvalues)
	meancwf=sumcwf/len(self.cwm)
	rangecwf=maxcwf-mincwf
	Ncwm=copy(self.cwm)
	for i in Ncwm:
            Ncwm[i]=(Ncwm[i]-meancwf)/rangecwf
        L=len(aseq)
	dvs=[]
	dv=0.0
	for k in range(0,L,step):
            dv+=Ncwm[aseq[k:k+self.rank]]
            dvs.append(dv)
	return dvs  
        

    def set_cwm(self,cwdict):         
        if isinstance(cwdict,dict):
            self.cwm=cwdict
        else : print 'improper argument!'

    def get_cwm(self):
        return self.cwm


    def set_pwf(self,mdlist):
        if isinstance(mdlist,list):
            self.pwf=mdlist
        else : print 'improper argument!'

    def get_pwf(self):
        return self.pwf

    def set_defgetcwf(self,mode):        
        if mode in ('order',0):
            self.defgetcwf='order'
        elif mode in ('random',1):
            self.defgetcwf='random'
        else: print 'improper argument!'

    def get_defgetcwf(self):
        return self.defgetcwf

    def set_defgetpwc(self,mode):
        if mode in ('order',0):
            self.defgetpwc='order'
        elif mode in ('random',1):
            self.defgetpwc='random'
        else: print 'improper argument!'

    def get_defgetpwc(self):
        return self.defgetpwc        

    def set_defprdfun(self,fun):
        if fun in ('cos',0):
            self.defprdfun='cos'
        elif fun in ('mod',1):
            self.defprdfun='mod'
        else: print "improper argument! should be 'cos' or 'mod'"

    def get_defprdfun(self):
        return self.defprdfun

    def pwf_table(self):
        '''Generate a value table for the current PWF'''
        for i in range(2, self.order+2):
            fvt = []
            for j in range(i):
                fvt.append(self.pwf[i-2][1](j))
            self.pwf[i-2][1] = fvt        
    
    def Agaus(self,a,b):
        #use Gauss method to solve the linear equations
        n=len(b)
        js=range(n)
        l=1 #is->ii
        for k in range(0,n-1):#
            d=0.0
            for i in range(k,n):#
                for j in range(k,n):#                    
                    t=fabs(a[i*n+j])
                    if t>d :
                        d=t; js[k]=j; ii=i
            if d+1.0==1.0 :
                l=0
            else :
                if js[k]!=k :
                    for i in range(n):#
                        p=i*n+k; q=i*n+js[k]
                        t=a[p]; a[p]=a[q]; a[q]=t
                if ii!=k :
                    for j in range(k,n):#
                        p=k*n+j; q=ii*n+j
                        t=a[p]; a[p]=a[q]; a[q]=t
                    t=b[k]; b[k]=b[ii]; b[ii]=t
            if l==0 :
                del js; print 'fail 1'
                return 0
            d=a[k*n+k]
            for j in range(k+1,n):#
                p=k*n+j; a[p]/=d
            b[k]/=d
            for i in range(k+1,n):#
                for j in range(k+1,n):#
                    p=i*n+j
                    a[p]-=a[i*n+k]*a[k*n+j]
                b[i]-=a[i*n+k]*b[k]      
        d=a[(n-1)*n+n-1]
        if fabs(d)+1.0==1.0 :
            del js; print 'fail 2'
            return(0)
        b[n-1]/=d
        for i in range(n-2,-1,-1):#
            t=0.0
            for j in range(i+1,n):#
                t+=a[i*n+j]*b[j]
            b[i]-=t
        js[n-1]=n-1
        for k in range(n-1,-1,-1):#
            if js[k]!=k :
                t=b[k]; b[k]=b[js[k]]; b[js[k]]=t
        del js
        return 1
