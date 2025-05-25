//************************************************************//
//**                        DDs.cpp                         **//
//************************************************************//
/*                对偶描述子标量类的实现文件                  // 
//            作者：马彬广        2003年11月18日              //
***************************************************************/


#include"DDs.h"


//*****************成员函数及其说明*******************//

/************************************************
//                                              *
  参数：cwfs 欲设置的组成权重因子数组；         *
        elmn 组成权重因子的个数；               *
  返回值：无；                                  *
  功能：设置组成权重因子的值；                  *
//                                              *
************************************************/
void DDs::set_cwfs(double cwfs[], int elmn)
{
	for(unsigned int i=0;i<D_elmn;i++) D_cwfs[i]=cwfs[i];
}


/************************************************
//                                              *
  参数：cwfs 存放获得的组成权重因子的数组；     *
        elmn 组成权重因子的个数；               *
  返回值：无；                                  *
  功能：获得组成权重因子的值；                  *
//                                              *
************************************************/
void DDs::get_cwfs(double cwfs[],int elmn)
{
	for(unsigned int i=0;i<D_elmn;i++) cwfs[i]=D_cwfs[i];	
}


/************************************************
//                                              *
  参数：pwfc 欲设置的位置权重函数系数的数组；   *
        rank 位权函数展开式的项数；             *
  返回值：无；                                  *
  功能：设置位置权重函数展开式的系数值          *
//                                              *
************************************************/
void DDs::set_pwfc(double pwfc[],int rank)
{
	for(unsigned int i=0;i<D_rank;i++) D_pwfc[i]=pwfc[i];
}


/************************************************
//                                              *
  参数：pwfc 存放获得的位置权重函数系数的数组； *
        rank 位权函数展开式的项数；             *
  返回值：无；                                  *
  功能：获得位置权重函数展开式的系数值；        *
//                                              *
************************************************/
void DDs::get_pwfc(double pwfc[],int rank)
{
	for(unsigned int i=0;i<D_rank;i++) pwfc[i]=D_pwfc[i];
}


/************************************************
//                                              *
  参数：bases[rank] 欲设置的位置权重函数的      * 
        基函数指针数组；                        *
        rank 位权函数展开式的项数；             *
  返回值：无；                                  *
  功能：设置位置权重函数的基函数;               *
//                                              *
************************************************/
void DDs::set_base(BaseFunc bases[], int rank)
{
	for(unsigned int i=0;i<D_rank;i++) D_pwfb[i]=bases[i];
}


/************************************************
//                                              *
  参数：bases[rank] 存放获得的位置权重函数的    * 
        基函数指针数组；                        *
        rank 位权函数展开式的项数；             *
  返回值：无；                                  *
  功能：获得位置权重函数的基函数;               *
//                                              *
************************************************/
void DDs::get_base(BaseFunc bases[], int rank)
{
	for(unsigned int i=0;i<D_rank;i++) bases[i]=D_pwfb[i];
}


/************************************************
//                                              *
  参数：as 欲提取特征的字符串（描述对象）；     *
        smode 学习模式，共取四个值：            *
		  1 给定组成权重因子，一次性学习；      *
		  2 给定位置权重函数，一次性学习；      *
		  3 从给定的组成权重因子进入交替式学习；*  
		  4 从给定的位置权重函数进入交替式学习；*
        step 描述的步长；                       *
  返回值：无；                                  *
  功能：在给定的学习模式和描述步长下，          *
        进入学习过程，以获得字符串的极佳描述；  *
//                                              *
************************************************/
void DDs::study(char* as, int smode, unsigned int step)
{
	//将单字符串转化成字符串数组
	char* temp[1];  
	temp[1]=new char[strlen(as)];
	strcpy(temp[1],as);
	temp[1][strlen(as)]='\0';
	//调用该类的另一重载函数
	study(temp,1,smode,step);	
}


/************************************************
//                                              *
  参数：asa 欲提取特征的字符串数组（描述对象）；*
        smode 学习模式，共取四个值：            *
		  1 给定组成权重因子，一次性学习；      *
		  2 给定位置权重函数，一次性学习；      *
		  3 从给定的组成权重因子进入交替式学习；*  
		  4 从给定的位置权重函数进入交替式学习；*
        step 描述的步长；                       *
  返回值：无；                                  *
  功能：在给定的学习模式和描述步长下，进入学习  *
        过程，以获得字符串数组的极佳描述；      *
//                                              *
************************************************/
void DDs::study(char* asa[], int n, int smode, unsigned int step)
{
	double D,D0=DBL_MAX;
	double* oldcwfs=new double[D_elmn];
	double* oldpwfc=new double[D_rank];
	clock_t start,end;
	unsigned int i=0,j=0;
	//根据学习模式选择学习方式
	switch(smode){
	case 1: 
		cout<<"模式偏离函数的初始值"<<endl;
		cout<<get_D(asa,n,step)<<endl;
		minD_get_p(asa,n,step);
		cout<<"一次学习后的模式偏离函数值"<<endl;
		cout<<get_D(asa,n,step)<<endl;
		break;
	case 2: 
		cout<<"模式偏离函数的初始值"<<endl;
		cout<<get_D(asa,n,step)<<endl;
		minD_get_c(asa,n,step);
		cout<<"一次学习后的模式偏离函数值"<<endl;
		cout<<get_D(asa,n,step)<<endl;
		break;
	case 3: 		
		cout<<"模式偏离函数的初始值\n";
		D=get_D(asa,n,step);
		cout<<D<<endl;
		cout<<"对偶描述子的初始参数"<<endl;
		print();
		printf("开始交替式学习\n");
		printf("。。。。。。。。。。。\n");	
		i=0;
		start=clock();
		while(D0>D){
			D0=D;
			for(j=0;j<D_elmn;j++) oldcwfs[j]=D_cwfs[j];
			for(j=0;j<D_rank;j++) oldpwfc[j]=D_pwfc[j];
			i++;
			printf("第 %d 次学习时的参数值\n",i);
			minD_get_p(asa,n,step);
			minD_get_c(asa,n,step);
			printf("第 %d 次学习后的D值\n",i);
			D=get_D(asa,n,step);
			cout<<D<<endl;
		}
		for(j=0;j<D_elmn;j++) D_cwfs[j]=oldcwfs[j];
		for(j=0;j<D_rank;j++) D_pwfc[j]=oldpwfc[j];
		printf("第 %d 次学习后得最小D值\n",i-1);
		cout<<D0<<endl;
		end=clock();
		printf("the time used: %3.3f seconds\n",(double)(end-start)/CLK_TCK);
		break;
	case 4:	
		cout<<"模式偏离函数的初始值\n";
		D=get_D(asa,n,step);
		cout<<D<<endl;
		cout<<"对偶描述子的初始参数"<<endl;
		print();
		printf("开始交替式学习\n");
		printf("。。。。。。。。。。。\n");	
		i=0;		
		start=clock();
		while(D0>D){
			D0=D;
			for(j=0;j<D_elmn;j++) oldcwfs[j]=D_cwfs[j];
			for(j=0;j<D_rank;j++) oldpwfc[j]=D_pwfc[j];
			i++;
			printf("第 %d 次学习时的参数值\n",i);
			minD_get_c(asa,n,step);
			minD_get_p(asa,n,step);			
			printf("第 %d 次学习后的D值\n",i);
			D=get_D(asa,n,step);
			cout<<D<<endl;
		}
		for(j=0;j<D_elmn;j++) D_cwfs[j]=oldcwfs[j];
		for(j=0;j<D_rank;j++) D_pwfc[j]=oldpwfc[j];
		printf("第 %d 次学习后得最小D值\n",i-1);
		cout<<D0<<endl;
		end=clock();
		printf("the time used: %3.3f seconds\n",(double)(end-start)/CLK_TCK);
		break;
	default:
		cout<<"该学习模式不存在"<<endl;
		break;
	}		
}


/************************************************
//                                              *
  参数：as 欲求取其模式偏离函数值的             *
        一个字符串；                            * 
        step 描述步长；                         *
  返回值：求得的模式偏离函数值；                *
  功能：在对偶描述子的当前状态下                *
        求给定字符串的模式偏离函数值；          *
//                                              *
************************************************/
double DDs::get_pdf_value(char* as, unsigned int step)
{
	return get_d(as,step);
}


/************************************************
//                                              *
  参数：asa 欲求取其模式偏离函数值的            *
        一个字符串数组；                        *
		n 字符串数组中的字符串个数              * 
        step 描述步长；                         *
  返回值：求得的模式偏离函数值；                *
  功能：在对偶描述子的当前状态下                *
        求给定字符串数组的模式偏离函数值；      *
//                                              *
************************************************/
double DDs::get_pdf_value(char* asa[], int n, unsigned int step)
{
	return get_D(asa,n,step);
}


/************************************************
//                                              *
  参数：as 欲求取其加权频率的一个字符串；       *
        step 描述步长；                         *
  返回值：求得的加权频率数组，该数组有          *
          D_elmn个元素，其中只有D_elmn-1个      *
		  是彼此独立的；                        *
  功能：在对偶描述子的当前状态下，求给定        * 
        字符串的加权频率；                      *
//                                              *
************************************************/
double* DDs::get_fpws(char* as, unsigned int step)
{
	unsigned int L=strlen(as);
	double* fpws=new double[D_elmn];
	for(unsigned int k=0;k<L;k+=step){
		double Ik=0;
		for(unsigned int i=0;i<D_rank;i++) Ik+=D_pwfc[i]*D_pwfb[i](k/step);
		char* temp=new char[D_order];
		strncpy(temp,as+k,D_order);
		temp[D_order]='\0';
		fpws[get_index(temp,D_charset)]+=Ik;
	}
	return fpws;
}


/************************************************
//                                              *
  参数：as 欲求取其对偶变量序列的一个字符串；   *
        step 描述步长；                         *
  返回值：求得的对偶变量序列（数组），该数组的  *
          元素个数等于给定字符串的长度；        *
  功能：在对偶描述子的当前状态下，求给定        * 
        字符串的对偶变量序列，可据此画出1维     *
		D曲线；                                 *
//                                              *
************************************************/
double* DDs::get_dvs(char* as, unsigned int step)
{
	unsigned j,k;
	//归一化后的组成权重因子
	double* N_cwfs=new double[D_elmn];
	double mincwf=D_cwfs[0];
	double maxcwf=D_cwfs[0];
	double sumcwf=D_cwfs[0];
	for(j=1;j<D_elmn;j++){
		if(D_cwfs[j]>maxcwf) maxcwf=D_cwfs[j];
		if(D_cwfs[j]<mincwf) mincwf=D_cwfs[j];
		sumcwf+=D_cwfs[j];
	}
	double meancwf=sumcwf/D_elmn;
	double rangecwf=maxcwf-mincwf;
	for(j=0;j<D_elmn;j++){
		N_cwfs[j]=(D_cwfs[j]-meancwf)/rangecwf;
	}
	unsigned int L=strlen(as);
	double* dvs=new double[L];
	double dv=0;
	for(k=0;k<L;k+=step){
		char* temp=new char[D_order];
		strncpy(temp,as+k,D_order);
		temp[D_order]='\0';
		dv+=N_cwfs[get_index(temp,D_charset)];
		dvs[k]=dv;
	}
	return dvs;
}


/************************************************
//                                              *
  参数：无；                                    *
  返回值：无；                                  *
  功能：输出对偶描述子在当前状态下的参数值：    *
        组成权重因子和位置权重函数展开式的系数；*
//                                              *
************************************************/
void DDs::print(){
	unsigned int i;
	for(i=0;i<D_elmn;i++){
		char* acwfs=new char[D_order];
		strncpy(acwfs,D_cwfstr+i*D_order,D_order);
		acwfs[D_order]='\0';
		printf("x%s=%1.10f\n",acwfs,D_cwfs[i]);
	}	
	for(i=0;i<D_rank;i++){
		printf("a%d=%1.10f\n",i+1,D_pwfc[i]);
	}
}
/************************************************
//                                              *
  参数：astr 欲求取其模式偏离函数值的           *
        一个字符串；                            * 
        step 描述步长；                         *
  返回值：求得的模式偏离函数值d；               *
  功能：在对偶描述子的当前状态下                *
        求给定字符串的模式偏离函数值；          *
//                                              *
************************************************/
double DDs::get_d(char* astr,unsigned int step)
{
	unsigned int L=strlen(astr);
	double d=0;
	for(unsigned int k=0;k<L;k+=step){
		double Ik=0;
		for(unsigned int i=0;i<D_rank;i++) Ik+=D_pwfc[i]*D_pwfb[i](k/step);
		char* temp=new char[D_order];
		strncpy(temp,astr+k,D_order);
		temp[D_order]='\0';
		d+=pow(D_cwfs[get_index(temp,D_charset)]*Ik-1,2.0);	
	}
	return d/(L/step);
}


/************************************************
//                                              *
  参数：astra 欲求取其模式偏离函数值的          *
        一个字符串数组；                        *
		n 字符串数组中的字符串个数              * 
        step 描述步长；                         *
  返回值：求得的模式偏离函数值D；               *
  功能：在对偶描述子的当前状态下                *
        求给定字符串数组的模式偏离函数值；      *
//                                              *
************************************************/
double DDs::get_D(char* astra[], int n, unsigned int step)
{
	double D=0;
	for(int j=0;j<n;j++){
        D+=get_d(astra[j],step);
	}
	return D/n;
}


/************************************************
//                                              *
  参数：astra 作为描述对象的一个字符串数组；    *
		n 字符串数组中的字符串个数              * 
        step 描述步长；                         *
  返回值：无；                                  *
  功能：在给定的组成权重因子下，通过模式偏离函数*
        取极小值的条件，求出位置权重函数的      *
		展开式系数，并修正对偶描述子的当前状态；*
//                                              *
************************************************/
void DDs::minD_get_p(char* astra[], int n, unsigned int step)
{
	double* u=new double[D_rank*D_rank];
	double* v=new double[D_rank];
	int j;
	unsigned int i,k;
	for(i=0;i<D_rank;i++){
		for(k=0;k<D_rank;k++){
			u[i*D_rank+k]=0;
		}
		v[i]=0;
	}
	for(j=0;j<n;j++){
		unsigned int L=strlen(astra[j]);
		for(k=1;k<L;k++){
			char* astr=new char[D_order];
			strncpy(astr,astra[j]+k,D_order);
			astr[D_order]='\0';
			double xk=D_cwfs[get_index(astr,D_charset)];
			for(unsigned int i1=0;i1<D_rank;i1++){
				for(unsigned int i2=0;i2<D_rank;i2++){
					u[i1*D_rank+i2]+=xk*D_pwfb[i1](k/step)*xk*D_pwfb[i2](k/step);
				}
				v[i1]+=xk*D_pwfb[i1](k/step);
			}
		}
	}
	Agaus(u,v,D_rank);
	for(i=0;i<D_rank;i++){
		D_pwfc[i]=v[i];
		printf("a%d=%1.10f\n",i+1,D_pwfc[i]);
	}
}


/************************************************
//                                              *
  参数：astra 作为描述对象的一个字符串数组；    *
		n 字符串数组中的字符串个数              * 
        step 描述步长；                         *
  返回值：无；                                  *
  功能：在给定的位置权重函数下，通过模式偏离函数*
        取极小值的条件，求出各组成权重因子，    *
		并修正对偶描述子的当前状态；            *
//                                              *
************************************************/
void DDs::minD_get_c(char* astra[], int n, unsigned int step)
{
	double (*I)[2]=(double(*)[2])calloc(D_elmn,2*sizeof(double));
	double (*I2)[2]=(double(*)[2])calloc(D_elmn,2*sizeof(double));
	int j;
	unsigned int i,k;
	for(i=0;i<D_elmn;i++){
		for(j=0;j<2;j++){
			I[i][j]=0;
			I2[i][j]=0;
		}
	}
	for(j=0;j<n;j++){
		unsigned int L=strlen(astra[j]);
		for(k=0;k<L;k+=step){
			double Ik=0;
			for(i=0;i<D_rank;i++) Ik+=D_pwfc[i]*D_pwfb[i](k/step);
			char* astr=new char[D_order];
			strncpy(astr,astra[j]+k,D_order);
			astr[D_order]='\0';
			unsigned int idx=get_index(astr,D_charset);
			I[idx][0]+=Ik;
			I2[idx][1]+=Ik*Ik;
		}
	}
	for(i=0;i<D_elmn;i++){
		D_cwfs[i]=I[i][0]/I2[i][1];
		char* acwfs=new char[D_order];
		strncpy(acwfs,D_cwfstr+i*D_order,D_order);
		acwfs[D_order]='\0';
		printf("x%s=%1.10f\n",acwfs,D_cwfs[i]);
	}	
}


/************************************************
//                                              *
  参数：无；                                    *
  返回值：无；                                  *
  功能：获得组成权重因子的字符串表示            *
//                                              *
************************************************/
void DDs::get_all_permustrs()
{
	unsigned int base=strlen(D_charset);
	char* aps=new char[D_elmn*D_order];
	for(unsigned int i=0;i<D_elmn;i++){
		unsigned int num=i;
		for(unsigned int j=D_order-1;j>=0;j--){
			aps[i*D_order+(D_order-1-j)]=D_charset[num/(int)pow(base,j)];
			num%=(unsigned int)pow(base,j);
		}
	}
}


/************************************************
//                                              *
  参数：strl 一个长度为l的字符串；              *
		charset 一个以字符串形式给定的字符集；  * 
  返回值：字符串strl在字符集charset的l长字符串  *
          的升序全排列中的索引值（位置坐标）；  *
  功能：由给定的字符集和该字符集上的给定长度    *
        的字符串，求出这个字符串在由字符集中的  *
		字符组成的该长度字符串的升序全排列中的  *
		位置坐标；                              *
//                                              *
************************************************/
unsigned int DDs::get_index(char* strl,char* charset)
{
	int l=strlen(strl);
    int L=strlen(charset);
	unsigned int index=0;
	for(int i=0;i<l;i++)
	{
		index+=(unsigned int)pow(L,l-i-1)*(strchr(charset,strl[i])-charset);
	}
	return index;
}


/************************************************
//                                              *
  参数：a 系数向量；                            *
		b 常数向量；                            * 
		n 变元个数；                            *
  返回值：成功与否的标志：                      *
            1  成功；                           *
			0 不成功；                          *
  功能：用高斯法解线性方程组；                  *
//                                              *
************************************************/
int DDs::Agaus(double a[],double b[],int n) //use gaus method to get the equation resolution
{
	int *js,l,k,i,j,is,p,q;
    double d,t;
    js=(int*)malloc(n*sizeof(int));
    l=1;
    for (k=0;k<=n-2;k++)
      { d=0.0;
        for (i=k;i<=n-1;i++)
          for (j=k;j<=n-1;j++)
            { t=fabs(a[i*n+j]);
              if (t>d) { d=t; js[k]=j; is=i;}
            }
        if (d+1.0==1.0) l=0;
        else
          { if (js[k]!=k)
              for (i=0;i<=n-1;i++)
                { p=i*n+k; q=i*n+js[k];
                  t=a[p]; a[p]=a[q]; a[q]=t;
                }
            if (is!=k)
              { for (j=k;j<=n-1;j++)
                  { p=k*n+j; q=is*n+j;
                    t=a[p]; a[p]=a[q]; a[q]=t;
                  }
                t=b[k]; b[k]=b[is]; b[is]=t;
              }
          }
        if (l==0)
          { free(js); printf("fail\n");
            return(0);
          }
        d=a[k*n+k];
        for (j=k+1;j<=n-1;j++)
          { p=k*n+j; a[p]=a[p]/d;}
        b[k]=b[k]/d;
        for (i=k+1;i<=n-1;i++)
          { for (j=k+1;j<=n-1;j++)
              { p=i*n+j;
                a[p]=a[p]-a[i*n+k]*a[k*n+j];
              }
            b[i]=b[i]-a[i*n+k]*b[k];
          }
      }
    d=a[(n-1)*n+n-1];
    if (fabs(d)+1.0==1.0)
      { free(js); printf("fail\n");
        return(0);
      }
    b[n-1]=b[n-1]/d;
    for (i=n-2;i>=0;i--)
      { t=0.0;
        for (j=i+1;j<=n-1;j++)
          t=t+a[i*n+j]*b[j];
        b[i]=b[i]-t;
      }
    js[n-1]=n-1;
    for (k=n-1;k>=0;k--)
      if (js[k]!=k)
        { t=b[k]; b[k]=b[js[k]]; b[js[k]]=t;}
    free(js);
    return(1);
}