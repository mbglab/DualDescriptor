//************************************************************//
//**                        DDs.cpp                         **//
//************************************************************//
/*                ��ż�����ӱ������ʵ���ļ�                  // 
//            ���ߣ�����        2003��11��18��              //
***************************************************************/


#include"DDs.h"


//*****************��Ա��������˵��*******************//

/************************************************
//                                              *
  ������cwfs �����õ����Ȩ���������飻         *
        elmn ���Ȩ�����ӵĸ�����               *
  ����ֵ���ޣ�                                  *
  ���ܣ��������Ȩ�����ӵ�ֵ��                  *
//                                              *
************************************************/
void DDs::set_cwfs(double cwfs[], int elmn)
{
	for(unsigned int i=0;i<D_elmn;i++) D_cwfs[i]=cwfs[i];
}


/************************************************
//                                              *
  ������cwfs ��Ż�õ����Ȩ�����ӵ����飻     *
        elmn ���Ȩ�����ӵĸ�����               *
  ����ֵ���ޣ�                                  *
  ���ܣ�������Ȩ�����ӵ�ֵ��                  *
//                                              *
************************************************/
void DDs::get_cwfs(double cwfs[],int elmn)
{
	for(unsigned int i=0;i<D_elmn;i++) cwfs[i]=D_cwfs[i];	
}


/************************************************
//                                              *
  ������pwfc �����õ�λ��Ȩ�غ���ϵ�������飻   *
        rank λȨ����չ��ʽ��������             *
  ����ֵ���ޣ�                                  *
  ���ܣ�����λ��Ȩ�غ���չ��ʽ��ϵ��ֵ          *
//                                              *
************************************************/
void DDs::set_pwfc(double pwfc[],int rank)
{
	for(unsigned int i=0;i<D_rank;i++) D_pwfc[i]=pwfc[i];
}


/************************************************
//                                              *
  ������pwfc ��Ż�õ�λ��Ȩ�غ���ϵ�������飻 *
        rank λȨ����չ��ʽ��������             *
  ����ֵ���ޣ�                                  *
  ���ܣ����λ��Ȩ�غ���չ��ʽ��ϵ��ֵ��        *
//                                              *
************************************************/
void DDs::get_pwfc(double pwfc[],int rank)
{
	for(unsigned int i=0;i<D_rank;i++) pwfc[i]=D_pwfc[i];
}


/************************************************
//                                              *
  ������bases[rank] �����õ�λ��Ȩ�غ�����      * 
        ������ָ�����飻                        *
        rank λȨ����չ��ʽ��������             *
  ����ֵ���ޣ�                                  *
  ���ܣ�����λ��Ȩ�غ����Ļ�����;               *
//                                              *
************************************************/
void DDs::set_base(BaseFunc bases[], int rank)
{
	for(unsigned int i=0;i<D_rank;i++) D_pwfb[i]=bases[i];
}


/************************************************
//                                              *
  ������bases[rank] ��Ż�õ�λ��Ȩ�غ�����    * 
        ������ָ�����飻                        *
        rank λȨ����չ��ʽ��������             *
  ����ֵ���ޣ�                                  *
  ���ܣ����λ��Ȩ�غ����Ļ�����;               *
//                                              *
************************************************/
void DDs::get_base(BaseFunc bases[], int rank)
{
	for(unsigned int i=0;i<D_rank;i++) bases[i]=D_pwfb[i];
}


/************************************************
//                                              *
  ������as ����ȡ�������ַ������������󣩣�     *
        smode ѧϰģʽ����ȡ�ĸ�ֵ��            *
		  1 �������Ȩ�����ӣ�һ����ѧϰ��      *
		  2 ����λ��Ȩ�غ�����һ����ѧϰ��      *
		  3 �Ӹ��������Ȩ�����ӽ��뽻��ʽѧϰ��*  
		  4 �Ӹ�����λ��Ȩ�غ������뽻��ʽѧϰ��*
        step �����Ĳ�����                       *
  ����ֵ���ޣ�                                  *
  ���ܣ��ڸ�����ѧϰģʽ�����������£�          *
        ����ѧϰ���̣��Ի���ַ����ļ���������  *
//                                              *
************************************************/
void DDs::study(char* as, int smode, unsigned int step)
{
	//�����ַ���ת�����ַ�������
	char* temp[1];  
	temp[1]=new char[strlen(as)];
	strcpy(temp[1],as);
	temp[1][strlen(as)]='\0';
	//���ø������һ���غ���
	study(temp,1,smode,step);	
}


/************************************************
//                                              *
  ������asa ����ȡ�������ַ������飨�������󣩣�*
        smode ѧϰģʽ����ȡ�ĸ�ֵ��            *
		  1 �������Ȩ�����ӣ�һ����ѧϰ��      *
		  2 ����λ��Ȩ�غ�����һ����ѧϰ��      *
		  3 �Ӹ��������Ȩ�����ӽ��뽻��ʽѧϰ��*  
		  4 �Ӹ�����λ��Ȩ�غ������뽻��ʽѧϰ��*
        step �����Ĳ�����                       *
  ����ֵ���ޣ�                                  *
  ���ܣ��ڸ�����ѧϰģʽ�����������£�����ѧϰ  *
        ���̣��Ի���ַ�������ļ���������      *
//                                              *
************************************************/
void DDs::study(char* asa[], int n, int smode, unsigned int step)
{
	double D,D0=DBL_MAX;
	double* oldcwfs=new double[D_elmn];
	double* oldpwfc=new double[D_rank];
	clock_t start,end;
	unsigned int i=0,j=0;
	//����ѧϰģʽѡ��ѧϰ��ʽ
	switch(smode){
	case 1: 
		cout<<"ģʽƫ�뺯���ĳ�ʼֵ"<<endl;
		cout<<get_D(asa,n,step)<<endl;
		minD_get_p(asa,n,step);
		cout<<"һ��ѧϰ���ģʽƫ�뺯��ֵ"<<endl;
		cout<<get_D(asa,n,step)<<endl;
		break;
	case 2: 
		cout<<"ģʽƫ�뺯���ĳ�ʼֵ"<<endl;
		cout<<get_D(asa,n,step)<<endl;
		minD_get_c(asa,n,step);
		cout<<"һ��ѧϰ���ģʽƫ�뺯��ֵ"<<endl;
		cout<<get_D(asa,n,step)<<endl;
		break;
	case 3: 		
		cout<<"ģʽƫ�뺯���ĳ�ʼֵ\n";
		D=get_D(asa,n,step);
		cout<<D<<endl;
		cout<<"��ż�����ӵĳ�ʼ����"<<endl;
		print();
		printf("��ʼ����ʽѧϰ\n");
		printf("����������������������\n");	
		i=0;
		start=clock();
		while(D0>D){
			D0=D;
			for(j=0;j<D_elmn;j++) oldcwfs[j]=D_cwfs[j];
			for(j=0;j<D_rank;j++) oldpwfc[j]=D_pwfc[j];
			i++;
			printf("�� %d ��ѧϰʱ�Ĳ���ֵ\n",i);
			minD_get_p(asa,n,step);
			minD_get_c(asa,n,step);
			printf("�� %d ��ѧϰ���Dֵ\n",i);
			D=get_D(asa,n,step);
			cout<<D<<endl;
		}
		for(j=0;j<D_elmn;j++) D_cwfs[j]=oldcwfs[j];
		for(j=0;j<D_rank;j++) D_pwfc[j]=oldpwfc[j];
		printf("�� %d ��ѧϰ�����СDֵ\n",i-1);
		cout<<D0<<endl;
		end=clock();
		printf("the time used: %3.3f seconds\n",(double)(end-start)/CLK_TCK);
		break;
	case 4:	
		cout<<"ģʽƫ�뺯���ĳ�ʼֵ\n";
		D=get_D(asa,n,step);
		cout<<D<<endl;
		cout<<"��ż�����ӵĳ�ʼ����"<<endl;
		print();
		printf("��ʼ����ʽѧϰ\n");
		printf("����������������������\n");	
		i=0;		
		start=clock();
		while(D0>D){
			D0=D;
			for(j=0;j<D_elmn;j++) oldcwfs[j]=D_cwfs[j];
			for(j=0;j<D_rank;j++) oldpwfc[j]=D_pwfc[j];
			i++;
			printf("�� %d ��ѧϰʱ�Ĳ���ֵ\n",i);
			minD_get_c(asa,n,step);
			minD_get_p(asa,n,step);			
			printf("�� %d ��ѧϰ���Dֵ\n",i);
			D=get_D(asa,n,step);
			cout<<D<<endl;
		}
		for(j=0;j<D_elmn;j++) D_cwfs[j]=oldcwfs[j];
		for(j=0;j<D_rank;j++) D_pwfc[j]=oldpwfc[j];
		printf("�� %d ��ѧϰ�����СDֵ\n",i-1);
		cout<<D0<<endl;
		end=clock();
		printf("the time used: %3.3f seconds\n",(double)(end-start)/CLK_TCK);
		break;
	default:
		cout<<"��ѧϰģʽ������"<<endl;
		break;
	}		
}


/************************************************
//                                              *
  ������as ����ȡ��ģʽƫ�뺯��ֵ��             *
        һ���ַ�����                            * 
        step ����������                         *
  ����ֵ����õ�ģʽƫ�뺯��ֵ��                *
  ���ܣ��ڶ�ż�����ӵĵ�ǰ״̬��                *
        ������ַ�����ģʽƫ�뺯��ֵ��          *
//                                              *
************************************************/
double DDs::get_pdf_value(char* as, unsigned int step)
{
	return get_d(as,step);
}


/************************************************
//                                              *
  ������asa ����ȡ��ģʽƫ�뺯��ֵ��            *
        һ���ַ������飻                        *
		n �ַ��������е��ַ�������              * 
        step ����������                         *
  ����ֵ����õ�ģʽƫ�뺯��ֵ��                *
  ���ܣ��ڶ�ż�����ӵĵ�ǰ״̬��                *
        ������ַ��������ģʽƫ�뺯��ֵ��      *
//                                              *
************************************************/
double DDs::get_pdf_value(char* asa[], int n, unsigned int step)
{
	return get_D(asa,n,step);
}


/************************************************
//                                              *
  ������as ����ȡ���ȨƵ�ʵ�һ���ַ�����       *
        step ����������                         *
  ����ֵ����õļ�ȨƵ�����飬��������          *
          D_elmn��Ԫ�أ�����ֻ��D_elmn-1��      *
		  �Ǳ˴˶����ģ�                        *
  ���ܣ��ڶ�ż�����ӵĵ�ǰ״̬�£������        * 
        �ַ����ļ�ȨƵ�ʣ�                      *
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
  ������as ����ȡ���ż�������е�һ���ַ�����   *
        step ����������                         *
  ����ֵ����õĶ�ż�������У����飩���������  *
          Ԫ�ظ������ڸ����ַ����ĳ��ȣ�        *
  ���ܣ��ڶ�ż�����ӵĵ�ǰ״̬�£������        * 
        �ַ����Ķ�ż�������У��ɾݴ˻���1ά     *
		D���ߣ�                                 *
//                                              *
************************************************/
double* DDs::get_dvs(char* as, unsigned int step)
{
	unsigned j,k;
	//��һ��������Ȩ������
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
  �������ޣ�                                    *
  ����ֵ���ޣ�                                  *
  ���ܣ������ż�������ڵ�ǰ״̬�µĲ���ֵ��    *
        ���Ȩ�����Ӻ�λ��Ȩ�غ���չ��ʽ��ϵ����*
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
  ������astr ����ȡ��ģʽƫ�뺯��ֵ��           *
        һ���ַ�����                            * 
        step ����������                         *
  ����ֵ����õ�ģʽƫ�뺯��ֵd��               *
  ���ܣ��ڶ�ż�����ӵĵ�ǰ״̬��                *
        ������ַ�����ģʽƫ�뺯��ֵ��          *
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
  ������astra ����ȡ��ģʽƫ�뺯��ֵ��          *
        һ���ַ������飻                        *
		n �ַ��������е��ַ�������              * 
        step ����������                         *
  ����ֵ����õ�ģʽƫ�뺯��ֵD��               *
  ���ܣ��ڶ�ż�����ӵĵ�ǰ״̬��                *
        ������ַ��������ģʽƫ�뺯��ֵ��      *
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
  ������astra ��Ϊ���������һ���ַ������飻    *
		n �ַ��������е��ַ�������              * 
        step ����������                         *
  ����ֵ���ޣ�                                  *
  ���ܣ��ڸ��������Ȩ�������£�ͨ��ģʽƫ�뺯��*
        ȡ��Сֵ�����������λ��Ȩ�غ�����      *
		չ��ʽϵ������������ż�����ӵĵ�ǰ״̬��*
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
  ������astra ��Ϊ���������һ���ַ������飻    *
		n �ַ��������е��ַ�������              * 
        step ����������                         *
  ����ֵ���ޣ�                                  *
  ���ܣ��ڸ�����λ��Ȩ�غ����£�ͨ��ģʽƫ�뺯��*
        ȡ��Сֵ����������������Ȩ�����ӣ�    *
		��������ż�����ӵĵ�ǰ״̬��            *
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
  �������ޣ�                                    *
  ����ֵ���ޣ�                                  *
  ���ܣ�������Ȩ�����ӵ��ַ�����ʾ            *
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
  ������strl һ������Ϊl���ַ�����              *
		charset һ�����ַ�����ʽ�������ַ�����  * 
  ����ֵ���ַ���strl���ַ���charset��l���ַ���  *
          ������ȫ�����е�����ֵ��λ�����꣩��  *
  ���ܣ��ɸ������ַ����͸��ַ����ϵĸ�������    *
        ���ַ������������ַ��������ַ����е�  *
		�ַ���ɵĸó����ַ���������ȫ�����е�  *
		λ�����ꣻ                              *
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
  ������a ϵ��������                            *
		b ����������                            * 
		n ��Ԫ������                            *
  ����ֵ���ɹ����ı�־��                      *
            1  �ɹ���                           *
			0 ���ɹ���                          *
  ���ܣ��ø�˹�������Է����飻                  *
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