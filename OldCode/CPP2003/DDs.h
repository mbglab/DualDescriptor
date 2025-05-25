//************************************************************//
//**                         DDs.h                          **//
//************************************************************//
/*                 对偶描述子标量类的头文件                   // 
//            作者：马彬广        2003年11月18日              //
***************************************************************/

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<float.h>
#include<time.h>
#include<iostream.h>
#include<iomanip.h>

typedef double (*BaseFunc)(double);

class DDs
{
public:
	DDs(char* charset, unsigned int order, unsigned int rank, BaseFunc bases[])
	{
		unsigned int l=strlen(charset);
		D_charset=new char[l];
		strcpy(D_charset,charset);
		D_order=order;
		D_rank=rank;
		D_elmn=(unsigned int)pow(l,D_order);
		D_cwfs=new double[D_elmn];
		unsigned int i;
		for(i=0;i<D_elmn;i++) D_cwfs[i]=i+1;
		D_pwfc=new double[D_rank];
		D_pwfb=new BaseFunc[D_rank];
		for(i=0;i<D_rank;i++){
			D_pwfc[i]=i+1;
			D_pwfb[i]=bases[i];
		}
		D_cwfstr=new char[D_elmn*D_order];
		get_all_permustrs();
	}
	~DDs()
	{
		delete[] D_charset; delete[] D_cwfs;
		delete[] D_pwfc;    delete[] D_pwfb;
		delete[] D_cwfstr;
	}
	void set_cwfs(double cwfs[],int elmn);
	void get_cwfs(double cwfs[],int elmn);
	void set_pwfc(double pwfc[],int rank);
	void get_pwfc(double pwfc[],int rank);
	void set_base(BaseFunc bases[],int rank);
	void get_base(BaseFunc bases[],int rank);
	void study(char* as, int smode, unsigned int step);
	void study(char* asa[], int n, int smode, unsigned int step);
	double get_pdf_value(char* as, unsigned int step);
	double get_pdf_value(char* asa[], int n, unsigned int step);
	double* get_fpws(char* as, unsigned int step);
	double* get_dvs(char* as, unsigned int step);
	void print();
protected:
	char* D_charset;
	unsigned int D_order;
	unsigned int D_rank;
	BaseFunc* D_pwfb;
	double get_d(char* astr, unsigned int step);
	double get_D(char* astra[], int n, unsigned int step);
	void minD_get_p(char* astra[], int n, unsigned int step);
	void minD_get_c(char* astra[], int n, unsigned int step);
private:
	double* D_cwfs;
	double* D_pwfc;
	unsigned int D_elmn; 
	char* D_cwfstr;
	void get_all_permustrs();
	unsigned int get_index(char* strl,char* charset);
	int Agaus(double a[],double b[],int n);
};