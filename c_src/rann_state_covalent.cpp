
#include "rann_state_covalent.h"
#include "pair_spin_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

State_covalent::State_covalent(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = 0;
  rc = 0;
  Ec = nullptr;
  a = nullptr;
  b = nullptr;
  c= nullptr;
  id = -1;
  style = "covalent";
  atomtypes = new int[n_body_type];
  empty = true;
  fullydefined = false;
  _pair->allscreen = false;
}

State_covalent::~State_covalent()
{
  delete [] atomtypes;
  delete [] Ec;
  delete [] a;
  delete [] b;
  delete [] c;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_covalent::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
  Ec = new double[pair->nelements];
  a = new double[pair->nelements];
  b = new double[pair->nelements];
  c = new double[pair->nelements];
  for (int i =0;i<pair->nelements;i++){
    Ec[i]=0;
    a[i] = 0;
    b[i] = 0;
    c[i] = 0;
  }
}

void State_covalent::eos_function(double *ep,double **force,int ii,int nn,
                                double *xn,double *yn,double *zn,int *tn,int jnum,int* jl)
{
  int nelements = pair->nelements;
  int j;
  double rsq,f;
  int res = pair->res;
  double cutinv2 = 1/rc/rc;
  int jj;
  double sum1,sum2;
  sum1=sum2=0;
  for (j=0;j<jnum;j++){
    if (atomtypes[1] != nelements && atomtypes[1] != tn[j])continue;
    rsq = xn[j]*xn[j]+yn[j]*yn[j]+zn[j]*zn[j];
    if (rsq > rc*rc)continue;
    //cubic interpolation from tables
    double r1 = (rsq*((double)res)*cutinv2);
    int m1 = (int)r1;
    if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
    m1 += (res+5)*tn[j];
    if (table1[m1]==0) {continue;}
    double *p = &table1[m1-1];
    double *q = &table2[m1-1];
    r1 = r1-trunc(r1);
    double t1 = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    double t2 = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
    sum1 += t1*Ec[tn[j]];
    sum2 += t2;
  }
  sum2 = exp(sum2);
  ep[0]+=sum1*sum2;
  for (j=0;j<jnum;j++){
    if (atomtypes[1] != nelements && atomtypes[1] != tn[j])continue;
    rsq = xn[j]*xn[j]+yn[j]*yn[j]+zn[j]*zn[j];
    if (rsq > rc*rc)continue;
    double r1 = (rsq*((double)res)*cutinv2);
    int m1 = (int)r1;
    if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
    m1 += (res+5)*tn[j];
    if (table1[m1]==0) {continue;}
    double *r = &table3[m1-1];
    double *s = &table4[m1-1];
    r1 = r1-trunc(r1);
    double t3 = r[1] + 0.5 * r1*(r[2] - r[0] + r1*(2.0*r[0] - 5.0*r[1] + 4.0*r[2] - r[3] + r1*(3.0*(r[1] - r[2]) + r[3] - r[0])));
    double t4 = s[1] + 0.5 * r1*(s[2] - s[0] + r1*(2.0*s[0] - 5.0*s[1] + 4.0*s[2] - s[3] + r1*(3.0*(s[1] - s[2]) + s[3] - s[0])));
    jj = jl[j];
    double t5 = sum2*(t3+sum1*t4);
    force[jj][0] += xn[j]*t5;
    force[jj][1] += yn[j]*t5;
    force[jj][2] += zn[j]*t5;
    force[ii][0] -= xn[j]*t5;
    force[ii][1] -= yn[j]*t5;
    force[ii][2] -= yn[j]*t5;
  }
  return;
}

bool State_covalent::parse_values(std::string constant,std::vector<std::string> line1) {
  int l;
  int nwords=line1.size();
  if (constant.compare("Ec")==0) {
    if (nwords > pair->nelements){pair->errorf(FLERR,"too many Ec values");}
    for (l=0;l<nwords;l++) {
      Ec[l]=strtod(line1[l].c_str(),nullptr);
    }
  }
  else if (constant.compare("a")==0) {
    if (nwords > pair->nelements){pair->errorf(FLERR,"too many a values");}
    for (l=0;l<nwords;l++) {
      a[l]=strtod(line1[l].c_str(),nullptr);
    }
  }
  else if (constant.compare("rc")==0) {
    rc = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("b")==0) {
    if (nwords > pair->nelements){pair->errorf(FLERR,"too many b values");}
    for (l=0;l<nwords;l++) {
      b[l]=strtod(line1[l].c_str(),nullptr);
    }
  }
  else if (constant.compare("c")==0) {
    if (nwords > pair->nelements){pair->errorf(FLERR,"too many c values");}
    for (l=0;l<nwords;l++) {
      c[l]=strtod(line1[l].c_str(),nullptr);
    }
  }
  else if (constant.compare("dr")==0) {
    dr = strtod(line1[0].c_str(),nullptr);
  }
  else pair->errorf(FLERR,"Undefined value for covalent equation of state");
 //allow undefined delta, (default = 0)
  if (a!=0 && rc!=0 && b!=0 && dr!=0 && Ec[0]!=0)return true;
  return false;
}

void State_covalent::allocate()
{
  int buf = 5;
  int m;
  double r1,as,dfc,das,uberpoly,duberpoly;
  int res = pair->res;
  int n = pair->nelements;
  int rb = res+buf;
  table1 = new double[rb*n];
  table2 = new double[rb*n];
  table3 = new double[rb*n];
  table4 = new double[rb*n];
  for (int i = 0;i<n;i++){
    for (m = 0; m < (res + buf); m++) {
      r1 = rc * rc * (double) (m) / (double) (res);
      if (sqrt(r1)>=rc){
        table1[m+rb*i]=0;
        table2[m+rb*i]=0;
        table3[m+rb*i]=0;
        table4[m+rb*i]=0;
      }
      else if (sqrt(r1) <= (rc-dr)) {
        double am = sqrt(r1)/a[i]-1;
        table1[m+rb*i] = exp(-am*c[i]);
        table1[m+rb*i] *= -am;
        table2[m+rb*i] = -b[i]/sqrt(r1)/r1;
        table3[m+rb*i] = exp(-am*c[i])/sqrt(r1);
        table3[m+rb*i] *= am*c[i]/a[i]-1/a[i];
        table4[m+rb*i] = -3*b[i]/r1/r1;
      }
      else{
        double am = sqrt(r1)/a[i]-1;
        double fc = cutofffunction(sqrt(r1), rc, dr);
        double dfc = -8*pow(1-(rc-sqrt(r1))/dr,3)/dr/(1-pow(1-(rc-sqrt(r1))/dr,4));
        table1[m+rb*i] = exp(-am*c[i]);
        table1[m+rb*i] *= -am;
        table1[m+rb*i] *= fc;
        table2[m+rb*i] = -b[i]/sqrt(r1)/r1;
        table3[m+rb*i] = exp(-am*c[i])/sqrt(r1);
        table3[m+rb*i] *= 1/a[i]*(am*c[i]-1)*fc;
        table3[m+rb*i] += table1[m+rb*i]*dfc;
        table4[m+rb*i] = -3*b[i]/r1/r1;
      }
    }
  }
}

void State_covalent::write_values(FILE *fid) {
  int i;
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:Ec:\n",style,id);
  for (i=0;i<(pair->nelements);i++) {
    fprintf(fid,"%f ",Ec[i]);
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:rc:\n",style,id);
  fprintf(fid,"%f\n",rc);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:dr:\n",style,id);
  fprintf(fid,"%f\n",dr);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:a:\n",style,id);
  for (i=0;i<(pair->nelements);i++) {
    fprintf(fid,"%f ",a[i]);
  }
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:b:\n",style,id);
  for (i=0;i<(pair->nelements);i++) {
    fprintf(fid,"%f ",b[i]);
  }
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:c:\n",style,id);
  for (i=0;i<(pair->nelements);i++) {
    fprintf(fid,"%f ",c[i]);
  }
}
