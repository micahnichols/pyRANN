
#include "rann_state_spin_j.h"
#include "pair_spin_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

State_spinj::State_spinj(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = 0;
  re = 0;
  rc = 0;
  a = 0;
  b = 0;
  id = -1;
  style = "spinj";
  atomtypes = new int[n_body_type];
  empty = true;
  fullydefined = false;
  _pair->allscreen = false;
  _pair->dospin = true;
  spin = true;
}

State_spinj::~State_spinj()
{
  delete [] atomtypes;
  delete [] spintable;
  delete [] spindtable;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_spinj::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}

void State_spinj::eos_function(double *ep,double **force,double **fm,int ii,int nn,
                                double *xn,double *yn,double *zn,int *tn,int jnum,int* jl)
{
  int nelements = pair->nelements;
  int j;
  double rsq,f;
  int res = pair->res;
  double cutinv2 = 1/rc/rc;
  int jj;
  PairRANN::Simulation *sim = &pair->sims[nn];
  double *si = sim->s[ii];
  for (j=0;j<jnum;j++){
    if (atomtypes[1] != nelements && atomtypes[1] != tn[j])continue;
    rsq = xn[j]*xn[j]+yn[j]*yn[j]+zn[j]*zn[j];
    if (rsq > rc*rc)continue;
    //cubic interpolation from tables
    double r1 = (rsq*((double)res)*cutinv2);
    int m1 = (int)r1;
    if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
    if (spintable[m1]==0) {continue;}
    double *sj = sim->s[j];
    double sp = si[0]*sj[0]+si[1]*sj[1]+si[2]*sj[2];
    double *p = &spintable[m1-1];
    double *q = &spindtable[m1-1];
    r1 = r1-trunc(r1);
    double de = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
    double e1 = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    jj = jl[j];
    fm[jj][0]+=e1*si[0];
    fm[jj][1]+=e1*si[1];
    fm[jj][2]+=e1*si[2];
    fm[ii][0]+=e1*sj[0];
    fm[ii][1]+=e1*sj[1];
    fm[ii][2]+=e1*sj[2];
    ep[0] += e1*sp;
    de*=sp;
    force[jj][0] += de*xn[j];
    force[jj][1] += de*yn[j];
    force[jj][2] += de*zn[j];
    force[ii][0] -= de*xn[j];
    force[ii][1] -= de*yn[j];
    force[ii][2] -= de*yn[j];
  }
  return;
}

bool State_spinj::parse_values(std::string constant,std::vector<std::string> line1) {
  int l;
  int nwords=line1.size();
  if (constant.compare("a")==0) {
    a = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("re")==0) {
    re = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("rc")==0) {
    rc = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("b")==0) {
    b = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("dr")==0) {
    dr = strtod(line1[0].c_str(),nullptr);
  }
  else pair->errorf(FLERR,"Undefined value for spinj equation of state");
 //allow undefined delta, (default = 0)
  if (re!=0 && rc!=0 && a!=0 && dr!=0 && b!=0)return true;
  return false;
}

void State_spinj::generate_spin_table()
{
  int buf = 5;
  int m;
  double r1,as,dfc,das,uberpoly,duberpoly;
  int res = pair->res;
  spintable = new double[res + buf];
  spindtable = new double[res + buf];
  for (m = 0; m < (res + buf); m++) {
    r1 = rc * rc * (double) (m) / (double) (res);
    if (sqrt(r1)>=rc){
      spintable[m]=0;
      spindtable[m]=0;
    }
    else if (sqrt(r1) <= (rc-dr)) {
      double a1 = 4*a*(r1/re/re);
      double a2 = (1-b*r1/re/re);
      double a3 = exp(-r1/re/re);
      spintable[m]=a1*a2*a3;
      spindtable[m]= 2*a3/re/re*(4*a*a2-a1-a1*a2);
    }
    else{
      double a1 = 4*a*(r1/re/re);
      double a2 = (1-b*r1/re/re);
      double a3 = exp(-r1/re/re);
      double a4 = cutofffunction(sqrt(r1),rc,dr);
      double dfc = dfc = -4*pow(1-(rc-sqrt(r1))/dr,3)/dr/(1-pow(1-(rc-sqrt(r1))/dr,4));
      spintable[m]=a1*a2*a3*a4;
      spindtable[m]= 2*a3*a4/re/re*(4*a*a2-a1-a1*a2+a1*a2*dfc*re*re/sqrt(r1));
    }
  }
}

void State_spinj::write_values(FILE *fid) {
  int i;
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:re:\n",style,id);
  fprintf(fid,"%f\n",re);
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
  fprintf(fid,"%f\n",a);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:b:\n",style,id);
  fprintf(fid,"%f\n",b);
}
