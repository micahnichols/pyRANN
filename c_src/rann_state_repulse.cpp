
#include "rann_state_repulse.h"
#include "pair_spin_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

State_repulse::State_repulse(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = 0;
  b = 0;
  rc = 0;
  alpha = 0;
  id = -1;
  style = "repulse";
  atomtypes = new int[n_body_type];
  empty = true;
  fullydefined = false;
  _pair->allscreen = false;
}

State_repulse::~State_repulse()
{
  delete [] atomtypes;
  delete [] table;
  delete [] dtable;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_repulse::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}

void State_repulse::eos_function(double *ep,double **force,int ii,int nn,
                                double *xn,double *yn,double *zn,int *tn,int jnum,int* jl)
{
  int nelements = pair->nelements;
  int j;
  double rsq,f;
  int res = pair->res;
  double cutinv2 = 1/rc/rc;
  int jj;
  for (j=0;j<jnum;j++){
    if (atomtypes[1] != nelements && atomtypes[1] != tn[j])continue;
    rsq = xn[j]*xn[j]+yn[j]*yn[j]+zn[j]*zn[j];
    if (rsq > rc*rc)continue;
    //cubic interpolation from tables
    double r1 = (rsq*((double)res)*cutinv2);
    int m1 = (int)r1;
    if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
    if (table[m1]==0) {continue;}
    double *p = &table[m1-1];
    double *q = &dtable[m1-1];
    r1 = r1-trunc(r1);
    double de = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
    double e1 = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    ep[0] += e1;
    jj = jl[j];
    force[jj][0] += de*xn[j];
    force[jj][1] += de*yn[j];
    force[jj][2] += de*zn[j];
    force[ii][0] -= de*xn[j];
    force[ii][1] -= de*yn[j];
    force[ii][2] -= de*yn[j];
  }
  return;
}

bool State_repulse::parse_values(std::string constant,std::vector<std::string> line1) {
  int l;
  int nwords=line1.size();
  if (constant.compare("b")==0) {
    b = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("rc")==0) {
    rc = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("alpha")==0) {
    alpha = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("dr")==0) {
    dr = strtod(line1[0].c_str(),nullptr);
  }
  else pair->errorf(FLERR,"Undefined value for rose equation of state");
  if (b!=0 && rc!=0 && alpha!=0 && dr!=0)return true;
  return false;
}

void State_repulse::generate_table()
{
  int buf = 5;
  int m;
  double r1,as,dfc,das,uberpoly,duberpoly;
  int res = pair->res;
  table = new double[res + buf];
  dtable = new double[res + buf];
  for (m = 0; m < (res + buf); m++) {
    r1 = rc * rc * (double) (m) / (double) (res);
    if (sqrt(r1)>=rc){
      table[m]=0;
      dtable[m]=0;
    }
    else if (sqrt(r1) <= (rc-dr)) {
      table[m] = b*exp(-alpha*sqrt(r1))/sqrt(r1);
      dtable[m]=table[m]/sqrt(r1)*(-alpha-1/r1);
    }
    else{
      dfc = -8*pow(1-(rc-sqrt(r1))/dr,3)/dr/(1-pow(1-(rc-sqrt(r1))/dr,4));
      table[m] = b*exp(-alpha*sqrt(r1))*cutofffunction(sqrt(r1), rc, dr)/sqrt(r1);
      dtable[m] = table[m]*(dfc-alpha-1/r1)/sqrt(r1);
    }
  }
}

void State_repulse::write_values(FILE *fid) {
  int i;
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:b:\n",style,id);
  fprintf(fid,"%f\n",b);
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
  fprintf(fid,":%s_%d:alpha:\n",style,id);
  fprintf(fid,"%f\n",alpha);
}
