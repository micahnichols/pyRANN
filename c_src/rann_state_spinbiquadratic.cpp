
#include "rann_state_spinbiquadratic.h"
#include "pair_spin_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

State_spinbiquadratic::State_spinbiquadratic(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = 0;
  aJ = 0;
  bJ = 0;
  dJ = 0;
  rc = 0;
  aK = 0;
  bK = 0;
  dK = 0;
  id = -1;
  style = "spinbiquadratic";
  atomtypes = new int[n_body_type];
  empty = true;
  fullydefined = false;
  _pair->doscreen = true;
  _pair->dospin = true;
  spin = true;
  screen = true;
}

State_spinbiquadratic::~State_spinbiquadratic()
{
  delete [] atomtypes;
  delete [] spintableJ;
  delete [] spindtableJ;
  delete [] spintableK;
  delete [] spindtableK;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_spinbiquadratic::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}

void State_spinbiquadratic::eos_function(double *ep,double **force,double **fm,double *Sik, double *dSikx, 
                                double*dSiky, double *dSikz, double *dSijkx, double *dSijky, 
                                double *dSijkz, bool *Bij,int ii,int nn,double *xn,double *yn,
                                double *zn,int *tn,int jnum,int* jl)
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
    if (spintableJ[m1]==0) {continue;}
    double *sj = sim->s[j];
    double sp = si[0]*sj[0]+si[1]*sj[1]+si[2]*sj[2];
    double sp2 = sp*sp;
    double *p = &spintableJ[m1-1];
    double *q = &spindtableJ[m1-1];
    double *r = &spintableK[m1-1];
    double *s = &spindtableK[m1-1];
    r1 = r1-trunc(r1);
    double deJ = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
    double e1J = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    double deK = r[1] + 0.5 * r1*(r[2] - r[0] + r1*(2.0*r[0] - 5.0*r[1] + 4.0*r[2] - r[3] + r1*(3.0*(r[1] - r[2]) + r[3] - r[0])));
    double e1K = s[1] + 0.5 * r1*(s[2] - s[0] + r1*(2.0*s[0] - 5.0*s[1] + 4.0*s[2] - s[3] + r1*(3.0*(s[1] - s[2]) + s[3] - s[0])));
    jj = jl[j];
    deJ *= Sik[j];
    e1J *= Sik[j];
    deK *= Sik[j];
    e1K *= Sik[j];
    fm[jj][0]+=e1J*si[0]+2*e1K*si[0]*sp;
    fm[jj][1]+=e1J*si[1]+2*e1K*si[1]*sp;
    fm[jj][2]+=e1J*si[2]+2*e1K+si[2]*sp;
    fm[ii][0]+=e1J*sj[0]+2*e1K*sj[0]*sp;
    fm[ii][1]+=e1J*sj[1]+2*e1K*sj[1]*sp;
    fm[ii][2]+=e1J*sj[2]+2*e1K*sj[2]*sp;
    //sp -= 1;
    ep[0] += e1J*sp+e1K*sp2;
    deJ*=sp;
    deK*=sp2;
    double de = deJ+deK;
    double e1 = e1J*sp+e1K*sp2;
    force[jj][0] += de*xn[j]+e1*dSikx[j];
    force[jj][1] += de*yn[j]+e1*dSiky[j];
    force[jj][2] += de*zn[j]+e1*dSikz[j];
    force[ii][0] -= de*xn[j]+e1*dSikx[j];
    force[ii][1] -= de*yn[j]+e1*dSiky[j];
    force[ii][2] -= de*zn[j]+e1*dSikz[j];
    for (int k=0;k<jnum;k++) {
      if (Bij[k]==false){continue;}
      int kk = jl[k];
      fm[kk][0] += e1*dSijkx[j*jnum+k]*si[0];
      fm[kk][1] += e1*dSijky[j*jnum+k]*si[1];
      fm[kk][2] += e1*dSijkz[j*jnum+k]*si[2];
      fm[ii][0] -= e1*dSijkx[j*jnum+k]*sj[0];
      fm[ii][1] -= e1*dSijky[j*jnum+k]*sj[1];
      fm[ii][2] -= e1*dSijkz[j*jnum+k]*sj[2];
      force[kk][0] += e1*dSijkx[j*jnum+k];
      force[kk][1] += e1*dSijky[j*jnum+k];
      force[kk][2] += e1*dSijkz[j*jnum+k];
      force[ii][0] -= e1*dSijkx[j*jnum+k];
      force[ii][1] -= e1*dSijky[j*jnum+k];
      force[ii][2] -= e1*dSijkz[j*jnum+k];
    }
  }
  return;
}

bool State_spinbiquadratic::parse_values(std::string constant,std::vector<std::string> line1) {
  int l;
  int nwords=line1.size();
  if (constant.compare("aJ")==0) {
    aJ = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("bJ")==0) {
    bJ = strtod(line1[0].c_str(),nullptr);
  }
    else if (constant.compare("dJ")==0) {
    dJ = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("aK")==0) {
    aK = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("bK")==0) {
    bK = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("dK")==0) {
    dK = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("rc")==0) {
    rc = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("dr")==0) {
    dr = strtod(line1[0].c_str(),nullptr);
  }
  else pair->errorf(FLERR,"Undefined value for spinj equation of state");
 //allow undefined delta, (default = 0)
  if (aJ!=0 && rc!=0 && aK!=0 && dr!=0 && bJ!=0 && bK!=0 && dJ!=0 && dK!=0)return true;
  return false;
}

void State_spinbiquadratic::generate_spin_table()
{
  int buf = 5;
  int m;
  double r1,as,dfc,das,uberpoly,duberpoly;
  int res = pair->res;
  spintableJ = new double[res + buf];
  spindtableJ = new double[res + buf];
  spintableK = new double[res + buf];
  spindtableK = new double[res + buf];
  double dJ2 = dJ*dJ;
  double dK2 = dK*dK;
  for (m = 0; m < (res + buf); m++) {
    r1 = rc * rc * (double) (m) / (double) (res);
    if (sqrt(r1)>=rc){
      spintableJ[m]=0;
      spindtableJ[m]=0;
      spintableK[m]=0;
      spindtableK[m]=0;
    }
    else if (sqrt(r1) <= (rc-dr)) {
      double a1 = -4*aJ*(r1/dJ2);
      double a2 = (1-bJ*r1/dJ2);
      double a3 = exp(-r1/dJ2);
      spintableJ[m]=a1*a2*a3;
      spindtableJ[m]= 2*a3/dJ2*(4*aJ*a2-a1-a1*a2);
      a1 = -4*aK*(r1/dK2);
      a2 = (1-bK*r1/dK2);
      a3 = exp(-r1/dK2);
      spintableK[m]=a1*a2*a3;
      spindtableK[m]= 2*a3/dK2*(4*aK*a2-a1-a1*a2);
    }
    else{
      double a1 = -4*aJ*(r1/dJ2);
      double a2 = (1-bJ*r1/dJ2);
      double a3 = exp(-r1/dJ2);
      double a4 = cutofffunction(sqrt(r1),rc,dr);
      double dfc = -4*pow(1-(rc-sqrt(r1))/dr,3)/dr/(1-pow(1-(rc-sqrt(r1))/dr,4));
      spintableJ[m]=a1*a2*a3*a4;
      spindtableJ[m]= 2*a3*a4/dJ2*(4*aJ*a2-a1-a1*a2+a1*a2*dfc*dJ2/sqrt(r1));
      a1 = -4*aK*(r1/dK2);
      a2 = (1-bK*r1/dK2);
      a3 = exp(-r1/dK2);
      spintableK[m]=a1*a2*a3*a4;
      spindtableK[m]= 2*a3*a4/dK2*(4*aK*a2-a1-a1*a2+a1*a2*dfc*dK2/sqrt(r1));
    }
  }
}

void State_spinbiquadratic::write_values(FILE *fid) {
  int i;
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:aJ:\n",style,id);
  fprintf(fid,"%f\n",aJ);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:bJ:\n",style,id);
  fprintf(fid,"%f\n",bJ);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:dJ:\n",style,id);
  fprintf(fid,"%f\n",dJ);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:aK:\n",style,id);
  fprintf(fid,"%f\n",aK);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:bK:\n",style,id);
  fprintf(fid,"%f\n",bK);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:dK:\n",style,id);
  fprintf(fid,"%f\n",dK);
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
}
