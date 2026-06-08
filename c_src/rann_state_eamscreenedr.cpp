
#include "rann_state_eamscreenedr.h"
#include "pair_spin_rann.h"
#include <cmath>
#include "rann_state_eam_ref_b1.h"
#include "rann_state_eam_ref_b2.h"
#include "rann_state_eam_ref_bcc.h"
#include "rann_state_eam_ref_ch4.h"
#include "rann_state_eam_ref_dia.h"
#include "rann_state_eam_ref_dim.h"
#include "rann_state_eam_ref_dimbin.h"
#include "rann_state_eam_ref_fcc.h"
#include "rann_state_eam_ref_hcp.h"
#include "rann_state_eam_ref_l12.h"

using namespace LAMMPS_NS::RANN;

State_eamscreenedr::State_eamscreenedr(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = nullptr;
  re = nullptr;
  rc = nullptr;
  alpha = nullptr;
  delta = nullptr;
  cweight = nullptr;
  beta0 = nullptr;
  Asub = nullptr;
  ec = nullptr;
  id = -1;
  style = "eamscreenedr";
  atomtypes = new int[n_body_type];
  empty = false;
  fullydefined = false;
  _pair->doscreen = true;
  screen = true;
}

State_eamscreenedr::~State_eamscreenedr()
{
  delete [] atomtypes;
  delete [] ec;
  delete [] cweight;
  delete [] alpha;
  delete [] delta;
  delete [] re;
  delete [] Asub;
  delete [] beta0;
  delete [] dr;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_eamscreenedr::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {
    atomtypes[j] = i[j];
    if (i[j]!=pair->nelements){pair->errorf(FLERR,"State equation EAM is designed to use with all_all atom types only!");}  
  }
  id = _id;
  ec = new double*[pair->nelements];
  cweight = new double[pair->nelements];
  alpha = new double*[pair->nelements];
  delta = new double*[pair->nelements];
  re = new double*[pair->nelements];
  Asub = new double[pair->nelements];
  beta0 = new double[pair->nelements];
  dr = new double*[pair->nelements];
  rc = new double*[pair->nelements];
  lat = new Reference_lattice**[pair->nelements];
  for (int i=0;i<pair->nelements;i++) {
    ec[i] = new double[pair->nelements];
    re[i] = new double[pair->nelements];
    alpha[i] = new double[pair->nelements];
    delta[i] = new double[pair->nelements];
    dr[i] = new double[pair->nelements];
    rc[i] = new double[pair->nelements];
    lat[i] = new Reference_lattice*[pair->nelements];
    Asub[i]=0;
    beta0[i]=0;
    cweight[i]=0;
    for (int j=0;j<pair->nelements;j++) {
      ec[i][j] =0;
      re[i][j]=0;
      alpha[i][j]=0;
      delta[i][j]=0;
      dr[i][j]=0;
      rc[i][j]=0;
      lat[i][j] = new Reference_lattice(this);
    }
  }
}

Reference_lattice *State_eamscreenedr::create_lattice(const char *style)
{
  if (strcmp(style,"fcc")==0) {
    return new Ref_FCC(this);
  }
  else if (strcmp(style,"bcc")==0) {
    return new Ref_BCC(this);
  }
  else if (strcmp(style,"hcp")==0) {
    return new Ref_HCP(this);
  }
  else if (strcmp(style,"b1")==0) {
	  return new Ref_B1(this);
  }
  else if (strcmp(style,"b2")==0) {
	  return new Ref_B2(this);
  }
  else if (strcmp(style,"ch4")==0) {
	  return new Ref_CH4(this);
  }
  else if (strcmp(style,"dia")==0) {
	  return new Ref_DIA(this);
  }
  else if (strcmp(style,"dim")==0) {
	  return new Ref_DIM(this);
  }
  else if (strcmp(style,"dimbin")==0) {
	  return new Ref_DIMBIN(this);
  }
  else if (strcmp(style,"l12")==0) {
	  return new Ref_L12(this);
  }
  else if (strcmp(style,"empty")==0) {
    return new Reference_lattice(this);
  }
  pair->errorf(FLERR,"Unknown reference lattice style");
  return nullptr;
}


void State_eamscreenedr::eos_function(double *ep,double **force,double *Sik, double *dSikx, 
                                double*dSiky, double *dSikz, double *dSijkx, double *dSijky, 
                                double *dSijkz, bool *Bij,int ii,int nn,double *xn,double *yn,
                                double *zn,int *tn,int jnum,int* jl)
{
  int nelements = pair->nelements;
  int i,j;
  double rsq,f;
  double rhobar = 0;
  int res = pair->res;
  int jj;
  double rho = 0;
  double phisum = 0;
  PairRANN::Simulation *sim = &pair->sims[nn];
  int itype = pair->map[sim->type[ii]];
  i = itype;
  for (j=0;j<jnum;j++){
    if (Bij[j]==false) {continue;}
    rsq = xn[j]*xn[j]+yn[j]*yn[j]+zn[j]*zn[j];
    if (rsq > rc[i][tn[j]]*rc[i][tn[j]])continue;
    //cubic interpolation from tables
    double cutinv2 = 1/rc[i][tn[j]]/rc[i][tn[j]];
    double r1 = (rsq*((double)res)*cutinv2);
    int m1 = (int)r1;
    if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
    if (phitable[i][tn[j]][m1]==0) {continue;}
    double *p = &phitable[i][tn[j]][m1-1];
    double *r = &rhosummandtable[i][tn[j]][m1-1];
    double *dp = &dphitable[i][tn[j]][m1-1];
    r1 = r1-trunc(r1);
    double phi = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    double rhosummand = r[1] + 0.5 * r1*(r[2] - r[0] + r1*(2.0*r[0] - 5.0*r[1] + 4.0*r[2] - r[3] + r1*(3.0*(r[1] - r[2]) + r[3] - r[0])));
    double dphi = dp[1] + 0.5 * r1*(dp[2] - dp[0] + r1*(2.0*dp[0] - 5.0*dp[1] + 4.0*dp[2] - dp[3] + r1*(3.0*(dp[1] - dp[2]) + dp[3] - dp[0])));
    rhosummand *= Sik[j];
    phi *= Sik[j];
    dphi *= Sik[j];
    phisum += phi;
    rho += rhosummand;
    jj = jl[j];
    force[jj][0] += dphi*xn[j]+phi*dSikx[j];
    force[jj][1] += dphi*yn[j]+phi*dSiky[j];
    force[jj][2] += dphi*zn[j]+phi*dSikz[j];
    force[ii][0] -= dphi*xn[j]+phi*dSikx[j];
    force[ii][1] -= dphi*yn[j]+phi*dSiky[j];
    force[ii][2] -= dphi*zn[j]+phi*dSikz[j];
    for (int k=0;k<jnum;k++) {
      if (Bij[k]==false){continue;}
      int kk = jl[k];
      force[kk][0] += phi*dSijkx[j*jnum+k];
      force[kk][1] += phi*dSijky[j*jnum+k];
      force[kk][2] += phi*dSijkz[j*jnum+k];
      force[ii][0] -= phi*dSijkx[j*jnum+k];
      force[ii][1] -= phi*dSijky[j*jnum+k];
      force[ii][2] -= phi*dSijkz[j*jnum+k];
    }
  }
  ep[0] += phisum;
  if (rho>0.0){
    double F = Asub[itype]*ec[itype][itype]*rho/D_ref[itype]*log(rho/D_ref[itype]);
    double dF = Asub[itype]*ec[itype][itype]/D_ref[itype]*(log(rho/D_ref[itype])+1);
    ep[0] += F;
    for (j=0;j<jnum;j++){
      if (Bij[j]==false) {continue;}
      if (atomtypes[1] != nelements && atomtypes[1] != tn[j])continue;
      rsq = xn[j]*xn[j]+yn[j]*yn[j]+zn[j]*zn[j];
      if (rsq > rc[i][tn[j]]*rc[i][tn[j]])continue;
      //cubic interpolation from tables
      double cutinv2 = 1/rc[i][tn[j]]/rc[i][tn[j]];
      double r1 = (rsq*((double)res)*cutinv2);
      int m1 = (int)r1;
      if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
      if (phitable[i][tn[j]][m1]==0) {continue;}
      double *r = &rhosummandtable[i][tn[j]][m1-1];
      double *dr = &drhosummandtable[i][tn[j]][m1-1];
      r1 = r1-trunc(r1);
      double rhosummand = r[1] + 0.5 * r1*(r[2] - r[0] + r1*(2.0*r[0] - 5.0*r[1] + 4.0*r[2] - r[3] + r1*(3.0*(r[1] - r[2]) + r[3] - r[0])));
      double drhosummand = dr[1] + 0.5 * r1*(dr[2] - dr[0] + r1*(2.0*dr[0] - 5.0*dr[1] + 4.0*dr[2] - dr[3] + r1*(3.0*(dr[1] - dr[2]) + dr[3] - dr[0])));
      rhosummand *= Sik[j];
      drhosummand *= Sik[j];
      jj = jl[j];
      force[jj][0] += dF*(drhosummand*xn[j]+rhosummand*dSikx[j]);
      force[jj][1] += dF*(drhosummand*yn[j]+rhosummand*dSiky[j]);
      force[jj][2] += dF*(drhosummand*zn[j]+rhosummand*dSikz[j]);
      force[ii][0] -= dF*(drhosummand*xn[j]+rhosummand*dSikx[j]);
      force[ii][1] -= dF*(drhosummand*yn[j]+rhosummand*dSiky[j]);
      force[ii][2] -= dF*(drhosummand*zn[j]+rhosummand*dSikz[j]);
      for (int k=0;k<jnum;k++) {
        if (Bij[k]==false){continue;}
        int kk = jl[k];
        force[kk][0] += dF*rhosummand*dSijkx[j*jnum+k];
        force[kk][1] += dF*rhosummand*dSijky[j*jnum+k];
        force[kk][2] += dF*rhosummand*dSijkz[j*jnum+k];
        force[ii][0] -= dF*rhosummand*dSijkx[j*jnum+k];
        force[ii][1] -= dF*rhosummand*dSijky[j*jnum+k];
        force[ii][2] -= dF*rhosummand*dSijkz[j*jnum+k];
      }
    }
  }
}

bool State_eamscreenedr::parse_values(std::string constant,std::vector<std::string> line1) {
  int l,ll,count;
  int nwords=line1.size();
  int nelements = pair->nelements;
  int len = nelements*(nelements+1);
  int len1 = nelements*nelements;
  len = len/2;
  if (constant.compare("ec")==0) {
    count = 0;
    if (nwords != (len1)){pair->errorf(FLERR,"too many or too few ec values");}
    for (l=0;l<nelements;l++) {
      for (ll=0;ll<nelements;ll++) {
        ec[l][ll] = strtod(line1[count].c_str(),nullptr);
	      count += 1;
      }
    }
  }
  else if (constant.compare("re")==0) {
    if (nwords != len1){pair->errorf(FLERR,"too many or too few re values");}
    count = 0;
    for (l=0;l<nelements;l++) { 
      for (ll=0;ll<nelements;ll++) {
        re[l][ll] = strtod(line1[count].c_str(),nullptr);
	      count += 1;
      }
    }  
  }
  else if (constant.compare("rc")==0) {
    if (nwords != len1){pair->errorf(FLERR,"too many or too few re values");}
    count = 0;
    for (l=0;l<nelements;l++) { 
      for (ll=0;ll<nelements;ll++) {
        rc[l][ll] = strtod(line1[count].c_str(),nullptr);
	      count += 1;
      }
    }
  }
  else if (constant.compare("alpha")==0) {
    if (nwords != len1){pair->errorf(FLERR,"too many or too few alpha values");}
    count = 0;
    for (l=0;l<nelements;l++) {
      for (ll=0;ll<nelements;ll++) {
        alpha[l][ll] = strtod(line1[count].c_str(),nullptr);
	      count += 1;
      }
    }
  }
  else if (constant.compare("delta")==0) {
    if (nwords != len1){pair->errorf(FLERR,"too many or too few delta values");}
    count = 0;
    for (l=0;l<nelements;l++) {
      for (ll=0;ll<nelements;ll++) {
        delta[l][ll] = strtod(line1[count].c_str(),nullptr);
	      count += 1;
      }
    }
  }
  else if (constant.compare("cweight")==0) {
    if (nwords > pair->nelements){pair->errorf(FLERR,"too many cweight values");}
    count = 0;
    for (l=0;l<nwords;l++) {
        cweight[l] = strtod(line1[count].c_str(),nullptr);
	      count += 1;
    }
  }
  else if (constant.compare("dr")==0) {
    count = 0;
    if (nwords != len1){pair->errorf(FLERR,"too many or too few dr values");}
    for (l=0;l<nelements;l++) {
      for (ll=0;ll<nelements;ll++) {
        dr[l][ll] = strtod(line1[count].c_str(),nullptr);
	      count += 1;
      }
    }
  }
  else if (constant.compare("beta")==0) {
    if (nwords > pair->nelements){pair->errorf(FLERR,"too many beta values");}
    for (l=0;l<nwords;l++) {
      beta0[l] = strtod(line1[l].c_str(),nullptr);
    }
  }
  else if (constant.compare("Asub")==0) {
    if (nwords > pair->nelements){pair->errorf(FLERR,"too many Asub values");}
    for (l=0;l<nwords;l++) {
      Asub[l] = strtod(line1[l].c_str(),nullptr);
    }
  }
  else if (constant.compare("lattice")==0) {
    //define full matrix of lattice values:
    //e1-e1 e1-e2 e1-e3 ... e2-e1 e2-e2 ... e3-e1 ...
    //off-diagonal lattices should be defined for either upper or lower entry, not both. Use lat=empty for the alternate side of the diagonal.
    //upper vs. lower determines mapping of potential elements to lattice positions in binary lattices.
    count = 0;
    if (nwords != nelements*nelements){pair->errorf(FLERR,"too many or too few lattice values");}
    for (l=0;l<nelements;l++) {
      for (ll=0;ll<nelements;ll++) {
        if (lat[l][ll]->empty==false) {
          count++;
          continue;
        }
        delete lat[l][ll];
        lat[l][ll]=create_lattice(line1[count].c_str());
        if (l==ll && lat[l][ll]->n_body_type==2){pair->errorf(FLERR,"tried to use binary lattice with one element!");}
        if (l!=ll && lat[l][ll]->n_body_type==1){pair->errorf(FLERR,"tried to use unary lattice with two elements!");}
        lat[l][ll]->empty=false;
        lat[l][ll]->typemap = new int [nelements];
        lat[l][ll]->typemap[l]=0;
        lat[l][ll]->invtypemap = new int [2];
        lat[l][ll]->invtypemap[0]=l;
        lat[l][ll]->invtypemap[1]=l;
        if (l!=ll) {
          lat[l][ll]->typemap[ll]=1;
          lat[l][ll]->invtypemap[1]=ll;
          if (lat[ll][l]->empty==false){pair->errorf(FLERR,"multiple definitions of lattice");}
          delete lat[ll][l];
          lat[ll][l]=create_lattice(line1[count].c_str());
          lat[ll][l]->empty=false;
          lat[ll][l]->typemap = new int [nelements];
          lat[ll][l]->typemap[l]=0;
          lat[ll][l]->typemap[ll]=1;
          lat[ll][l]->invtypemap = new int [2];
          lat[ll][l]->invtypemap[0]=l;
          lat[ll][l]->invtypemap[1]=ll;
        }
	      count += 1;
      }
    }
  }
  else pair->errorf(FLERR,"Undefined value for eamscreenedr equation of state");
  bool finished = true;
  for (l=0;l<nelements;l++){
    if (Asub[l]==0)finished=false;break;
    if (beta0[l]==0)finished=false;break;
    if (cweight[l]==0)finished=false;break;
    for (ll=0;ll<nelements;ll++){
      if (re[l][ll]==0)finished=false;break;
      if (rc[l][ll]==0)finished=false;break;
      if (dr[l][ll]==0)finished=false;break;
      if (alpha[l][ll]==0)finished=false;break;
      if (delta[l][ll]==0)finished=false;break;
      if (ec[l][ll]==0)finished=false;break;
      if (lat[l][ll]->empty)finished=false;break;
    }
  }
  return finished;
}

void State_eamscreenedr::allocate()
{
  int buf = 5;
  int i,mn,m;
  double r1;
  int res = pair->res;
  int nelements = pair->nelements;

  phitable = new double **[nelements];
  rhosummandtable = new double **[nelements];
  dphitable = new double **[nelements];
  drhosummandtable = new double **[nelements];
  D_ref = new double[nelements];
  for (i=0;i < nelements;i++){
    for (mn=0;mn < nelements;mn++){
      lat[i][mn]->compute_reference_parameters();
    }
  }
  for (i = 0; i < nelements; i++) {
    phitable[i] = new double *[nelements];
    rhosummandtable[i] = new double *[nelements];
    dphitable[i] = new double *[nelements];
    drhosummandtable[i] = new double *[nelements];
    int I = lat[i][i]->typemap[i];
    D_ref[i] = cweight[i]*lat[i][i]->Z+cweight[i]*lat[i][i]->Yij[I][I]*lat[i][i]->Sij[I][I]*exp(-beta0[i]*(lat[i][i]->aij-1));
    //have to do diagonal first
    mn = i;
    phitable[i][mn] = new double[res+buf];
    rhosummandtable[i][mn] = new double[res+buf];
    dphitable[i][mn] = new double[res+buf];
    drhosummandtable[i][mn] = new double[res+buf];
    for (m = 0; m < (res + buf); m++) {
      r1 = rc[i][mn] * rc[i][mn] * (double) (m) / (double) (res);
      if (sqrt(r1)>=rc[i][mn]){
        phitable[i][mn][m]=0.0;
        rhosummandtable[i][mn][m]=0.0;
        dphitable[i][mn][m]=0.0;
        drhosummandtable[i][mn][m]=0.0;
      }
      else {
        double r = sqrt(r1);
        double f = cutofffunction(r,rc[i][mn],dr[i][mn]);
        double df = dcutofffunction(r,rc[i][mn],dr[i][mn]);
        phitable[i][i][m]=get_phi_single(r,i)*f/2;
        rhosummandtable[i][i][m]=cweight[mn]*exp(-beta0[mn]*(r/re[i][mn]-1))*f;
        dphitable[i][i][m]=(get_dphi_single(r,i)*f/2+phitable[i][i][m]*df)/r;
        drhosummandtable[i][i][m]=(-beta0[mn]/re[i][mn]*cweight[mn]*exp(-beta0[mn]*(r/re[i][mn]-1))*f+rhosummandtable[i][i][m]*df)/r;
      }
    }
  }
  for (i=0;i<nelements;i++){
    //do off-diagonal
    for (mn = 0; mn < nelements; mn++) {
      if (mn==i) continue;
      phitable[i][mn] = new double[res+buf];
      rhosummandtable[i][mn] = new double[res+buf];
      dphitable[i][mn] = new double[res+buf];
      drhosummandtable[i][mn] = new double[res+buf];
      for (m = 0; m < (res + buf); m++) {
        r1 = rc[i][mn] * rc[i][mn] * (double) (m) / (double) (res);
        if (sqrt(r1)>=rc[i][mn] || m==0){
          phitable[i][mn][m]=0.0;
          rhosummandtable[i][mn][m]=0.0;
          dphitable[i][mn][m]=0.0;
          drhosummandtable[i][mn][m]=0.0;
        }
        else {
          double r = sqrt(r1);
          double f = cutofffunction(r,rc[i][mn],dr[i][mn]);
          double df = dcutofffunction(r,rc[i][mn],dr[i][mn]);
          phitable[i][mn][m]=get_phi_binary(r,i,mn)*f/2;
          rhosummandtable[i][mn][m]=cweight[mn]*exp(-beta0[mn]*(r/re[mn][mn]-1))*f;
          dphitable[i][mn][m]=(get_dphi_binary(r,i,mn)*f/2+phitable[i][mn][m]*df)/r;
          drhosummandtable[i][mn][m]=(-beta0[mn]/re[mn][mn]*cweight[mn]*exp(-beta0[mn]*(r/re[mn][mn]-1))*f+rhosummandtable[i][mn][m]*df)/r;
        }
      }
    }
  }
  // FILE *file = fopen("eamtabletest.csv","w");
  // i=0;int j=1;
  // int I0 = lat[i][i]->typemap[i];
  // int J0 = lat[i][i]->typemap[i];
  // int I1 = lat[j][j]->typemap[j];
  // int J1 = lat[j][j]->typemap[j];
  // double rho0i;
  // printf("%f %f %d %d %f %f\n",lat[i][i]->Sij[I0][J0],lat[j][j]->Sij[I1][J1],lat[i][i]->Yij[I0][J0],lat[j][j]->Yij[I1][J1],lat[i][i]->aij,lat[j][j]->aij);
  // for (m=1;m<(res+buf);m++){
  //   r1 = rc[i][j] * rc[i][j] * (double) (m) / (double) (res);
  //   double r = sqrt(r1);
  //   fprintf(file,"%f, %.10f, %.10f, %.10f, %.10f\n",r,get_Fbar(r,0,0),get_Fbar(r,1,1),get_psi_single(r,0),get_psi_single(r,1));
  // }
  // fclose(file);
  //pair->errorf(FLERR,"stop");
}

double State_eamscreenedr::get_Fbar(double r,int i,int j){
  int I = lat[i][j]->typemap[i];
  int J = lat[i][j]->typemap[j];
  double rho0i;
  if (i!=j){
    rho0i = lat[i][j]->Z*cweight[j]*exp(-beta0[j]*(r/re[j][j]-1))+
                  lat[i][j]->Yij[I][I]*lat[i][j]->Sij[I][I]*cweight[i]*exp(-beta0[i]*(lat[i][j]->aij*r/re[i][i]-1))+
                  lat[i][j]->Yij[I][J]*lat[i][j]->Sij[I][J]*cweight[j]*exp(-beta0[j]*(lat[i][j]->aij*r/re[j][j]-1));
  }
  else {
    rho0i = lat[i][j]->Z*cweight[j]*exp(-beta0[j]*(r/re[i][j]-1))+
                  lat[i][j]->Yij[I][I]*lat[i][j]->Sij[I][I]*cweight[i]*exp(-beta0[i]*(lat[i][j]->aij*r/re[i][j]-1));
  }
  //Di is rho0i when r=re
  double Fi = Asub[i]*ec[i][i]*rho0i/D_ref[i]*log(rho0i/D_ref[i]);
  return Fi;
}

double State_eamscreenedr::get_dFbar(double r,int i,int j){
  int I = lat[i][j]->typemap[i];
  int J = lat[i][j]->typemap[j];
  double rho0i,drho0i;
  if (i!=j){
    rho0i = lat[i][j]->Z*cweight[j]*exp(-beta0[j]*(r/re[j][j]-1))+
                  lat[i][j]->Yij[I][I]*lat[i][j]->Sij[I][I]*cweight[i]*exp(-beta0[i]*(lat[i][j]->aij*r/re[i][i]-1))+
                  lat[i][j]->Yij[I][J]*lat[i][j]->Sij[I][J]*cweight[j]*exp(-beta0[j]*(lat[i][j]->aij*r/re[j][j]-1));
    drho0i = -beta0[j]*lat[i][j]->Z*cweight[j]/re[j][j]*exp(-beta0[j]*(r/re[j][j]-1))-
                  beta0[i]*lat[i][j]->aij/re[i][i]*lat[i][j]->Yij[I][I]*lat[i][j]->Sij[I][I]*cweight[i]*exp(-beta0[i]*(lat[i][j]->aij*r/re[i][i]-1))-
                  beta0[j]*lat[i][j]->aij/re[j][j]*lat[i][j]->Yij[I][J]*lat[i][j]->Sij[I][J]*cweight[j]*exp(-beta0[j]*(lat[i][j]->aij*r/re[j][j]-1));
  }
  else {
    rho0i = lat[i][j]->Z*cweight[i]*exp(-beta0[i]*(r/re[i][i]-1))+
                  lat[i][j]->Yij[I][I]*lat[i][j]->Sij[I][I]*cweight[i]*exp(-beta0[i]*(lat[i][j]->aij*r/re[i][i]-1));
    drho0i = -beta0[j]*lat[i][i]->Z*cweight[i]/re[i][i]*exp(-beta0[i]*(r/re[i][i]-1))-
                  beta0[i]*lat[i][i]->aij/re[i][i]*lat[i][i]->Yij[I][I]*lat[i][i]->Sij[I][I]*cweight[i]*exp(-beta0[i]*(lat[i][i]->aij*r/re[i][i]-1));
  }
  //Di is rho0i when r=re
  double dFi = Asub[i]*ec[i][i]/D_ref[i]*(log(rho0i/D_ref[i])+1)*drho0i;
  return dFi;
}

double State_eamscreenedr::get_rose(double r,int i,int j){
  return -ec[i][j]*(1+alpha[i][j]*(r/re[i][j]-1)+delta[i][j]*alpha[i][j]*alpha[i][j]*alpha[i][j]*(r/re[i][j]-1)*(r/re[i][j]-1)*(r/re[i][j]-1))*exp(-alpha[i][j]*(r/re[i][j]-1));
}


double State_eamscreenedr::get_drose(double r,int i,int j){
  return -ec[i][j]*exp(-alpha[i][j]*(r/re[i][j]-1))*(alpha[i][j]/re[i][j]+2*delta[i][j]*alpha[i][j]*alpha[i][j]/re[i][j]*(r/re[i][j]-1)+3*delta[i][j]*alpha[i][j]*alpha[i][j]*alpha[i][j]/re[i][j]*(r/re[i][j]-1)*(r/re[i][j]-1)+(1+alpha[i][j]*(r/re[i][j]-1)+delta[i][j]*alpha[i][j]*alpha[i][j]*alpha[i][j]*(r/re[i][j]-1)*(r/re[i][j]-1)*(r/re[i][j]-1))*-alpha[i][j]/re[i][j]);
}

double State_eamscreenedr::get_psi_single(double r,int i){
  if (r>=rc[i][i])return 0;
  double eroseii = get_rose(r,i,i);
  double Fi = get_Fbar(r,i,i);
  double psii = 2.0/lat[i][i]->Z*(eroseii-Fi);
  return psii;
}

double State_eamscreenedr::get_dpsi_single(double r,int i){
  if (r>=rc[i][i])return 0;
  double deroseii = get_drose(r,i,i);
  double dFi = get_dFbar(r,i,i);
  double dpsii = 2.0/lat[i][i]->Z*(deroseii-dFi);
  return dpsii;
}

double State_eamscreenedr::get_phi_single(double r,int i){
  int I = lat[i][i]->typemap[i];
  double phi = get_psi_single(r,i);
  double b = -lat[i][i]->Yij[I][I]*lat[i][i]->Sij[I][I]/lat[i][i]->Z;
  double a = lat[i][i]->aij;
  for (int ii=0;ii<10;ii++){
    phi+=b*get_psi_single(a*r,i);
    b*=-lat[i][i]->Yij[I][I]*lat[i][i]->Sij[I][I]/lat[i][i]->Z;
    a*= lat[i][i]->aij;
    if (a*r>rc[i][i])break;
  }
  return phi;
}

double State_eamscreenedr::get_dphi_single(double r,int i){
  int I = lat[i][i]->typemap[i];
  double dphi = get_dpsi_single(r,i);
  double b = -lat[i][i]->Yij[I][I]*lat[i][i]->Sij[I][I]/lat[i][i]->Z;
  double a = lat[i][i]->aij;
  for (int ii=1;ii<10;ii++){
    dphi+=b*get_dpsi_single(a*r,i)*a;
    b*=-lat[i][i]->Yij[I][I]*lat[i][i]->Sij[I][I]/lat[i][i]->Z;
    a*= lat[i][i]->aij;
    if (a*r>rc[i][i])break;
  }
  return dphi;
}

double State_eamscreenedr::interpolate(double *table,double r1){
  double output;
  double res = pair->res;
  int m1 = (int)r1;
  if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
  if (table[m1]==0) {output=0;}
  else {
    double *phip = &table[m1-1];
    r1 = r1-trunc(r1);
    output = phip[1] + 0.5 * r1*(phip[2] - phip[0] + r1*(2.0*phip[0] - 5.0*phip[1] + 4.0*phip[2] - phip[3] + r1*(3.0*(phip[1] - phip[2]) + phip[3] - phip[0])));
  }
  return output;
}

double State_eamscreenedr::get_psi_binary(double r,int i,int j){
  if (r>=rc[i][j])return 0;
  int I = lat[i][j]->typemap[i];
  int J = lat[i][j]->typemap[j];
  double eroseij = get_rose(r,i,j);
  double Fi = get_Fbar(r,i,j);
  double Fj = get_Fbar(r,j,i);
  int res = pair->res;
  double cutinv2 = 1/rc[i][j]/rc[i][j];
  double ar = r*lat[i][j]->aij;
  double r1 = (ar*ar*((double)res)*cutinv2);
  double psiij = 2*(eroseij-Fi)/lat[i][j]->Z;
  if (ar>=rc[i][j])return psiij;
  psiij = 2*(eroseij-Fi-(lat[i][j]->Yij[I][I]*lat[i][j]->Sij[I][I]*interpolate(phitable[i][i],r1)))/lat[i][j]->Z;
  return psiij;
}

double State_eamscreenedr::get_dpsi_binary(double r,int i,int j){
  if (r>=rc[i][j])return 0;
  int I = lat[i][j]->typemap[i];
  int J = lat[i][j]->typemap[j];
  double deroseij = get_drose(r,i,j);
  double dFi = get_dFbar(r,i,j);
  double dFj = get_dFbar(r,j,i);
  int res = pair->res;
  double cutinv2 = 1/rc[i][j]/rc[i][j];
  double ar = r*lat[i][j]->aij;
  double r1 = (ar*ar*((double)res)*cutinv2);
  double dpsiij = 2*(deroseij-dFi)/lat[i][j]->Z;
  if (ar>=rc[i][j])return dpsiij;
  dpsiij = 2*(deroseij-dFi-lat[i][j]->aij*ar*(lat[i][j]->Yij[I][I]*lat[i][j]->Sij[I][I]*interpolate(dphitable[i][i],r1)))/lat[i][j]->Z;
  return dpsiij;
}

double State_eamscreenedr::get_phi_binary(double r,int i,int j){
  int I = lat[i][j]->typemap[i];
  int J = lat[i][j]->typemap[j];
  double phi = get_psi_binary(r,i,j);
  double b = -lat[i][j]->Yij[I][J]*lat[i][j]->Sij[I][J]/lat[i][j]->Z;
  double a = lat[i][j]->aij;
  for (int ii=0;ii<10;ii++){
    phi+=b*get_psi_binary(a*r,i,j);
    b*=-lat[i][j]->Yij[I][J]*lat[i][j]->Sij[I][J]/lat[i][j]->Z;
    a*= lat[i][j]->aij;
    if (a*r>rc[i][j])break;
  }
  return phi;
}

double State_eamscreenedr::get_dphi_binary(double r,int i,int j){
  int I = lat[i][j]->typemap[i];
  int J = lat[i][j]->typemap[j];
  double dphi = get_dpsi_binary(r,i,j);
  double b = -lat[i][j]->Yij[I][J]*lat[i][j]->Sij[I][J]/lat[i][j]->Z;
  double a = lat[i][j]->aij;
  for (int ii=0;ii<10;ii++){
    dphi+=b*get_dpsi_binary(a*r,i,j)*a;
    b*=-lat[i][j]->Yij[I][J]*lat[i][j]->Sij[I][J]/lat[i][j]->Z;
    a*= lat[i][j]->aij;
    if (a*r>rc[i][j])break;
  }
  return dphi;
}

void State_eamscreenedr::write_values(FILE *fid) {
  int i,j;
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:ec:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    for (j=0;j<pair->nelements;j++) {
      fprintf(fid,"%f ",ec[i][j]);
    }
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:re:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    for (j=0;j<pair->nelements;j++) {
      fprintf(fid,"%f ",re[i][j]);
    }
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:rc:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    for (j=0;j<pair->nelements;j++) {
      fprintf(fid,"%f ",rc[i][j]);
    }
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:alpha:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    for (j=0;j<pair->nelements;j++) {
      fprintf(fid,"%f ",alpha[i][j]);
    }
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:delta:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    for (j=0;j<pair->nelements;j++) {
      fprintf(fid,"%f ",delta[i][j]);
    }
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:dr:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    for (j=0;j<pair->nelements;j++) {
      fprintf(fid,"%f ",dr[i][j]);
    }
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:cweight:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    fprintf(fid,"%f ",cweight[i]);
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:beta:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    fprintf(fid,"%f ",beta0[i]);
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:Asub:\n",style,id);
  for (i=0;i<pair->nelements;i++) {
    fprintf(fid,"%f ",Asub[i]);
  }
  fprintf(fid,"\n");
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:lattice:\n",style,id);
  for (i=0; i<pair->nelements;i++) {
    for (j=0;j<pair->nelements;j++) {
      if (lat[i][j]->typemap[i]==0){
        fprintf(fid,"%s ",lat[i][j]->style);
      }
      else {
        fprintf(fid,"%s ","empty");
      }
    }
  }
  fprintf(fid,"\n");
}

