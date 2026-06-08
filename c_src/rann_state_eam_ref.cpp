/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/*  ----------------------------------------------------------------------
   Contributing authors: Christopher Barrett (MSU) barrett@me.msstate.edu
                              Doyl Dickel (MSU) doyl@me.msstate.edu
    ----------------------------------------------------------------------*/
/*
“The research described and the resulting data presented herein, unless
otherwise noted, was funded under PE 0602784A, Project T53 "Military
Engineering Applied Research", Task 002 under Contract No. W56HZV-17-C-0095,
managed by the U.S. Army Combat Capabilities Development Command (CCDC) and
the Engineer Research and Development Center (ERDC).  The work described in
this document was conducted at CAVS, MSU.  Permission was granted by ERDC
to publish this information. Any opinions, findings and conclusions or
recommendations expressed in this material are those of the author(s) and
do not necessarily reflect the views of the United States Army.​”

DISTRIBUTION A. Approved for public release; distribution unlimited. OPSEC#4918
 */
#include "rann_stateequation.h"
#include "rann_state_eam_ref.h"


using namespace LAMMPS_NS::RANN;

Reference_lattice::Reference_lattice(State *_state)
{
      state = _state;//store a pointer to the general state class to access pair and other needed objects and methods
      empty = true;
      natoms = 0;
      style = "empty";
}

Reference_lattice::~Reference_lattice(){
  int i;
  if (!empty){
    for (i=0;i<natoms;i++){
      delete [] x[i];
      delete [] firstneigh[i];
    }
    for (i=0;i<3;i++){
      delete [] box[i];
    }
    delete [] x;
    delete [] box;
    delete [] firstneigh;
    delete [] type;
    delete [] id;
    delete [] ilist;
    delete [] numneigh;
    delete [] typemap;
  }
}

void Reference_lattice::compute_reference_parameters(){
  int i;
  bool types[2];
  types[0]=types[1]=0;
  create_neighbor_lists();
  for (i=0;i<natoms;i++){
    if (types[type[i]]==true)continue;
    types[type[i]]=true;
    int jnum = numneigh[i];
    //TO DO: move away from stack allocation
    double xn[jnum],yn[jnum],zn[jnum];
    int tn[jnum],jl[jnum];
    cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,0);
    double Sik[jnum];
    bool Bij[jnum];
    screen(Sik,Bij,i,0,xn,yn,zn,tn,jnum-1);
    screen_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,0,Bij,Sik);
    int *s;
    get_neighbor_parameters(s,xn,yn,zn,tn,&jnum,jl,i,0,Sik);
  }
}

// https://docs.lammps.org/Howto_triclinic.html
void triclinic2uppertriangular_ref(double **box,double **x,int natoms,State *state){
	double x1[natoms][3];
	double norm[3];
	int i,j,k;
	double Ax = box[0][0],Ay=box[1][0],Az=box[2][0];
	double Bx = box[0][1],By=box[1][1],Bz=box[2][1];
	double Cx = box[0][2],Cy=box[1][2],Cz=box[2][2];
  norm[0] = sqrt(Ax*Ax+Ay*Ay+Az*Az);
  norm[1] = sqrt(Bx*Bx+By*By+Bz*Bz);
  norm[2] = sqrt(Cx*Cx+Cy*Cy+Cz*Cz);
	double ax = norm[0];
	double bx = (Bx*Ax+By*Ay+Bz*Az)/ax;
	double by = sqrt(norm[1]*norm[1]-bx*bx);
	double cx = (Cx*Ax+Cy*Ay+Cz*Az)/ax;
	double cy = (Bx*Cx+By*Cy+Bz*Cz-bx*cx)/by;
	double cz = sqrt(norm[2]*norm[2]-cx*cx-cy*cy);
	double box2[3][3] = {{ax,bx,cx},{0,by,cy},{0,0,cz}};
	double V = ax*by*cz;
  if (V<0){state->pair->errorf(FLERR,"left-handed box found");}
	double R[3][3] = {{(ax*(By*Cz - Bz*Cy) + bx*(-Ay*Cz + Az*Cy) + cx*(Ay*Bz - Az*By))/(ax*by*cz), (ax*(-Bx*Cz + Bz*Cx) + bx*(Ax*Cz - Az*Cx) + cx*(-Ax*Bz + Az*Bx))/(ax*by*cz),  (ax*(Bx*Cy - By*Cx) + bx*(-Ax*Cy + Ay*Cx) + cx*(Ax*By - Ay*Bx))/(ax*by*cz)},
                    {(by*(-Ay*Cz + Az*Cy) + cy*(Ay*Bz - Az*By))/(ax*by*cz),                      (by*(Ax*Cz - Az*Cx) + cy*(-Ax*Bz + Az*Bx))/(ax*by*cz),                        (by*(-Ax*Cy + Ay*Cx) + cy*(Ax*By - Ay*Bx))/(ax*by*cz)},
                    {(Ay*Bz - Az*By)/(ax*by),                                                    (-Ax*Bz + Az*Bx)/(ax*by),                                                     (Ax*By - Ay*Bx)/(ax*by)}};
	for (i=0;i<natoms;i++){
		for (j=0;j<3;j++){
      x1[i][j]=0.0;
			for (k=0;k<3;k++){
				x1[i][j]+=R[j][k]*x[i][k];
			}
		}
	}
	for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			box[i][j]=box2[i][j];
		}
	}
  for (i=0;i<natoms;i++){
    for (j=0;j<3;j++){
      x[i][j]=x1[i][j];
    }
  }
}

  void Reference_lattice::get_neighbor_parameters(int *s,double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,double* Sik){
    //sort by ascending radius
    s = new int[jnum[0]-1];
    int j,k;
    int itype = type[i];
    for (j=0;j<jnum[0]-1;j++)s[j]=j;
    for (j=0;j<jnum[0]-1;j++){
      double r2j = xn[s[j]]*xn[s[j]]+yn[s[j]]*yn[s[j]]+zn[s[j]]*zn[s[j]];
      for (k=j+1;k<jnum[0]-1;k++){
        double r2k = xn[s[k]]*xn[s[k]]+yn[s[k]]*yn[s[k]]+zn[s[k]]*zn[s[k]];
        if (r2k<r2j) {
          int pivot = s[j];
          s[j] = s[k];
          s[k] = pivot;
          r2j = r2k;
        }
      }
    }
    //count 1st neighbor shell
    double r0 = xn[s[0]]*xn[s[0]]+yn[s[0]]*yn[s[0]]+zn[s[0]]*zn[s[0]];
    int c = 1;
    for (j=1;j<jnum[0]-1;j++){
      double r1 = xn[s[j]]*xn[s[j]]+yn[s[j]]*yn[s[j]]+zn[s[j]]*zn[s[j]];
      if (r1<neighborbandwidth+r0){
        c++;
      }
      else {
        Z = c;
        aij = sqrt(r1/r0);
        r0 = r1;
        break;
      }
    }
    //count 2nd neighbor shell
    int c2[2];
    c2[0]=c2[1]=0;
    if (c<jnum[0]-1){
      c2[tn[s[c]]]++;
    }
    else {
      Z = c;
      aij = 1;
      Yij[itype][0]=c2[0];
      Yij[itype][1]=c2[1];
      Sij[itype][0]=1.0;
      Sij[itype][1]=1.0;
      Iij[itype][0]=1.0;
      Iij[itype][1]=1.0;
      return;
    }
    Iij[itype][tn[s[c]]]=j;
    Sij[itype][tn[s[c]]]=Sik[s[c]];
    for (j=c+1;j<jnum[0]-1;j++){
      double r1 = xn[s[j]]*xn[s[j]]+yn[s[j]]*yn[s[j]]+zn[s[j]]*zn[s[j]];
      if (r1<neighborbandwidth+r0){
        c2[tn[s[j]]]++;
      }
      else {
        break;
      }
    }
    Yij[itype][0]=c2[0];
    Yij[itype][1]=c2[1];
  };


  void Reference_lattice::create_neighbor_lists(){
    //brute force search technique rather than tree search because we only do it once and most simulations are small.
    //I did optimize for low memory footprint by only adding ghost neighbors
    //within cutoff distance of the box
    int i,ix,iy,iz,j,k;
    double buffer = 0.01;//over-generous compensation for roundoff error
    i = 0;
    int inum = natoms;
    triclinic2uppertriangular_ref(box,x,natoms,state);
    int xb = floor(cutmax/box[0][0]+1);
    int yb = floor(cutmax/box[1][1]+1);
    int zb = floor(cutmax/box[2][2]+1);
    int buffsize = natoms*(xb*2+1)*(yb*2+1)*(zb*2+1);
    double x1[buffsize][3];
    int type1[buffsize];
    int id1[buffsize];
    double spins[buffsize][3];
    int count = 0;

    //force all atoms to be inside the box:
    double xtemp[3];
    double xp[3];
    double boxt[9];
    for (j=0;j<3;j++){
      for (k=0;k<3;k++){
        boxt[j*3+k]=box[j][k];
      }
    }
    for (j=0;j<natoms;j++){
      for (k=0;k<3;k++){
        xp[k] = x[j][k]-origin[k];
      }
      qrsolve(boxt,3,3,xp,xtemp);//convert coordinates from Cartesian to box basis
      for (k=0;k<3;k++){
        xtemp[k]-=floor(xtemp[k]);//if atom is outside box find periodic replica in box
      }
      for (k=0;k<3;k++){
        x[j][k] = 0.0;
        for (int l=0;l<3;l++){
          x[j][k]+=box[k][l]*xtemp[l];//convert back to Cartesian
        }
        x[j][k]+=origin[k];
      }
    }

    //calculate box face normal directions and plane intersections
    double xpx,xpy,xpz,ypx,ypy,ypz,zpx,zpy,zpz;
    zpx = 0;zpy=0;zpz =1;
    double ym,xm;
    ym = sqrt(box[1][2]*box[1][2]+box[2][2]*box[2][2]);
    xm = sqrt(box[1][1]*box[2][2]*box[1][1]*box[2][2]+box[0][1]*box[0][1]*box[2][2]*box[2][2]+(box[0][1]*box[1][2]-box[0][2]*box[1][1])*(box[0][1]*box[1][2]-box[0][2]*box[1][1]));
    //unit vectors normal to box faces:
    ypx = 0;
    ypy = box[2][2]/ym;
    ypz = -box[1][2]/ym;
    xpx = box[1][1]*box[2][2]/xm;
    xpy = -box[0][1]*box[2][2]/xm;
    xpz = (box[0][1]*box[1][2]-box[0][2]*box[1][1])/xm;
    double fxn,fxp,fyn,fyp,fzn,fzp;
    //minimum distances from origin to planes aligned with box faces:
    fxn = origin[0]*xpx+origin[1]*xpy+origin[2]*xpz;
    fyn = origin[0]*ypx+origin[1]*ypy+origin[2]*ypz;
    fzn = origin[0]*zpx+origin[1]*zpy+origin[2]*zpz;
    fxp = (origin[0]+box[0][0])*xpx+(origin[1]+box[1][0])*xpy+(origin[2]+box[2][0])*xpz;
    fyp = (origin[0]+box[0][1])*ypx+(origin[1]+box[1][1])*ypy+(origin[2]+box[2][1])*ypz;
    fzp = (origin[0]+box[0][2])*zpx+(origin[1]+box[1][2])*zpy+(origin[2]+box[2][2])*zpz;
    //fill buffered atom list
    double px,py,pz;
    double xe,ye,ze;
    double theta,sx,sy,sz;
    for (j=0;j<natoms;j++){
      x1[count][0] = x[j][0];
      x1[count][1] = x[j][1];
      x1[count][2] = x[j][2];
      type1[count] = type[j];
      id1[count] = j;
      count++;
    }

    //add ghost atoms outside periodic boundaries:
    for (ix=-xb;ix<=xb;ix++){
      for (iy=-yb;iy<=yb;iy++){
        for (iz=-zb;iz<=zb;iz++){
          if (ix==0 && iy == 0 && iz == 0)continue;
          for (j=0;j<natoms;j++){
            xe = ix*box[0][0]+iy*box[0][1]+iz*box[0][2]+x1[j][0];
            ye = iy*box[1][1]+iz*box[1][2]+x1[j][1];
            ze = iz*box[2][2]+x1[j][2];
            px = xe*xpx+ye*xpy+ze*xpz;
            py = xe*ypx+ye*ypy+ze*ypz;
            pz = xe*zpx+ye*zpy+ze*zpz;
            //include atoms if their distance from the box face is less than cutmax
            if (px>cutmax+fxp+buffer || px<fxn-cutmax-buffer){continue;}
            if (py>cutmax+fyp+buffer || py<fyn-cutmax-buffer){continue;}
            if (pz>cutmax+fzp+buffer || pz<fzn-cutmax-buffer){continue;}
            x1[count][0] = xe;
            x1[count][1] = ye;
            x1[count][2] = ze;
            type1[count] = type[j];
            id1[count] = j;
            count++;
            if (count>buffsize){state->pair->errorf(FLERR,"neighbor overflow!\n");}
          }
        }
      }
    }
    //update stored lists
    buffsize = count;
    for (j=0;j<natoms;j++){
      delete [] x[j];
    }
    delete [] x;
    delete [] type;
    //delete [] ilist;
    type = new int [buffsize];
    x = new double *[buffsize];
    id = new int [buffsize];
    ilist = new int [buffsize];

    for (j=0;j<buffsize;j++){
      x[j] = new double [3];
      for (k=0;k<3;k++){
        x[j][k] = x1[j][k];
      }
      type[j] = type1[j];
      id[j] = id1[j];
      ilist[j] = j;
    }
    inum = natoms;
    gnum = buffsize-natoms;
    numneigh = new int[natoms];
    firstneigh = new int*[natoms];
    //do double count, slow, but enables getting the exact size of the neighbor list before filling it.
    for (j=0;j<natoms;j++){
      numneigh[j]=0;
      for (k=0;k<buffsize;k++){
        if (k==j)continue;
        double xtmp = x[j][0]-x[k][0];
        double ytmp = x[j][1]-x[k][1];
        double ztmp = x[j][2]-x[k][2];
        double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
        if (r2<cutmax*cutmax){
          numneigh[j]++;
        }
      }
      firstneigh[j] = new int[numneigh[j]];
      count = 0;
      for (k=0;k<buffsize;k++){
        if (k==j)continue;
        double xtmp = x[j][0]-x[k][0];
        double ytmp = x[j][1]-x[k][1];
        double ztmp = x[j][2]-x[k][2];
        double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
        if (r2<cutmax*cutmax){
          firstneigh[j][count] = k;
          count++;
        }
      }
    }
  };

  void Reference_lattice::cull_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn){
    int *jlist,j,count,jj,jtype;
    double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
    double cutmax = state->cutmax;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    count = 0;
    for (jj=0;jj<jnum[0];jj++){
      j = jlist[jj];
      j &= NEIGHMASK;
      //jtype = pair->map[type[j]];
      jtype = type[j];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq>cutmax*cutmax){
        continue;
      }
      xn[count]=delx;
      yn[count]=dely;
      zn[count]=delz;
      tn[count]=jtype;
      //jl[count]=sims[sn].id[j];
      jl[count]=j;
      //jl is currently only used to calculate spin dot products.
      //j includes ghost atoms. id maps back to atoms in the box across periodic boundaries.
      //lammps code uses id instead of j because spin spirals are not supported.
      count++;
    }
    jnum[0]=count+1;
  };

  void Reference_lattice::screen(double *Sik, bool *Bij, int ii,int sid,double *xn,double *yn,double *zn,int *tn,int jnum)
  {
    
    //see Baskes, Materials Chemistry and Physics 50 (1997) 152-1.58
    int i,*jlist,jj,j,kk,k,itype,jtype,ktype;
    double Sijk,Cijk,Cn,Cd,C;
    double xtmp,ytmp,ztmp,delx,dely,delz,rij,delx2,dely2,delz2,rik,delx3,dely3,delz3,rjk;
    i = ilist[ii];
    //itype = pair->map[sim->type[i]];
    itype = invtypemap[type[i]];
    double cutmax = state->cutmax;
    int nelements = state->pair->nelements;
    for (int jj=0;jj<jnum;jj++){
      Sik[jj]=1;
      Bij[jj]=true;
    }

    for (kk=0;kk<jnum;kk++){//outer sum over k in accordance with source, some others reorder to outer sum over jj
      if (Bij[kk]==false){continue;}
      ktype = invtypemap[tn[kk]];
      delx2 = xn[kk];
      dely2 = yn[kk];
      delz2 = zn[kk];
      rik = delx2*delx2+dely2*dely2+delz2*delz2;
      if (rik>cutmax*cutmax){
        Bij[kk]= false;
        continue;
      }
      for (jj=0;jj<jnum;jj++){
        if (jj==kk){continue;}
        if (Bij[jj]==false){continue;}
        jtype = invtypemap[tn[jj]];
        delx = xn[jj];
        dely = yn[jj];
        delz = zn[jj];
        rij = delx*delx+dely*dely+delz*delz;
        if (rij>cutmax*cutmax){
          Bij[jj] = false;
          continue;
        }
        delx3 = delx2-delx;
        dely3 = dely2-dely;
        delz3 = delz2-delz;
        rjk = delx3*delx3+dely3*dely3+delz3*delz3;
        if (rik+rjk-rij<1e-13){continue;}//bond angle > 90 degrees
        if (rik+rij-rjk<1e-13){continue;}//bond angle > 90 degrees
        double Cmax = state->pair->screening_max[itype*nelements*nelements+jtype*nelements+ktype];
        double Cmin = state->pair->screening_min[itype*nelements*nelements+jtype*nelements+ktype];
        double temp1 = rij-rik+rjk;
        Cn = temp1*temp1-4*rij*rjk;
        temp1 = rij-rjk;
        Cd = temp1*temp1-rik*rik;
        Cijk = Cn/Cd;
        C = (Cijk-Cmin)/(Cmax-Cmin);
        if (C>=1){continue;}
        else if (C<=0){
          Bij[kk]=false;
          break;
        }
        temp1 = 1-C;
        temp1 *= temp1;
        temp1 *= temp1;
        Sijk = 1-temp1;
        Sijk *= Sijk;
        Sik[kk] *= Sijk;
      }
    }
  };

  void Reference_lattice::screen_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,bool *Bij,double *Sik){
    double xnc[jnum[0]],ync[jnum[0]],znc[jnum[0]];
    double Sikc[jnum[0]];
    int jj,kk,count,tnc[jnum[0]],jlc[jnum[0]];
    count = 0;
    for (jj=0;jj<jnum[0]-1;jj++){
      if (Bij[jj]){
        xnc[count]=xn[jj];
        ync[count]=yn[jj];
        znc[count]=zn[jj];
        tnc[count]=tn[jj];
        jlc[count]=jl[jj];
        Sikc[count]=Sik[jj];
        count++;
      }
    }
    jnum[0]=count+1;
    for (jj=0;jj<count;jj++){
      xn[jj]=xnc[jj];
      yn[jj]=ync[jj];
      zn[jj]=znc[jj];
      tn[jj]=tnc[jj];
      jl[jj]=jlc[jj];
      Bij[jj] = true;
      Sik[jj]=Sikc[jj];
    }
  };

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
//replaced with Cholesky solution for greater speed for finding solve step. Still used to process input data.
void Reference_lattice::qrsolve(double *A,int m,int n,double *b, double *x_){
	double QR_[m*n];
//	char str[MAXLINE];
	double Rdiag[n];
	int i=0, j=0, k=0;
	int j_off, k_off;
	double nrm;
    // loop to copy QR from A.
	for (k=0;k<n;k++){
		k_off = k*m;
		for (i=0;i<m;i++){
			QR_[k_off+i]=A[i*n+k];
		}
	}
    for (k = 0; k < n; k++) {
       // Compute 2-norm of k-th column.
       nrm = 0.0;
       k_off = k*m;
       for (i = k; i < m; i++) {
			nrm += QR_[k_off+i]*QR_[k_off+i];
       }
       if (nrm==0.0){
    	   state->pair->errorf(FLERR,"Jacobian is rank deficient!\n");
       }
       nrm = sqrt(nrm);
	   // Form k-th Householder vector.
	   if (QR_[k_off+k] < 0) {
		 nrm = -nrm;
 	   }
	   for (i = k; i < m; i++) {
		 QR_[k_off+i] /= nrm;
	   }
	   QR_[k_off+k] += 1.0;

	   // Apply transformation to remaining columns.
	   for (j = k+1; j < n; j++) {
		 double s = 0.0;
		 j_off = j*m;
		 for (i = k; i < m; i++) {
			s += QR_[k_off+i]*QR_[j_off+i];
		 }
		 s = -s/QR_[k_off+k];
		 for (i = k; i < m; i++) {
			QR_[j_off+i] += s*QR_[k_off+i];
		 }
	   }
       Rdiag[k] = -nrm;
    }
    //loop to find least squares
    for (int j=0;j<m;j++){
    	x_[j] = b[j];
    }
    // Compute Y = transpose(Q)*b
	for (int k = 0; k < n; k++)
	{
		k_off = k*m;
		double s = 0.0;
		for (int i = k; i < m; i++)
		{
		   s += QR_[k_off+i]*x_[i];
		}
		s = -s/QR_[k_off+k];
		for (int i = k; i < m; i++)
		{
		   x_[i] += s*QR_[k_off+i];
		}
	}
	// Solve R*X = Y;
	for (int k = n-1; k >= 0; k--)
	{
		k_off = k*m;
		x_[k] /= Rdiag[k];
		for (int i = 0; i < k; i++) {
		   x_[i] -= x_[k]*QR_[k_off+i];
		}
	}
}


