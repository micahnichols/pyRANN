/* -*- c++ -*- ----------------------------------------------------------
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

----------------*/

#ifndef LMP_RANN_EAM_REF_H
#define LMP_RANN_EAM_REF_H

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "rann_stateequation.h"
#include "pair_spin_rann.h"

namespace LAMMPS_NS {
namespace RANN {
  class State;
  class Reference_lattice {
   public:
    Reference_lattice(State *);
    virtual ~Reference_lattice();
    int *typemap,*invtypemap;
    //part 1: defined by particular lattice constructor
    double **x; //atom coordinates in unitcell
    double **box;
    double origin[3];   
    int n_body_type;    //single or binary
    const char *style; //lattice name
    int natoms;
    int *type;
    //part 2 computed by functions below:
    int *id,*ilist,*numneigh,**firstneigh,inum,gnum;
    double snn;
    int Z;
    double Sij[2][2];
    int Yij[2][2];
    int Iij[2][2];
    double aij;

    State *state;
    virtual void compute_reference_parameters();
    virtual void create_neighbor_lists();
    virtual void cull_neighbor_list(double *, double *, double *, int *, int *, int *, int, int);
    virtual void screen(double *, bool *, int, int, double *, double *, double *, int *, int);
    virtual void screen_neighbor_list(double *, double *, double *, int *, int *, int *, int, int, bool *, double *);
    virtual void get_neighbor_parameters(int *,double *,double *, double *, int *, int *, int *, int, int, double*);
    void qrsolve(double *,int,int,double*,double *);
    //parameters used to create neighbor list. May need to be overwritten for some lattices:
    double neighborbandwidth=0.2;
    double cutmax = 2.5;
    bool empty=true;
  };

}    // namespace RANN
}    // namespace LAMMPS_NS

#endif /* RANN_FINGERPRINT_H_ */
