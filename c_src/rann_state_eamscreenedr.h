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
 */

#ifndef LMP_RANN_STATE_EAMSCREENEDR_H
#define LMP_RANN_STATE_EAMSCREENEDR_H

#include "rann_stateequation.h"
#include <string>
namespace LAMMPS_NS {
namespace RANN {

class Reference_lattice;

  class State_eamscreenedr : public State {
   public:
    State_eamscreenedr(class PairRANN *);
    ~State_eamscreenedr();
    void eos_function(double*,double**,double*,double*,double*,
                      double*,double*,double*,double*,bool*,int,
                      int,double*,double*,double*,int*,int,int*);
    bool parse_values(std::string, std::vector<std::string>);
    void allocate();
    void write_values(FILE *);
    void init(int*,int);
    Reference_lattice *create_lattice(const char *);
    double get_psi_single(double,int);
    double get_dpsi_single(double,int);
    double get_Fbar(double,int,int);
    double get_dFbar(double,int,int);
    double get_psi_binary(double,int,int);
    double get_dpsi_binary(double,int,int);
    double get_phi_single(double,int);
    double get_dphi_single(double,int);
    double get_phi_binary(double,int,int);
    double get_dphi_binary(double,int,int);
    double get_rose(double,int,int);
    double get_drose(double,int,int);
    double interpolate(double*,double,int,int);
    double interpolate(double*,double);
    double **ec;
    double **re;
    double **alpha;
    double **delta;
    double *cweight;
    double **dr;
    double **rc;
    double *beta0;
    double *Asub;
    double *D_ref;
    Reference_lattice ***lat;
    double ***rhosummandtable;
    double ***phitable;
    double ***drhosummandtable;
    double ***dphitable;
    int *map;
  };



}    // namespace RANN
}    // namespace LAMMPS_NS

#endif /* LMP_RANN_STATE_EAMSCREENEDR_H */
