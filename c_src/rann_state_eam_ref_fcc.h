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
#ifndef LMP_RANN_STATE_EAM_REF_FCC_H
#define LMP_RANN_STATE_EAM_REF_FCC_H

#include "rann_state_eam_ref.h"

namespace LAMMPS_NS {
namespace RANN {

  class Ref_FCC : public Reference_lattice {
   public:
    Ref_FCC (State *_state) : Reference_lattice(_state) {
      //Part 1: define lattice data:
      empty = false;
      n_body_type = 1;//single or binary
      style = "fcc";//lattice name
      natoms = 4;
      int atomtypes1 [] = {0,0,0,0};
      double box1 [3][3] = {{1,0,0},{0,1,0},{0,0,1}};
      double atoms1[][3] = {{0,0,0},
                                  {0.5,0.5,0},
                                  {0.5,0,0.5},
                                  {0,0.5,0.5}};
      double origin1[3] = {0,0,0};
      //Part 2: save lattice data to struct
      type = new int[natoms];
      x = new double *[natoms];
      box = new double*[3];
      for (int i = 0;i<3;i++){
        box[i] = new double[3];
        origin[i] = origin1[i];
        for (int j = 0;j<3;j++){
          box[i][j] = box1[i][j];
        }
      }
      for (int i = 0;i<natoms;i++){
        x[i] = new double[3];
        type[i] = atomtypes1[i];
        for (int j = 0;j<3;j++){
          x[i][j] = atoms1[i][j];
        }
      }
    };
    ~Ref_FCC() {

    };
  };

}    // namespace RANN
}    // namespace LAMMPS_NS

#endif /* ACTIVATION_CAPPED_H_ */
