// clang-format off
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

#include "rann_fingerprint_temperature.h"
#include "pair_spin_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

Fingerprint_temperature::Fingerprint_temperature(PairRANN *_pair) : Fingerprint(_pair)
{
  n_body_type = 1;
  rc = 0;
  id = -1;
  style = "temperature";
  atomtypes = new int[n_body_type];
  empty = false;
  fullydefined = true;
  _pair->allscreen = false;
}

Fingerprint_temperature::~Fingerprint_temperature()
{
  delete[] atomtypes;
}

bool Fingerprint_temperature::parse_values(std::string constant,std::vector<std::string> line1) {
  return false;
}

void Fingerprint_temperature::write_values(FILE *fid) {
 
}

//called after fingerprint is fully defined and tables can be computed.
void Fingerprint_temperature::allocate()
{
}

//called after fingerprint is declared for i-j type, but before its parameters are read.
void Fingerprint_temperature::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}

void Fingerprint_temperature::compute_fingerprint(double * features,double * dfeaturesx,double *dfeaturesy,double *dfeaturesz,int ii,int sid,double *xn,double *yn,double*zn,int *tn,int jnum,int * /*jl*/)
{
  PairRANN::Simulation *sim = &pair->sims[sid];
  int count=startingneuron;
  features[count]=sim->temp;
}

int Fingerprint_temperature::get_length()
{
  return 1;
}
