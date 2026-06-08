/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/*  ----------------------------------------------------------------------
   Contributing author: Christopher Barrett (MSU) barrett@me.msstate.edu
    ----------------------------------------------------------------------*/

#ifndef STATEEQUATION_H_
#define STATEEQUATION_H_
#include <string>
#include <vector>
#include <cmath>

namespace LAMMPS_NS {
  class PairRANN;
  namespace RANN {
    class State {
      public:
        State(PairRANN *_pair){
          empty = true;
          style = "empty";
          fullydefined = true;
          screen = false;
          spin = false;
          pair = _pair;
          rc = 0.0;
        };
        virtual ~State(){};
        virtual void eos_function(double*,double**,int,int,double*,double*,double*,int*,int,int*){}//noscreen,nospin
        virtual void eos_function(double*,double**,double*,double*,double*,double*,double*,double*,
                                  double*,bool*,int,int,double*,double*,double*,int*,int,int*){}//screen,nospin
        virtual void eos_function(double*,double**,double**,int,int,double*,double*,double*,int*,int,int*){}//noscreen,spin
        virtual void eos_function(double*,double**,double**,double*,double*,double*,double*,double*,double*,
                                  double*,bool*,int,int,double*,double*,double*,int*,int,int*){}//screen,spin
        virtual bool parse_values(std::string, std::vector<std::string>){return false;}
        virtual double cutofffunction(double r, double rc, double dr){
          double out;
          if (r < (rc - dr))
            out = 1;
          else if (r > rc)
            out = 0;
          else {
            out = 1 - (rc - r) / dr;
            out *= out;
            out *= out;
            out = 1 - out;
            out *= out;
          }
          return out;
        };

        virtual double dcutofffunction(double r, double rc, double dr){
          if (r>=rc || r <= (rc-dr)) {
            return 0;
          }
          return -8*pow(1-(rc-r)/dr,3)/dr/(1-pow(1-(rc-r)/dr,4));
        }

        virtual void allocate() {}
        virtual void write_values(FILE *) {}
        virtual void init(int*,int){}
        bool empty;
        double rc;
        bool fullydefined;
        const char *style;
        bool screen;
        bool spin;
        int n_body_type;    //i-j vs. i-j-k vs. i-j-k-l, etc.
        int *atomtypes;
        int id; //based on ordering of state equations listed for i-j in potential file
        PairRANN *pair;
        double cutmax = 3.0;
        double nlattices;

    };
  }//RANN
}//LAMMPS_NS

#endif /* STATEEQUATION_H_ */