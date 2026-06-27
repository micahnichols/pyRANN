/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Alexander Shapeev, Ivan Novikov
 */


#ifndef MLIP_PAIR_POTENTIALS_H
#define MLIP_PAIR_POTENTIALS_H


#include "basic_potentials.h"


class PairPotential : public AnyPotential
{
public:
    double max_cutoff = 5.0;

    virtual double F(double r, int type1=0, int type2=0) = 0;
    virtual double dF_dr(double r, int type1=0, int type2=0) = 0;

    void CalcEFS(Configuration& cfg)
    {
        ResetEFS(cfg);
        cfg.has_site_energies(true);
        memset(&cfg.site_energy(0), 0, cfg.size()*sizeof(double));

        if (cfg.nbh_cutoff != max_cutoff)
            cfg.InitNbhs(max_cutoff);

        for (int i=0; i<cfg.size(); i++) {
            Neighborhood& nbh = cfg.nbhs[i];
            for (int j=0; j<nbh.count; j++)
            {
                double r = nbh.dists[j];

                cfg.site_energy(i) += 0.5*F(r, nbh.my_type, nbh.types[j]);

                for (int a=0; a<3; a++)
                {
                    double force_val = 0.5 * dF_dr(r, nbh.my_type, nbh.types[j]) * nbh.vecs[j][a] / r;
                    cfg.force(nbh.my_ind, a) += force_val;
                    cfg.force(nbh.inds[j], a) -= force_val;
                    for (int b=0; b<3; b++)
                        cfg.stresses[a][b] -= force_val * nbh.vecs[j][b];
                }
            }
            cfg.energy += cfg.site_energy(i);
        }
    }
};


class TestPotential : public PairPotential
{
public:
    double r0;
    double alpha;

    TestPotential(  double _r0, double _alpha, double _cutoff) : 
        r0(_r0), alpha(_alpha) 
    {
        max_cutoff = _cutoff;
    };

    double F(double r, int type1=0, int type2=0) override
    {
        //  return exp(-2 * alpha * (r - r0)) - 2 * exp(-alpha * (r - r0));
        double tmp = exp(-alpha * (r - r0));
        return tmp * (tmp - 2.0);
        //      return -2.0 / r;
    }

    double dF_dr(double r, int type1=0, int type2=0) override
    {
        //  return -2 * exp(-2 * alpha * (r - r0)) + 2 * exp(-alpha * (r - r0));
        double tmp = exp(-alpha * (r - r0));
        return 2.0 * alpha * tmp * (1 - tmp);
        //      return 2.0 / (r*r);
    }
};

// TODO: rename to LennardJones
class LJ : public PairPotential, protected InitBySettings
{
private:
    double r_min; // minimizer of the pair function (in Angstroms)
    double scale; // Value of pair function at the minimum (in eV)

    void InitSettings() // Sets correspondence between variables and setting names in settings file
    {
        MakeSetting(r_min, "r_min");
        MakeSetting(scale, "scale");
        MakeSetting(max_cutoff, "cutoff");
    }

public:
    LJ(const Settings& settings)
    {
        InitSettings();
        ApplySettings(settings);
    }

    LJ(double _r_min, double _scale, double _cutoff) :
        r_min(_r_min), scale(_scale)
    {
        max_cutoff = _cutoff;
    };


    double F(double r, int type1=0, int type2=0) override
    {
        r = r_min/r;
        double r6 = pow(r*r, 3);
        double r12 = r6*r6;
        return scale * (r12 - 2.0*r6);
    }

    double dF_dr(double r, int type1=0, int type2=0) override
    {
        r = r_min/r;
        double r6 = pow(r*r, 3);
        double r12 = r6*r6;
        return 12 * scale * (r6 - r12) * r;
    }
};

class RepulsivePotential : public PairPotential
{
public:
    double power=1.0;

    std::map<int, double> cutoffs;

    double F(double r, int type1=0, int type2=0) override
    {
        if (cutoffs.find(type1) != cutoffs.end() && cutoffs.find(type2) != cutoffs.end())
        {
            double r0 = cutoffs[type1] + cutoffs[type2];
            if (r < r0)
                return 0.5 * power * (r-r0)*(r-r0);
            else
                return 0.0;
        }
        else
            ERROR("Repulsive potential is not defined for type " + 
                  to_string(type1) + " or " + to_string(type2));
    }

    double dF_dr(double r, int type1=0, int type2=0) override
    {
        if (cutoffs.find(type1) != cutoffs.end() && cutoffs.find(type2) != cutoffs.end())
        {
            double r0 = cutoffs[type1] + cutoffs[type2];
            if (r < r0)
                return power * (r - r0);
            else
                return 0.0;
        }
        else
            ERROR("Repulsive potential is not defined for type " +
                to_string(type1) + " or " + to_string(type2));
    }
};


#endif //#ifndef MLIP_PAIR_POTENTIALS_H

