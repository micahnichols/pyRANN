/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_BASIC_TRAINER_H
#define MLIP_BASIC_TRAINER_H


#include "basic_mlip.h"
#include <list>


// basic class for fitting MLIP
class AnyTrainer : public InitBySettings //: public LogWriting
{
protected:
    static const char* tagname;                             // tag name of object

    unsigned int learn_cntr = 0;
    unsigned int train_cntr = 0;

public:
    AnyLocalMLIP* p_mlip = nullptr;                         // pointer to MLIP that is fitted

    double wgt_eqtn_energy = 1.0;                           // Optimization weights balancing equations for energy, forces on each atom of the system and stress components in minimizaion problem 
    double wgt_eqtn_forces = 0.01;                         // Optimization weights balancing equations for energy, forces on each atom of the system and stress components in minimizaion problem
    double wgt_eqtn_stress = 0.001;                           // Optimization weights balancing equations for energy, forces on each atom of the system and stress components in minimizaion problem
    double wgt_eqtn_constr = 1.0e-6;                        // Optimization weight for constrains
    double wgt_rel_forces = 0.0;                            // if <= 0 all forces in configuration are fitte with the same weight. Otherwise each force in cfg is fitted with its own weight equal wgt_eqtn_forces * wgt_rel_forces / (cfg.force(i).NormSq() + wgt_rel_forces)
    double selection_factor = 1.0;                          // While fitting on selected configurations this factor is at equations that have NOT been selected (selected equations have a factor == 1). Value of 0 switches off fitting to non-selected equations.
    int wgt_scale_power_energy = 1;
    int wgt_scale_power_forces = 0;
    int wgt_scale_power_stress = 1;
    std::string mlip_fitted_fnm = "";
    std::string fit_log = "";

    void InitSettings()                        // Sets correspondence between variables and setting names in settings file
    {
        MakeSetting(mlip_fitted_fnm,        "save_to");             
        MakeSetting(wgt_eqtn_energy,        "energy_weight");    
        MakeSetting(wgt_eqtn_forces,        "force_weight");     
        MakeSetting(wgt_eqtn_stress,        "stress_weight");    
        MakeSetting(wgt_eqtn_constr,        "penalty_weight");    
        MakeSetting(wgt_rel_forces,         "scale_by_force");  
        MakeSetting(selection_factor,       "select_factor");
        MakeSetting(wgt_scale_power_energy, "weight_scaling");
        MakeSetting(wgt_scale_power_forces, "weight_scaling_forces");
        MakeSetting(fit_log,                "log");                // only the first process writes log 
    };

protected:
    AnyTrainer() {};
public:
    AnyTrainer(AnyLocalMLIP* _p_mlip, const Settings& settings);
    ~AnyTrainer() {};

    virtual void Train() { ERROR("Inapropriate call of \"Train()\""); };    // Required if training set is already submitted (via inherited functions)
    virtual void Train(std::vector<Configuration>& train_set) = 0;          // function starting training procedure
    void Train(std::list<Configuration>& train_set)
    {
        std::vector<Configuration> train_set_cfg = std::vector<Configuration>(train_set.begin(), train_set.end());
        Train(train_set_cfg);
    }

    double wgt_energy(const Configuration& cfg);
    double wgt_forces(const Configuration& cfg, int ind, int a);
    double wgt_stress(const Configuration& cfg, int a, int b);
};


inline double AnyTrainer::wgt_energy(const Configuration& cfg)
{
    double wgt_energy = wgt_eqtn_energy / pow(cfg.nbhs.size(), wgt_scale_power_energy);

    if (cfg.fitting_items.count(0) == 0)
        wgt_energy *= selection_factor;

    return wgt_energy;
}

inline double AnyTrainer::wgt_forces(const Configuration& cfg, int ind, int a)
{
    double wgt_forces = wgt_eqtn_forces / pow(cfg.nbhs.size(), wgt_scale_power_forces);

    if (wgt_rel_forces > 0.0)
        wgt_forces *= wgt_rel_forces / (cfg.force(ind).NormSq() + wgt_rel_forces);

    if (cfg.fitting_items.count(3*ind+a + 1) == 0)
        wgt_forces *= selection_factor;

    return wgt_forces;
}

inline double AnyTrainer::wgt_stress(const Configuration& cfg, int a, int b)
{
    double wgt_stress = wgt_eqtn_stress / pow(cfg.nbhs.size(), wgt_scale_power_stress);

    if (cfg.fitting_items.count((3*a + b) + (1 + 3*(int)cfg.nbhs.size())) == 0)
        wgt_stress *= selection_factor;

    return wgt_stress;
}


#endif // MLIP_BASIC_TRAINER_H
