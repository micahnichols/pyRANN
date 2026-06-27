/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_LOTF_LINEAR_H
#define MLIP_LOTF_LINEAR_H


#include "basic_trainer.h"
#include "cfg_selection.h"
#include "linear_regression.h"


// Learning on the fly
class LOTF_linear : public AnyPotential
{
private:
    LinearRegression* p_linreg = nullptr;

protected:
    static const char* tagname;                           // tag name of object

    void Retrain();
    void DoSingleAbinitioCalc(Configuration& cfg);      // finds entered to the training set configuration among p_selector->last_updated, does abinition calculation, and copies EFS to argument cfg if it enetered to training set on this processor

    void CopyEFS(Configuration& from, Configuration& to);   

public:
    AnyPotential* p_abinitio = nullptr;                 // pointer to Ab-initio potential
    CfgSelection* p_selector = nullptr;                 // pointer to selecting object
    AnyTrainer* p_trainer = nullptr;                    // pointer to training object. Learning potential is p_trainer->p_mlip

    bool EFS_via_MTP = true;                            // if false EFS are taken from ab-initio potential on configurations chosen for learning (a bit faster)
    bool collect_abinitio_cfgs = false;                 // whether to collect abinitio_cfgs
    bool ignore_entry_count = false;
#ifdef MLIP_MPI
    bool sequentional_abinitio_treatment = true;        // sequantional treatment of configurations by abinitio potential if true, otherwise abinito calculations can run simultaneously
#endif
    
    std::vector<Configuration> abinitio_cfgs;           // configuration container with ab-initio EFS 

    int pot_calcs_count = 0;                            // counter for base_pot->CalcEFS() calls (Ab-initio calcs) 
    int MTP_calcs_count = 0;                            // counter for MTP->CalcEFS() calls 
    int learn_count = 0;                                // counter for learner->AddForTrain() and learner->remove_from_SLAE() calls
    int train_count = 0;                                // counter for regression SLAE solving attmpt

    LOTF_linear(   AnyPotential* _p_abinitio,                  // default constructor
            CfgSelection* _p_selector,
            AnyTrainer* _p_trainer,
            bool _EFS_via_MTP = false) : 
        p_abinitio(_p_abinitio),
        p_selector(_p_selector),
        p_trainer(_p_trainer),
        EFS_via_MTP(_EFS_via_MTP) {};
    LOTF_linear(   AnyPotential* _p_abinitio,                  // This constructor creats optimized for the linear regression object (fast retraining instead of full training on train set in default constructor)
            CfgSelection* _p_selector,
            LinearRegression* _p_trainer,
            bool _EFS_via_MTP = false) :
        p_abinitio(_p_abinitio),
        p_selector(_p_selector),
        p_trainer(_p_trainer),
        p_linreg(_p_trainer),
        EFS_via_MTP(_EFS_via_MTP) { p_selector->track_changes = true; };

    void CalcEFS(Configuration & cfg) override;         // decides if learning is required, querries ab-initio EFS and learns configuration if necessary, and calculates EFS
};

#endif // MLIP_LOTF_LINEAR_H
