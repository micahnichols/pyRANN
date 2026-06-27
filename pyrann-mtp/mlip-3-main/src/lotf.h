/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_LOTF_H
#define MLIP_LOTF_H


#include "basic_trainer.h"
#include "cfg_sampling.h"
#include "linear_regression.h"


// Learning on the fly
class LOTF : public AnyPotential, public InitBySettings
{
private:
    std::vector<Configuration> new_cfgs;
    int global_sampled_count=0;

protected:
    static const char* tagname;                           // tag name of object

    void WriteCfgToFile(Configuration& cfg, std::string filename);
    void Select();
    void CalcAbInitio();      // finds entered to the training set configuration among p_selector->last_updated, does abinition calculation, and copies EFS to argument cfg if it enetered to training set on this processor
    void Retrain();
    void UpdateSelectionState();

public:
    AnyPotential* p_abinitio = nullptr;                 // pointer to Ab-initio potential
    CfgSampling* p_selector = nullptr;                 // pointer to selecting object
    AnyTrainer* p_trainer = nullptr;                    // pointer to training object. Learning potential is p_trainer->p_mlip
    AnyLocalMLIP* p_mlip = nullptr;

    std::vector<Configuration> train_set;

    int pot_calcs_count = 0;                            // counter for base_pot->CalcEFS() calls (Ab-initio calcs) 
    int MTP_calcs_count = 0;                            // counter for MTP->CalcEFS() calls 
    int train_count = 0;                                // counter for regression SLAE solving attmpt
    int interp_cfg_count = 0;
    int extrap_cfg_count = 0;
    int sampled_count = 0;
    int step_count = 0;

    std::string break_cfgs_fnm = "";
    std::string lotf_log = "";
    std::string tmp_cfg_fnm = "";
    int max_sampled_count = HUGE_INT;
    bool replace_ts = false;
    bool EFS_via_abinitio = false;                            // if false EFS are taken from ab-initio potential on configurations chosen for learning (a bit faster)
    bool abinitio_changes_cfg = false;
    bool save_history = false;

    void InitSettings()                                     // Sets correspondence between variables and setting names in settings file
    {
        MakeSetting(lotf_log,               "log");
        MakeSetting(max_sampled_count,      "sampling_limit");
        MakeSetting(replace_ts,             "fit_to_active_set");
        MakeSetting(EFS_via_abinitio,       "efs_only_by_mlip");
        MakeSetting(tmp_cfg_fnm,            "save_ts_to");
        MakeSetting(save_history,           "save_updates");
        //        MakeSetting(abinitio_changes_cfg,   "abinitio_changes_cfg");
    };
        
    LOTF(AnyPotential* _p_abinitio,                  
         CfgSampling* _p_selector,              // must be p_selector->p_mlip == p_trainer->p_mlip
         AnyTrainer* _p_trainer,
         Settings _settings);

    void UpdateMLIP();
    void CalcEFS(Configuration & cfg) override;         // decides if learning is required, querries ab-initio EFS and learns configuration if necessary, and calculates EFS
};

#endif // MLIP_LOTF_H
