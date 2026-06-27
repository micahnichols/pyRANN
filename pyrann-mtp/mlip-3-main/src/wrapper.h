/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_WRAPPER_H
#define MLIP_WRAPPER_H


#include "lotf.h"
#include "cfg_sampling.h"
#include "error_monitor.h"
#include "external_potential.h"


// Class that allows working with MLIP in multiple regimes (e.g. fitting, EFS calculating, active learning, fitting error calculation, etc.)
class Wrapper : protected InitBySettings //, public LogWriting
{
  protected:
  static const char* tagname;                           // tag name of object
  private:
    // settings initialized by default values
    std::string mlip_type = "";             // MLIP type
    std::string mlip_fnm = "";              // file name, MLIP will be loaded from. MLIP is not required if empty
    bool enable_mlip = false;
    bool enable_EFScalc = false;            // if true MLIP will calculate EFS
    bool enable_sampling = false;           // turns sampling on of true
    bool enable_select = false;             // turns selection on of true
    bool enable_learn = false;              // if true MLIP fitting will be performed
    bool enable_lotf = false;
    bool enable_cfg_writing = false;
    bool enable_err_check = false;

    bool efs_ignored = false;               // if efs ignored by driver it is possible to speed up some scenarios
    std::string cfgs_fnm = "";              // filename for recording configurations (calculated by CalcEFS()). No configurations are recorded if empty
    int skip_saving_N_cfgs = 0;             // every skip_saving_N_cfgs+1 configuration will be saved (see previous option)
    std::string log_output = "";            // filename or "stdout" or "stderr" for this object log output. No logging if empty
    bool enable_abinitio = false;           // if true MLIP will use external potential for calculations 

    Configuration cfg_valid;                // temporal for errmon

    int call_count = 0;                     // number of processed cfg
    int learn_count = 0;                    // number of learned cfg
    int MTP_count = 0;                      // number of MTP calcs
    int abinitio_count = 0;                 // number of abinitio calcs

    void InitSettings()                        // Sets correspondence between variables and setting names in settings file
    {
        MakeSetting(enable_mlip,            "mlip");                 
        MakeSetting(mlip_fnm,               "mlip:load_from");       
        MakeSetting(enable_EFScalc,         "calculate_efs");        
        MakeSetting(enable_learn,           "fit");                  
        MakeSetting(enable_sampling,        "extrapolation_control");
        MakeSetting(enable_select,          "select");
        MakeSetting(enable_lotf,            "lotf");
        MakeSetting(enable_cfg_writing,     "write_cfgs");
        MakeSetting(cfgs_fnm,               "write_cfgs:to");
        MakeSetting(skip_saving_N_cfgs,     "write_cfgs:skip");
        MakeSetting(enable_err_check,       "check_errors");
        MakeSetting(enable_abinitio,        "abinitio");                  
    };
    void SetUpMLIP(const Settings&);    // Initialize object state according to settings

  public:
    AnyPotential* p_abinitio = nullptr;        // pointer to abinitio potential
    ExternalPotential* p_external = nullptr;
    AnyLocalMLIP* p_mlip = nullptr;            // pointer to MLIP
    AnyTrainer* p_learner = nullptr;        // pointer to training object
    CfgSelection* p_selector = nullptr;    // pointer to selecting object
    CfgSampling* p_sampler = nullptr;    // pointer to selecting object
    LOTF* p_lotf = nullptr;                    // pointer to LOTF object
    std::vector<Configuration> training_set;// training set (for pure training regime)
    ErrorMonitor* p_errmon = nullptr;                    // error monitoring object

    Wrapper(const Settings&);
    virtual ~Wrapper();
    
    void Process(Configuration& cfg);        // function that does work configured according settings file (for example may perform fitting, EFS calculation, error caclulation, LOTF, etc.)
    void Release();;
};

#endif
