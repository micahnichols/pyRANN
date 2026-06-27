/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_SAMPLING_H
#define MLIP_SAMPLING_H

#include "basic_mlip.h"
#include "basic_trainer.h"
#include "cfg_selection.h"


class CfgSampling : public CfgSelection //: public LogWriting
{
protected:
    static const char* tagname;                             // tag name of object



public:
    // threshold parameters
    double threshold_sample = 2.0;                          // >=1.0. is used in MaxVol routines. Configuration is selected if Grade() > 1+threshold_slct
    double threshold_break = 11.0;                           // ativates interruption mode of selection if > 1. Interrupts program execution (throws an exception) if MV_grade for a configuration exceeds this value

    bool add_grade_to_cfg = true;

    // Counters
    int cfg_sampled_count = 0;                                // counter for configurations accumulated for selection (for multiple selection only)
    int cfg_eval_count = 0;                                 // evaluated configurations counter

    std::string sampled_cfgs_fnm = "";                     // if interruption mode is activated (threshold_break > 1), 'Grade(), writes configurations which grade is above 'threshold_select' and bellow 'threshold_break'

    void InitSettings()                                     // Sets correspondence between variables and setting names in settings file
    {              
        MakeSetting(threshold_sample,       "threshold_save");     
        MakeSetting(threshold_break,        "threshold_break");
        MakeSetting(sampled_cfgs_fnm,       "save_extrapolative_to");
        MakeSetting(add_grade_to_cfg,       "add_grade_feature");
        MakeSetting(slct_log,               "log");
    };

public:
    CfgSampling(AnyLocalMLIP* _p_mlip, 
                std::string load_state_from, 
                const Settings& _settings);
    CfgSampling(const CfgSelection& selector, const Settings& _settings);
    ~CfgSampling();

    double Grade(Neighborhood& nbh);                        // returns MV grade for nbh
    double Grade(Configuration& cfg);                       // returns MV grade for cfg. Also adds "MV_grade" feature to cfg. In contrast to GetGrade() is safe and clears maxvol.B after. Does not change cfgs_queue. If some configurations are accumulated for multiple selection calls Select() before procissing cfg to finalize their selection
 
    void Evaluate(Configuration& cfg);                       // Process cfg depending on setiings
};
#endif // MLIP_SAMPLING_H
