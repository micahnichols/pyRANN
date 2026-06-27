/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_SELECTION_H
#define MLIP_SELECTION_H

#include "basic_mlip.h"
#include "basic_trainer.h"
#include "maxvol.h"


// A class for selection of configurations (active learning) for learning based on Maxvol algorithm
// USAGE: there are two options:
//  1.  Selecting configurations one-by-one (guarantees that no extrapolation occurs on the treated configurations, 
//      not guarantees construction the training set of maximum volume)
//      - just call .Select(cfg) method
//  2.  Selecting from the set of configurations or multiple selection (guarantees no extrapolation on the treated configurations, 
//      guarantees construction the training set of maximum volume)
//      - 1) commit configurations one-by-one for selection via .AddForSelection(cfg) method
//        2) call .Select() method to construct the selected set
// Specific of parallel treatment:
//      - Both implementation of .Select(...) method involve collective operations and should be called on ALL processors.
//      - If the number of configurations is not exact division of the number of processors 
//          a void (i.e. of zero size) configuration can be used
//      - When selecting configurations one-by-one different configurations on different processors 
//          can be put as argument for .Select(cfg) method. 
//          In this case the result is equivalent to multiple selection of mpi_size configuration on a sigle process.
//      - Selected set and the set of commited configurations (in multiple selection) is distributed among processors. 
//      - Configurations are not transfered between processors. 
//          Therefore the number of configuration (as selected as committed) can by different on each processor.
// Structure and principles of functioning:
//      - Each configuration produce one or more matrix rows that are selected in Maxvol routine.
//      - The first thing is done while selection or calculating the grade is calculation of matrix rows for a given configuration.
//          It is done in .Grade(), .AddForSelection(cfg) and .Select(cfg) routines. 
//      - in case of multiple selection and in single selection when 'threshold_select' exceeded 
//          the selecting configurations are copied and stored in cfgs_queue.
//      - The 'eqn2cfg_links' stores for each row
//          * a pointer to the producing configuration,
//          * the index of a particular equation (energy equation has index 0, 
//                                                  forces equations have 1 - 3*cfg.size() indices, 
//                                                  stress equations have 3*cfg.size()+1 - 3*cfg.size()+10 indices)
//                                                  site energy equations have 3*cfg.size()+1 - 3*cfg.size()+10 indices),
//          * mpi_rank of processor, on which the corresponding configuration is hosted.
//      - Different rows of 'eqn2cfg_links' can point to the same configuration but must have different equation indices.
//          'eqn2cfg_links' stores the data for selected configurations as well as for candidates. 
//          Selected rows occupy the first p_mlip->CoeffCount() positions.
//      - In the very beginning the following data is set eqn2cfg_links[i].p_cfg == nullptr,
//                                                          eqn2cfg_links[i].ind == -1,
//                                                          eqn2cfg_links[i].mpi_at_rank == i % mpi_rank.
//      - eqn2cfg_links.ind holds the proper value only at the processor where the corresponding configuration is hosted,
//          otherwise eqn2cfg_links.ind == -1 and eqn2cfg_links.p_cfg == nullptr.
//      - The configuration selection process is based on the swap rows history (storing the numbers of swapped rows)
//          after Maxvol algorithm maximized the matrix volume. 
//          According to the swap history. The configurations are moving between 'cfgs_queue' and 'selected_cfgs'.
//          At the same time the records in 'eqn2cfg_links' are also swapping, whereas pointer to configurations, 
//              equation indices and the rank of hosting processor kepps unchanged 
//              (that is why use of std::list for storage of configurations is important). 
//      - After selection ends the rest of eqn2cfg_links is deleted, cfgs_queue is cleared.
//      - 'track_changes' option is required for fast retraining while LOTF. 
//              It enables to obtain the changes in 'selected_set' after each selection event.
//      - When loading the state of the selecing object from a file thresholds are not loaded
//      - After loading the state of the selecing object from a file 
//          the selected configurations are distributed nearly eventually among processors, 
//          that may not coinside with their occupation when saving the state.
class CfgSelection : public InitBySettings //: public LogWriting
{
protected:
    static const char* tagname;                             // tag name of object

    struct EqnToCfgLink
    {
        Configuration* p_cfg;
        int eqn_ind;
#ifdef MLIP_MPI
        int mpi_host;
        EqnToCfgLink(Configuration* _p_cfg, int _eqn_ind, int _mpi_host) :
            p_cfg(_p_cfg), eqn_ind(_eqn_ind), mpi_host(_mpi_host) {};
        EqnToCfgLink() : p_cfg(nullptr), eqn_ind(-1), mpi_host(-1) {};
#else
        EqnToCfgLink(Configuration* _p_cfg, int _eqn_ind) : p_cfg(_p_cfg), eqn_ind(_eqn_ind) {};
        EqnToCfgLink() : p_cfg(nullptr), eqn_ind(-1) {};
#endif
    };

    std::vector<EqnToCfgLink> eqn2cfg_links;

    std::vector<int> eqn_counts;                            // stores equation count for each cfg in multiple selection

    int PrepareMatrix(Configuration& cfg);                  // creates matrix B for MaxVol selection
    void PrepareLinks(Configuration & cfg);
    int UpateLinks(int swap_limit=HUGE_INT);                // moves configurations between cfgs_queue and selected_cfgs according to maxvol.swap_tracker. Clears maxvol.B. Does not clear cfgs_queue 
    void UpateSelected();                                   // moves configurations between cfgs_queue and selected_cfgs according to maxvol.swap_tracker. Clears maxvol.B. Does not clear cfgs_queue 
    void UpdateChanges(const EqnToCfgLink& in_A, const EqnToCfgLink& in_B);

public:
    AnyLocalMLIP* p_mlip = nullptr;                         // pointer to actively learning MLIP

    std::list<Configuration> selected_cfgs;                 // storage of selected configurations
    std::list<Configuration> cfgs_queue;                    // storage for the configurations accumulated for selection. Only std::list not a std::vector! Algorithm uses pointers to cfgs in the container which must not change while deleting/insertion of elements in the container
    std::list<Configuration> outsorted_cfgs;                // storage of outsorted configurations

    MaxVol maxvol;                                          // MaxVol algorithm object

    //Settings
    // Selection weights
    double MV_nbh_cmpnts_weight = 0.0;                      // weight for site energies equaions in selection
    double MV_ene_cmpnts_weight = 1.0;                      // weight for energy equaions in selection
    double MV_frc_cmpnts_weight = 0.0;                      // weight for forces equaions in selection
    double MV_str_cmpnts_weight = 0.0;                      // weight for stress equaions in selection

    // Scaling parameter
    int wgt_scale_power = 2;                                // division on different powers of cfg.size()

    // threshold parameters
    double threshold_init = 0.000001;                      // Maxvol matrix is initialized by IdentityMatrix * threshold_init
    double threshold_select = 1.001;                          // >=1.0. is used in MaxVol routines. Configuration is selected if Grade() > 1+threshold_slct

    // Mode selection
    int batch_size = 9999;                                  // selection by batches of this count of configuration (multiplied by a number of cores)
    int swap_limit = 0;                                     // swap limit 

    //logging
    std::string slct_log = "";

    // Counters
    int cfg_added_count = 0;                                // counter of configurations accumulated for selection at local processor (used for multiple selection). Includes empty configurations, therefore may not coincide with cfgs_queue.size() 
    int cfg_selct_count = 0;                                // in the case of single selection counter for ever-selected configurations. In the case of multiple selection counter for selection calls with at least one swap (selection grade > 1 + threshold)
    int cfg_eval_count = 0;                                 // evaluated configurations counter

    bool track_changes = false;
    std::map<Configuration*, std::set<int>> last_updated;   // an array of pointers to configurations changed during the last selection. update_data[i].second contain difference in selected equations. positive values - entered, negative - leaved

    std::string selected_cfgs_fnm = "";                     // if interruption mode is activated (threshold_break > 1), 'Grade(), writes configurations which grade is above 'threshold_select' and bellow 'threshold_break'
    std::string state_save_fnm = "";

    void InitSettings()                                     // Sets correspondence between variables and setting names in settings file
    {              
        MakeSetting(threshold_select,       "threshold");     
        MakeSetting(threshold_init,         "init_value");
        MakeSetting(MV_nbh_cmpnts_weight,   "site_en_weight");
        MakeSetting(MV_ene_cmpnts_weight,   "energy_weight"); 
        MakeSetting(MV_frc_cmpnts_weight,   "force_weight");  
        MakeSetting(MV_str_cmpnts_weight,   "stress_weight"); 
        MakeSetting(wgt_scale_power,        "weight_scaling");
        MakeSetting(selected_cfgs_fnm,      "save_selected_to");
        MakeSetting(batch_size,             "batch_size");
        MakeSetting(swap_limit,             "swap_limit");
        MakeSetting(state_save_fnm,         "save_to");    
        MakeSetting(slct_log,               "log");
    };

    CfgSelection();
    CfgSelection(AnyLocalMLIP* _p_mlip, const Settings& _settings);
    ~CfgSelection();

    void Reset();

    void AddForSelection(Configuration& cfg);               // Accumulates cfg for multiple selection
    int Select(int swap_limit = HUGE_INT);                  // Performs multiple selection for accumulated configuration set. No "MV_grade" feature is added to selecting configurations.
    int Select(Configuration& cfg);                         // Optimized version (for single configuration) of the previous function. Single selection of accumulated configuration. Also adds "MV_grade" feature to cfg. If some configurations are accumulated for multiple selection calls Select() before procissing cfg to finalize their selection.
 
    void Process(Configuration& cfg);                       // Process cfg depending on setiings

    void LoadSelected(const std::string& filename);
    void SaveSelected(std::string filename);                // writes MV_selected to a file
    void Save(const std::string& filename);                 // Saves the state of CfgSelection to specified file
    void Load(const std::string& filename);                 // Loads the state of CfgSelection to specified file (MLIP will be changed to loaded)
};
#endif // MLIP_SELECTION_H
