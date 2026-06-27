/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "cfg_selection.h"
#include <sstream>
#include <numeric>


using namespace std;


const char* CfgSelection::tagname = {"select"};


CfgSelection::CfgSelection() 
{
    p_mlip = nullptr;
    SetTagLogStream("select", slct_log);
}

CfgSelection::CfgSelection(AnyLocalMLIP * _p_mlip, const Settings & settings) :
    p_mlip(_p_mlip)
{
    InitSettings();
    ApplySettings(settings);
    if (0 == mpi.rank)
        PrintSettings();

    if (threshold_init <= 0.0 || threshold_select <= 1.0)
        ERROR("Invalid threshold specified");

    if (!slct_log.empty() &&
        slct_log != "stdout" &&
        slct_log != "stderr")
        slct_log += mpi.fnm_ending;
    
    // clean output files
    if (!slct_log.empty() && slct_log!="stdout" && slct_log!="stderr")
        ofstream ofs(slct_log);

    if (!selected_cfgs_fnm.empty())
    {
        { ofstream ofs(selected_cfgs_fnm); }
        { ofstream ofs(selected_cfgs_fnm + mpi.fnm_ending); }
        remove(selected_cfgs_fnm.c_str());
        remove((selected_cfgs_fnm + mpi.fnm_ending).c_str());
    }
    if (!state_save_fnm.empty())
    {
        ofstream ofs(state_save_fnm);
        remove(state_save_fnm.c_str());
    }

    Reset();

    SetTagLogStream("select", slct_log);
}

CfgSelection::~CfgSelection()
{
    if (batch_size != 1)
        Select((swap_limit>0) ? swap_limit : HUGE_INT);

    int global_selected_size=0;
    int local_selected_size=(int)selected_cfgs.size();
    MPI_Allreduce(&local_selected_size, &global_selected_size, 1, MPI_INT, MPI_SUM, mpi.comm);
    if (!selected_cfgs_fnm.empty() && global_selected_size>0) // 
        SaveSelected(selected_cfgs_fnm);

    if (!state_save_fnm.empty())
        Save(state_save_fnm);
}

void CfgSelection::Reset()
{
    // 1. clean selected 
    maxvol.Reset(p_mlip->CoeffCount(), threshold_init);

    cfg_added_count = 0;    
    cfg_eval_count = 0;
    selected_cfgs.clear();
    cfgs_queue.clear();
    eqn_counts.clear();
    maxvol.B.resize(0, p_mlip->CoeffCount());

#ifdef MLIP_MPI
    eqn2cfg_links.resize(p_mlip->CoeffCount());
    for (int i=0; i<eqn2cfg_links.size(); i++)
        eqn2cfg_links[i] = EqnToCfgLink(nullptr, -1, i % mpi.size);
#else
    eqn2cfg_links.resize(p_mlip->CoeffCount());
    for (int i=0; i<eqn2cfg_links.size(); i++)
        eqn2cfg_links[i] = EqnToCfgLink(nullptr, -1);
#endif
}

int CfgSelection::PrepareMatrix(Configuration& cfg)
{
    if (cfg.nbh_cutoff != p_mlip->CutOff())
        cfg.InitNbhs(p_mlip->CutOff());

    if (!cfg.is_mpi_splited && cfg.nbhs.size() == 0)
        return maxvol.B.size1;

    int n = p_mlip->CoeffCount();

    //  Number of rows to be selected/evaluated 
    int  max_rows_cnt = (MV_ene_cmpnts_weight != 0.0)*1 +
                        (MV_frc_cmpnts_weight != 0.0)*3*(int)cfg.nbhs.size() +
                        (MV_str_cmpnts_weight != 0.0)*9 +
                        (MV_nbh_cmpnts_weight != 0.0)*(int)cfg.nbhs.size();

    int row_count = maxvol.B.size1;
    maxvol.B.resize(maxvol.B.size1 + max_rows_cnt, n);
 
    if (MV_ene_cmpnts_weight != 0.0 &&      // Selection by energy equation
        MV_frc_cmpnts_weight == 0.0 && 
        MV_str_cmpnts_weight == 0.0)
    {
        std::vector<double> tmp(n);

        if (cfg.nbhs.size() != 0)
            p_mlip->CalcEnergyGrad(cfg, tmp);
        else
            FillWithZero(tmp);
        
        if (cfg.is_mpi_splited)
        {
            int global_cfg_size=0;
            int local_cfg_size=(int)cfg.nbhs.size();
            MPI_Allreduce(&local_cfg_size, &global_cfg_size, 1, MPI_INT, MPI_SUM, mpi.comm);
            double scale = pow(global_cfg_size, wgt_scale_power/2.0);
            for (double& el : tmp)
                el *= MV_ene_cmpnts_weight / scale;
            MPI_Allreduce(tmp.data(), &maxvol.B(row_count, 0), n, MPI_DOUBLE, MPI_SUM, mpi.comm);
        }
        else
        {
            double scale = pow(cfg.nbhs.size(), wgt_scale_power/2.0);
            for (double& el : tmp)
                el *= MV_ene_cmpnts_weight / scale;
            memcpy(&maxvol.B(row_count, 0), tmp.data(), n*sizeof(double));
        }

        row_count++;
    }
    else if (MV_ene_cmpnts_weight != 0.0 || // Selection by all equations
             MV_frc_cmpnts_weight != 0.0 || 
             MV_str_cmpnts_weight != 0.0)
    {   // TODO: optimize it for different weights combinations
        Array1D tmp4ene(n);
        Array3D tmp4frc((int)cfg.nbhs.size(), 3, n);
        Array3D tmp4str(3, 3, n);

        if (cfg.nbhs.size() != 0)
            p_mlip->CalcEFSGrads(cfg, tmp4ene, tmp4frc, tmp4str);
        else
        {
            FillWithZero(tmp4ene);
            tmp4str.set(0.0);
        }

        if (MV_ene_cmpnts_weight != 0.0)
        {
            double scale;
            if (cfg.is_mpi_splited)
            {
                int global_cfg_size=0;
                int local_cfg_size=(int)cfg.nbhs.size();
                MPI_Allreduce(&local_cfg_size, &global_cfg_size, 1, MPI_INT, MPI_SUM, mpi.comm);
                scale = pow(global_cfg_size, wgt_scale_power/2.0);
            }
            else
                scale = pow(cfg.nbhs.size(), wgt_scale_power/2.0);

            if (cfg.is_mpi_splited)
                MPI_Allreduce(tmp4ene.data(), &maxvol.B(row_count, 0), n, MPI_DOUBLE, MPI_SUM, mpi.comm);
            else
                memcpy(&maxvol.B(row_count, 0), tmp4ene.data(), n*sizeof(double));

            row_count++;
        }

        if (MV_frc_cmpnts_weight != 0.0)
        {
            for (int j=0; j<cfg.nbhs.size(); j++)
                for (int a=0; a<3; a++)
                    for (int i=0; i<n; i++)
                        maxvol.B(row_count + (3*j+a), i) = 
                                MV_frc_cmpnts_weight * tmp4frc(cfg.nbhs[j].my_ind, a, i);
            row_count += 3*(int)cfg.nbhs.size();
        }

        if (MV_str_cmpnts_weight != 0.0)
        {
            double scale;
            if (cfg.is_mpi_splited)
            {
                int global_cfg_size=0;
                int local_cfg_size=(int)cfg.nbhs.size();
                MPI_Allreduce(&local_cfg_size, &global_cfg_size, 1, MPI_INT, MPI_SUM, mpi.comm);
                scale = pow(global_cfg_size, wgt_scale_power/2.0);
            }
            else
                scale = pow(cfg.nbhs.size(), wgt_scale_power/2.0);

            for (int a=0; a<3; a++)
                for (int b=0; b<3; b++)
                    for (int i=0; i<n; i++)
                        tmp4str(a, b, i) *= MV_str_cmpnts_weight / scale;
                   
            if (cfg.is_mpi_splited)
                MPI_Allreduce(tmp4str.data(), &maxvol.B(row_count, 0), 9*n, MPI_DOUBLE, MPI_SUM, mpi.comm);
            else
                memcpy(&maxvol.B(row_count, 0), &tmp4str(0,0,0), 9*n*sizeof(double));

            row_count += 9;
        }
    }

    if (MV_nbh_cmpnts_weight != 0.0)        // Selection by neighborhoods
    {
        vector<double> tmp(n);
        for (Neighborhood nbh : cfg.nbhs)
        {
            p_mlip->CalcSiteEnergyGrad(nbh, tmp);
            memcpy(&maxvol.B(row_count++, 0), &tmp[0], n*sizeof(double));
        }
    }

    return row_count;
}
 
void CfgSelection::PrepareLinks(Configuration& cfg)
{
    if (cfg.is_mpi_splited)
        ERROR("Selection of MPI-distributed configurations is not supported! (But MPI-parallel grade calculation is fully supported)");
    
    // adding equation-to-cfg links corresponding to new cfg
    if (cfg.nbhs.size() == 0)
        return;

#ifdef MLIP_MPI
    if (MV_ene_cmpnts_weight != 0.0)
        eqn2cfg_links.emplace_back(&cfg, 0, mpi.rank);
    if (MV_frc_cmpnts_weight != 0.0)
        for (int i=0; i<3*(int)cfg.nbhs.size(); i++)
            eqn2cfg_links.emplace_back(&cfg, 1+i, mpi.rank);
    if (MV_str_cmpnts_weight != 0.0)
        for (int i=0; i<9; i++)
            eqn2cfg_links.emplace_back(&cfg, 1 + 3*(int)cfg.nbhs.size() + i, mpi.rank);
    if (MV_nbh_cmpnts_weight != 0.0)
        for (int i=0; i<(int)cfg.nbhs.size(); i++)
            eqn2cfg_links.emplace_back(&cfg, 1 + 3*(int)cfg.nbhs.size() + 9 + i, mpi.rank);
#else
    if (MV_ene_cmpnts_weight != 0.0)
        eqn2cfg_links.emplace_back(&cfg, 0);
    if (MV_frc_cmpnts_weight != 0.0)
        for (int i=0; i<3*(int)cfg.nbhs.size(); i++)
            eqn2cfg_links.emplace_back(&cfg, 1 + i);
    if (MV_str_cmpnts_weight != 0.0)
        for (int i=0; i<9; i++)
            eqn2cfg_links.emplace_back(&cfg, 1 + 3*(int)cfg.nbhs.size() + i);
    if (MV_nbh_cmpnts_weight != 0.0)
        for (int i=0; i<(int)cfg.nbhs.size(); i++)
            eqn2cfg_links.emplace_back(&cfg, 1 + 3*(int)cfg.nbhs.size() + 9 + i);

#endif // MLIP_MPI
}

void CfgSelection::AddForSelection(Configuration& cfg)
{
    if (cfg.nbh_cutoff != p_mlip->CutOff())
        cfg.InitNbhs(p_mlip->CutOff());

    cfg_added_count++;

    if (cfg.nbhs.size() == 0)
        return;

    // adding rows to selection matrix corresponding to a new cfg
    int prev_rows_count = maxvol.B.size1;
    int rows_count = PrepareMatrix(cfg);

    // updating 'eqn_counts'. Required to setting the grades
    eqn_counts.push_back(rows_count - prev_rows_count);
    // making a copy of 'cfg' in 'cfgs_queue1' list
    cfgs_queue.emplace_back(cfg);
    // adding equation-to-cfg links corresponding to new cfg
    PrepareLinks(cfgs_queue.back());

    // logging
    std::stringstream logstrm1;
    logstrm1 << "Selection: " << cfg_added_count
            << " added for selection" << endl;
    MLP_LOG(tagname, logstrm1.str());
}

void CfgSelection::UpdateChanges(const EqnToCfgLink & in_A, const EqnToCfgLink & in_B)
{
    // composing cfg data for entering cfg
    if (in_B.p_cfg != nullptr)  // this can be only at hosting processor
    {
        set<int> dummy;
        auto inserted = last_updated.emplace(in_B.p_cfg, dummy);
        set<int>& changed_inds = inserted.first->second;
        if (inserted.second)                                        // if new element inserted in 'changed_cfgs'
            changed_inds.insert(in_B.eqn_ind+1);                    // write changes data for a copy of changed configuration. +1 is required for proper handling of the equation with zero index
        else                                                        // if such element already in
            if (changed_inds.count(-(in_B.eqn_ind+1)) == 0)         // if no records related to equationa with 'entering_ind'
                changed_inds.insert(in_B.eqn_ind+1);                // write changes data for a copy of changed configuration. +1 is required for proper handling of the equation with zero index
            else                                                    // if this equation leaved MV_set but is returning now
                if (changed_inds.count(-(in_B.eqn_ind+1)) == 1)
                {
                    changed_inds.erase(-(in_B.eqn_ind+1));
                    if (changed_inds.empty())
                        last_updated.erase(in_B.p_cfg);
                }
                else                                                // imposiible case
                    ERROR("an update of learning data is corrupted");
    }

    // composing cfg data for leaving cfg
    if (in_A.p_cfg != nullptr)  // this can be only at hosting processor
    {
        set<int> dummy;
        auto inserted = last_updated.emplace(in_A.p_cfg, dummy);
        set<int>& changed_inds = inserted.first->second;
        if (inserted.second)    // if new element inserted in 'changed_cfgs'
            changed_inds.insert(-(in_A.eqn_ind+1)); // write changes data for a copy of changed configuration
        else
            if (changed_inds.count(in_A.eqn_ind+1) == 0)
                changed_inds.insert(-(in_A.eqn_ind+1)); // write changes data for a copy of changed configuration
            else
                if (changed_inds.count(in_A.eqn_ind+1) == 1)
                {
                    changed_inds.erase(in_A.eqn_ind+1);
                    if (changed_inds.empty())
                        last_updated.erase(in_A.p_cfg);
                }
                else
                    ERROR("an update of learning data is corrupted");
    }
}

int CfgSelection::UpateLinks(int swap_limit)
{
    int n = p_mlip->CoeffCount();
    int swap_count = maxvol.MaximizeVol(threshold_select, swap_limit);

    last_updated.clear();
    outsorted_cfgs.clear();

#ifdef MLIP_MPI
    vector<int> links_sizes(mpi.size);
    int init_loc_size = (int)eqn2cfg_links.size();
    MPI_Allgather(&init_loc_size, 1, MPI_INT, links_sizes.data(), 1, MPI_INT, mpi.comm);
#endif

    // updating 'fitting_items' data for each cfg in 'cfgs_queue1' and 'selected_cfgs1'. cfgs are not moving between 'cfgs_queue1' and 'selected_cfgs1' in this codeblock
    int swap_tracker_size = (int)maxvol.swap_tracker.size();
    for (int i=0; i<swap_tracker_size; i++)
    {
        auto swapka = maxvol.swap_tracker[i];

#ifdef MLIP_MPI
        EqnToCfgLink* p_in_A = &eqn2cfg_links[swapka.ind_A];
        EqnToCfgLink tmp_link(nullptr, -1, swapka.mpi_root_B);  // it keeps proper value of 'mpi_at_rank'
        EqnToCfgLink* p_in_B = &tmp_link;
        if (swapka.mpi_root_B == mpi.rank)
            p_in_B = &eqn2cfg_links[n + swapka.ind_B];
        //else p_in_B = &tmp_link
#else
        EqnToCfgLink* p_in_A = &eqn2cfg_links[swapka.ind_A];
        EqnToCfgLink* p_in_B = &eqn2cfg_links[n + swapka.ind_B];
#endif // MLIP_MPI

#ifdef MLIP_DEBUG   // check if leaving and entering cfgs have/havn't the proper fitting_items
        if (p_in_B->p_cfg != nullptr &&     // entering cfg is not dummy (initial) and on this processor
            p_in_B->p_cfg->fitting_items.count(p_in_B->eqn_ind) > 0)    // entering cfg already has this equation (cannot be so!)
            ERROR("Invalid cfgs_queue state");
        if (p_in_A->p_cfg != nullptr && 
            p_in_A->p_cfg->fitting_items.count(p_in_A->eqn_ind) == 0)
            ERROR("Invalid selected_cfgs state");
#endif // MLIP_DEBUG

        if (p_in_A->p_cfg != nullptr)   // leaving cfg is hosted ont this processor and not the case of initial dummy values
            p_in_A->p_cfg->fitting_items.erase(p_in_A->eqn_ind);
        if (p_in_B->p_cfg != nullptr)   // entering cfg is hosted ont this processor and not the case of initial dummy values
            p_in_B->p_cfg->fitting_items.insert(p_in_B->eqn_ind);

        if (track_changes)
            UpdateChanges(*p_in_A, *p_in_B);

#ifdef MLIP_MPI
        // swap at appropriate process. Not before previous step
        if (p_in_B->mpi_host == p_in_A->mpi_host)           // both entering and leaving rows are hosted at the same processor
            swap(*p_in_B, *p_in_A);                         // keep proper data for both leaving and entering cfgs in eqn2cfg_links
        else                                                // entering and leaving rows are hosted at different processors
        {
            if (p_in_A->mpi_host == mpi.rank)               // at processor where leaving row is hosted
            {
                eqn2cfg_links.push_back(tmp_link);          // allocate a place for storage of leaving row (leaving row can return to selected set, so should not be lost)
                p_in_A = &eqn2cfg_links[swapka.ind_A];      // previous eqn2cfg_links.push_back() may change an adress of container
                p_in_B = &eqn2cfg_links.back();             // 
            }
            links_sizes[p_in_A->mpi_host]++;                //

            // Selected and candidates rows are distributed among processoes.
            // A selected row from one processor may be replaced by a row from another rocessor, and then, enter again to selected set.
            // To prevent configuration transfer between processors, correction of the swapping history is required.
            // This correction is the same on all processors and is related to hosting processor (proper ro).
            // In particular, if some selected row is replaced by another one from a different processor, it is just put at the end of matrix.
            for (int j=i+1; j<swap_tracker_size; j++)       // 
                if (maxvol.swap_tracker[j].ind_B == swapka.ind_B &&
                    maxvol.swap_tracker[j].mpi_root_B == swapka.mpi_root_B)
                {
                    maxvol.swap_tracker[j].mpi_root_B = p_in_A->mpi_host;
                    maxvol.swap_tracker[j].ind_B = links_sizes[p_in_A->mpi_host] - n - 1;   // the last element (freshely inserted)
                }

            swap(*p_in_B, *p_in_A);
        }

#   ifdef MLIP_DEBUG    // Make sure that data of the swapped links is the same on all processors
        int foo;
        MPI_Allreduce(&p_in_A->mpi_host, &foo, 1, MPI_INT, MPI_MAX, mpi.comm);
        if (foo != p_in_A->mpi_host)
            ERROR("Internal inclonsistency: hosting data is different on processors");
#   endif //MLIP_DEBUG
#else   // MLIP_MPI
        swap(eqn2cfg_links[n + swapka.ind_B], eqn2cfg_links[swapka.ind_A]);
#endif  // MLIP_MPI
    }

    return swap_count;
}

// moves configurations between cfgs_queue and selected_cfgs according to maxvol.swap_tracker. Clears maxvol.B. Does not clear cfgs_queue 
void CfgSelection::UpateSelected()              // moves configurations between cfgs_queue and selected_cfgs according to maxvol.swap_tracker. Clears maxvol.B. Does not clear cfgs_queue 
{
    // moving between 'selected_cfgs' and 'outsorted_cfgs' 
    auto ptr=selected_cfgs.begin();
    while (ptr != selected_cfgs.end())
        if (ptr->fitting_items.empty())
        //  selected_cfgs.erase(ptr++);
            outsorted_cfgs.splice(outsorted_cfgs.end(), selected_cfgs, ptr++);
        else
            ++ptr;

    // moving between 'cfgs_queue' and 'selected_cfgs' 
    ptr=cfgs_queue.begin();
    while (ptr != cfgs_queue.end())
        if (!ptr->fitting_items.empty())
            selected_cfgs.splice(selected_cfgs.end(), cfgs_queue, ptr++);
        else
            ++ptr;
}

int CfgSelection::Select(Configuration& cfg)
{
    if (cfg.nbh_cutoff != p_mlip->CutOff())
        cfg.InitNbhs(p_mlip->CutOff());

#ifndef MLIP_MPI    
    if (cfg.nbhs.size() == 0)
        return 0;
#endif 

    if (cfg_added_count > 0)
    {
        Warning("Performing multiple selection prior to selection of single configuration");
        Select();
    }
#ifdef MLIP_DEBUG
    //cout << cfg_added_count << " " << cfgs_queue.size() << " " << maxvol.B.size1 << " " << eqn2cfg_links.size() << " " << p_mlip->CoeffCount() << endl;
    if (cfg_added_count != 0 ||
        cfgs_queue.size() != 0 ||
        maxvol.B.size1 != 0 || 
        eqn2cfg_links.size() != p_mlip->CoeffCount())
            ERROR("Inconsistent state detected");
#endif // MLIP_DEBUG

    int row_count = PrepareMatrix(cfg);

    double grade = maxvol.CalcSwapGrade();
    cfg_eval_count++;

    double max_grade = 0;
    MPI_Allreduce(&grade, &max_grade, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    
    //cfg.features["MV_grade"] = to_string(grade);

    int swap_count = 0;
    if (max_grade > threshold_select)
    {
        cfgs_queue.emplace_back(cfg);
        PrepareLinks(cfgs_queue.back());

        swap_count = UpateLinks((swap_limit > 0) ? swap_limit : HUGE_INT);
        UpateSelected();
        if (cfg.nbhs.size() != 0)
            cfg_selct_count++;

        // clean up
        eqn2cfg_links.resize(p_mlip->CoeffCount()); // cut extra elements
        cfgs_queue.clear();
    }

    maxvol.B.resize(0, p_mlip->CoeffCount());

    std::stringstream logstrm1;
    logstrm1 << "Selection: cfg# " << cfg_eval_count
        << "\tgrade: " << grade
        << "\tswaps: " << swap_count
        << "\tselected: " << cfg_selct_count
        << "\tTS size: " << selected_cfgs.size()
        << "\tlog(vol): " << maxvol.log_n_det
        << endl;
    MLP_LOG(tagname, logstrm1.str());

    return swap_count;
}

void CfgSelection::Process(Configuration & cfg)
{
    if (batch_size != 1)
    {
        AddForSelection(cfg);
        if (batch_size != 0 && cfg_added_count >= batch_size)
            Select((swap_limit > 0) ? swap_limit : HUGE_INT);
    }
    else
        Select(cfg);
}

// Single selection of accumulated configuration. Also adds "MV_grade" feature to cfg. If some configurations are accumulated for multiple selection calls Select() before procissing cfg to finalize their selection.
int CfgSelection::Select(int swap_limit)                        // Single selection of accumulated configuration. Also adds "MV_grade" feature to cfg. If some configurations are accumulated for multiple selection calls Select() before procissing cfg to finalize their selection.
{
    int global_cfg_added_count=0;
    int local_cfg_added_count=(int)cfgs_queue.size();
    MPI_Allreduce(&local_cfg_added_count, &global_cfg_added_count, 1, MPI_INT, MPI_SUM, mpi.comm);
    if (global_cfg_added_count == 0)
        return 0;

#ifdef MLIP_DEBUG
    int sum = accumulate(eqn_counts.begin(), eqn_counts.end(), 0);
    if (sum != maxvol.B.size1 || sum != (eqn2cfg_links.size()-p_mlip->CoeffCount()))
        ERROR("Inconsistent state detected" + to_string(sum) + " " +to_string(maxvol.B.size1));
#endif // MLIP_DEBUG
        
    std::stringstream logstrm1;
    logstrm1 << "Selection: selecting " << global_cfg_added_count << " configurations" << endl;
    MLP_LOG(tagname, logstrm1.str()); logstrm1.str("");

    // 1. Grades calculation
    double local_max_grade = maxvol.CalcSwapGrade();

    cfg_eval_count += cfg_added_count;

    double max_grade = 0;
    MPI_Allreduce(&local_max_grade, &max_grade, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    
    // 2. Selection procedure
    int swap_count = 0;
    if (max_grade > threshold_select)
    {
        swap_count = UpateLinks(swap_limit);
        UpateSelected();
        cfg_selct_count++;
    }

    logstrm1 << "Selection: evaluated: " << cfg_eval_count
        << "\tmax_grade: " << local_max_grade
        << "\tswaps: " << swap_count
        << "\tTS size: " << selected_cfgs.size()
        << endl;
    MLP_LOG(tagname, logstrm1.str());

    cfg_added_count = 0;
    eqn_counts.clear();
    maxvol.B.resize(0, p_mlip->CoeffCount());
    eqn2cfg_links.resize(p_mlip->CoeffCount());
    cfgs_queue.clear();

    return swap_count;
}

void CfgSelection::LoadSelected(const string& filename)
{
    Message("Selection: loading and selecting configuration from " + filename);

#ifdef MLIP_MPI
    auto cfgs = MPI_LoadCfgs(filename);
#else
    auto cfgs = LoadCfgs(filename);
#endif
    for (Configuration& cfg : cfgs)
        AddForSelection(cfg);

    Select();

    cfg_eval_count = 0;
    cfg_selct_count = 0;

    Message("Selection: loading selection set complete");
}

void CfgSelection::SaveSelected(string filename)
{
    if (mpi.rank==0)
    {
        ofstream ofs(filename, ios::binary); // clean output file
        if (!ofs.is_open())
            ERROR("Can't open \"" + filename + "\" for writing configuration");
    }

    for (int rnk=0; rnk<mpi.size; rnk++)
    {
        if (mpi.rank == rnk && !selected_cfgs.empty())
        {
            ofstream ofs(filename, ios::binary | ios::app);
            if (!ofs.is_open())
                ERROR("Can't open \"" + filename + "\" for writing configuration");

            int cntr=0;
            for (Configuration& cfg : selected_cfgs)
                if (cfg.size() > 0)
                {
                    cfg.features.erase("selected_eqn_inds");
                    int first = *cfg.fitting_items.begin();
                    for (int ind : cfg.fitting_items)
                        cfg.features["selected_eqn_inds"] += ((ind == first) ? "" : ",") + to_string(ind);
                    cfg.features["AS_index"] = to_string(++cntr);
                    cfg.Save(ofs, Configuration::SAVE_GHOST_ATOMS);
                }
            ofs.close();
        }

        MPI_Barrier(mpi.comm);
    }

    Message("Selection: selected set saved to \"" + filename + '\"');
}

void CfgSelection::Save(const std::string & filename)
{
    ofstream ofs;

    if (mpi.rank == 0)
    {
        // 1. save MLIP
        p_mlip->Save(filename);
        // open file for write at the end
        ofs.open(filename, ios::binary | ios::app);
        ofs << "\n#MVS_v1.1\n"; // key symbol (end of mlip data)

        // 2. writing selection weights
        ofs << "energy_weight "  << scientific << setprecision(16) << MV_ene_cmpnts_weight << '\n';
        ofs << "force_weight "   << scientific << setprecision(16) << MV_frc_cmpnts_weight << '\n';
        ofs << "stress_weight "  << scientific << setprecision(16) << MV_str_cmpnts_weight << '\n';
        ofs << "site_en_weight " << scientific << setprecision(16) << MV_nbh_cmpnts_weight << '\n';
        ofs << "weight_scaling " << wgt_scale_power << '\n';
        ofs << "#"; // key symbol (end of weights data)

        // 3. writing matrices maxvol.A and maxvol.invA
        maxvol.WriteData(ofs);
        ofs << "\n#\n"; // key symbol (end of matrix data)
    }

#ifdef MLIP_MPI
    // 4. writing equation-to-configuration correspondence data
#   ifdef MLIP_DEBUG    // Make sure that links is the same on all processors (except .p_cfg)
    for (auto item : eqn2cfg_links)
    {
        int foo;
        MPI_Allreduce(&item.mpi_host, &foo, 1, MPI_INT, MPI_MAX, mpi.comm);
        if (foo != item.mpi_host)
            ERROR("Internal inclonsistency: hosting data are not synchronized among processors");
    }
#   endif //MLIP_DEBUG
    // 4.1 gathering an array with local numbers of configurations
    vector<int> global_index_shifts(mpi.size); 
    int lcl_cfg_count = (int)selected_cfgs.size();
    MPI_Allgather(&lcl_cfg_count, 1, MPI_INT, &global_index_shifts[0], 1, MPI_INT, mpi.comm);
    // 4.2 Caculating index shifts array
    int summ=0, prev=0;
    for (int i=0; i<global_index_shifts.size(); i++)
    {
        prev = global_index_shifts[i];
        global_index_shifts[i] = summ;
        summ += prev;
    }
    // 4.3 Constructing an array with configuration indices
    // 4.3.1 Finding global indices of configurations and filling the array on that processors
    vector<int> cfg_inds(p_mlip->CoeffCount());
    for (int i=0; i<p_mlip->CoeffCount(); i++)
        if (eqn2cfg_links[i].mpi_host == mpi.rank)  // only hosting processors work here
            if (eqn2cfg_links[i].p_cfg == nullptr)      // initial dummy data
                cfg_inds[i] = -1;
            else                                        // real (not initial) configuration data
            {
                // finding index 'ind' of the configuration in the list
                int ind = 0;
                for (auto& cfg : selected_cfgs)
                    if (&cfg == eqn2cfg_links[i].p_cfg)
                        break;
                    else
                        ind++;

                cfg_inds[i] = ind + global_index_shifts[mpi.rank];
            }
    // 4.3.2 Gathering configuration indices at all processors
    for (int i=0; i<p_mlip->CoeffCount(); i++)
        MPI_Bcast(&cfg_inds[i], 1, MPI_INT, eqn2cfg_links[i].mpi_host, mpi.comm);
    // 4.4 Gathering equation indices at all processors
    for (int i=0; i<p_mlip->CoeffCount(); i++)
        MPI_Bcast(&eqn2cfg_links[i].eqn_ind, 1, MPI_INT, eqn2cfg_links[i].mpi_host, mpi.comm);
    // 4.5 Writing equation-to-configuration correspondence data
    if (mpi.rank == 0)
    {
        for (int i=0; i<p_mlip->CoeffCount(); i++)
            ofs << cfg_inds[i] << ' ' << eqn2cfg_links[i].eqn_ind << '\n';
    
        ofs << "#\n";// key symbol (end of link data)

        ofs.close();
    }
#else //MLIP_MPI
    // 4. writing equation-to-configuration correspondence data
    for (EqnToCfgLink& link : eqn2cfg_links)
        if (link.p_cfg != nullptr)
        {
            // finding index 'ind' of the configuration in the list
            int ind = 0;
            for (auto& cfg : selected_cfgs)
                if (&cfg == link.p_cfg)
                    break;
                else
                    ind++;
#ifdef MLIP_DEBUG
            if (ind > selected_cfgs.size())
                ERROR("Inconsistent index value");
#endif // MLIP_DEBUG
            ofs << ind << ' ' << link.eqn_ind << '\n';
        }
        else    // uninitialized row
            ofs << -1 << ' ' << -1 << '\n';
    ofs << "#\n";// key symbol (end of link data)

    ofs.close();
#endif

    // 5. writing selected configurations
    for (int rank=0; rank<mpi.size; rank++)
    {
        if (rank == mpi.rank)
        {
            ofs.open(filename, ios::binary | ios::app);

            for (Configuration& cfg : selected_cfgs)
            {
                cfg.features.erase("selected_eqn_inds");
                int first = *cfg.fitting_items.begin();
                for (int ind : cfg.fitting_items)
                    cfg.features["selected_eqn_inds"] += ((ind == first) ? "" : ",") + to_string(ind);
                cfg.Save(ofs, Configuration::SAVE_NO_LOSS | Configuration::SAVE_GHOST_ATOMS);
            }

            ofs.close();
        }
        MPI_Barrier(mpi.comm);
    }

    Message("Selection: state saved to \"" + filename + '\"');
}

void CfgSelection::Load(const std::string & filename)
{
    Message("Selection: loading state from \"" + filename + '\"');

    // 1. loading mlip from file
    if (p_mlip == nullptr)
        ERROR("MLIP must be initialized before loading selection state");
    else
        p_mlip->Load(filename);
    ifstream ifs(filename, ios::binary);
    ifs.ignore(HUGE_INT, '#');
    if (ifs.fail() || ifs.eof())
        throw MlipException("No selection data provided with MLIP");

    // 2. loading selection weights
    if (ifs.peek() == 'M')
    {
        string tmpstr;
        ifs >> tmpstr;
        if (tmpstr != "MVS_v1.1")
            ERROR("Invalid MVS-file format");

        ifs >> tmpstr;
        if (tmpstr != "energy_weight")
            ERROR("Invalid MVS-file format");
        ifs >> MV_ene_cmpnts_weight;

        ifs >> tmpstr;
        if (tmpstr != "force_weight")
            ERROR("Invalid MVS-file format");
        ifs >> MV_frc_cmpnts_weight;

        ifs >> tmpstr;
        if (tmpstr != "stress_weight")
            ERROR("Invalid MVS-file format");
        ifs >> MV_str_cmpnts_weight;

        ifs >> tmpstr;
        if (tmpstr != "site_en_weight")
            ERROR("Invalid MVS-file format");
        ifs >> MV_nbh_cmpnts_weight;

        ifs >> tmpstr;
        if (tmpstr != "weight_scaling")
            ERROR("Invalid MVS-file format");
        ifs >> wgt_scale_power;

        ifs.ignore(HUGE_INT, '#');
    }
    else if (ifs.peek() == '\n')
    {
        ifs >> MV_ene_cmpnts_weight;
        ifs >> MV_frc_cmpnts_weight;
        ifs >> MV_str_cmpnts_weight;
        ifs >> MV_nbh_cmpnts_weight;

        ifs.ignore(HUGE_INT, '#');
    }
    else
        ERROR("No selection data found");

    Message("Selection: loaded the following weights:\n"
            "\tenergy_weight = " + to_string(MV_ene_cmpnts_weight) + "\n"
            "\tforce_weight = " + to_string(MV_frc_cmpnts_weight) + "\n"
            "\tstress_weight = " + to_string(MV_str_cmpnts_weight) + "\n"
            "\tsite_en_weight = " + to_string(MV_nbh_cmpnts_weight) + "\n");

    if (ifs.fail() || ifs.eof())
        ERROR("Error reading file " + filename);

    // 3. reading maxvol matrices
    maxvol.Reset(p_mlip->CoeffCount(), threshold_init); // reseting maxvol
    maxvol.ReadData(ifs);                               // reading matrices maxvol.A and maxvol.invA

    ifs.ignore(HUGE_INT, '#');
    if (ifs.fail() || ifs.eof())
        ERROR("Error reading file " + filename);

    // 4. loading equation-to-configuration correspondence data
    eqn2cfg_links.resize(p_mlip->CoeffCount());
    vector<int> cfg_indices(p_mlip->CoeffCount());
    for (int i=0; i<eqn2cfg_links.size(); i++)
        ifs >> cfg_indices[i]
            >> eqn2cfg_links[i].eqn_ind;

    ifs.ignore(HUGE_INT, '#');
    if (ifs.fail() || ifs.eof())
        ERROR("Error reading file \"" + filename + "\"");

#ifdef MLIP_MPI
    // 4.1 Correcting 'eqn2cfg_links' according to parllel treatment
    for (int i=0; i<eqn2cfg_links.size(); i++)
        eqn2cfg_links[i].mpi_host = 
            (cfg_indices[i] != -1) ? (cfg_indices[i] % mpi.size) : (i % mpi.size);

    // 5. loading selected configurations
    selected_cfgs.clear();
    Configuration foo;
    selected_cfgs.push_back(foo);
    for (int i=0; selected_cfgs.back().Load(ifs); i++)
        if (i % mpi.size == mpi.rank)
            selected_cfgs.push_back(foo);
    selected_cfgs.pop_back();
#else
    // 5. loading selected configurations
    selected_cfgs.clear();
    Configuration foo;
    selected_cfgs.push_back(foo);
    while (selected_cfgs.back().Load(ifs))
        selected_cfgs.push_back(foo);
    selected_cfgs.pop_back();
#endif

#ifdef MLIP_MPI
    // 6. setting up connection between selected equations and selected configurations
    for (int i=0; i<eqn2cfg_links.size(); i++)
        if (eqn2cfg_links[i].eqn_ind != -1 && eqn2cfg_links[i].mpi_host == mpi.rank)
        {
            // scrolling to the proper configuration
            auto p_cfg = selected_cfgs.begin();
            advance(p_cfg, cfg_indices[i] / mpi.size);
            eqn2cfg_links[i].p_cfg = &(*p_cfg);

            p_cfg->fitting_items.insert(eqn2cfg_links[i].eqn_ind);
        }
        else    // uninitialized row
            eqn2cfg_links[i].p_cfg = nullptr;
#else
    // 6. setting up connection between selected equations and selected configurations
    for (int i=0; i<eqn2cfg_links.size(); i++)
        if (eqn2cfg_links[i].eqn_ind != -1)
        {
            auto p_cfg = selected_cfgs.begin();
            advance(p_cfg, cfg_indices[i]);
            eqn2cfg_links[i].p_cfg = &(*p_cfg);

            p_cfg->fitting_items.insert(eqn2cfg_links[i].eqn_ind);
        }
        else    // uninitialized row
            eqn2cfg_links[i].p_cfg = nullptr;

#endif
    // 7. reset MV_grade feature for loaded cfgs
    if (mpi.rank == 0)
        for (auto& cfg : selected_cfgs)
          cfg.features["MV_grade"] = "0";

    Message("Selection: loading state complete.");
}
