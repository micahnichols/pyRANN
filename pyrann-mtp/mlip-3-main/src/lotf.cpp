/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */


#include "lotf.h"
#include "external_potential.h"
#include <typeinfo>
#include <sstream>


using namespace std;


const char* LOTF::tagname = {"lotf"};


LOTF::LOTF(AnyPotential * _p_abinitio,
    CfgSampling * _p_selector,
    AnyTrainer * _p_trainer,
    Settings _settings) :
    p_abinitio(_p_abinitio),
    p_selector(_p_selector),
    p_trainer(_p_trainer),
    EFS_via_abinitio(EFS_via_abinitio)
{
    if (p_selector->p_mlip != p_trainer->p_mlip)
        ERROR("p_selector->p_mlip != p_trainer->p_mlip");
    p_mlip = p_selector->p_mlip;

    if (p_selector->sampled_cfgs_fnm.empty())   // selection without saving sampled to a file is temporary unavailable. It requires splitting maxvol.B matrices between Cfg_Sampling and Cfg_selection.
        ERROR("Invalid settings: \"sample:save_extrapolative_to\" must be specified for MPI splitted configurations");

    InitSettings();
    ApplySettings(_settings);
    PrintSettings();

    if (!p_selector->sampled_cfgs_fnm.empty())
    {
        p_selector->sampled_cfgs_fnm += mpi.fnm_ending;
        ofstream ofs(p_selector->sampled_cfgs_fnm); // cleaning up sampling files
    }
    else
        if (p_selector->batch_size * mpi.size < max_sampled_count)
            ERROR("Inconsistent settings: select:batch_size * Ncores < lotf:sampling_limit");

    if (!lotf_log.empty() && lotf_log != "stdout" && lotf_log != "stderr")
    {
        lotf_log += mpi.fnm_ending;
        // clean output files
        std::ofstream ofs(lotf_log);
        if (!ofs.is_open())
            ERROR("Can not open file \""+lotf_log+"\" for output");
    }

    SetTagLogStream("lotf", lotf_log);
}

void LOTF::Select()
{
    Message("LOTF: Starting configuration selection");

    p_selector->track_changes = true;

    if (!p_selector->sampled_cfgs_fnm.empty())
    {
        ifstream ifs(p_selector->sampled_cfgs_fnm, ios::binary);
        if (!ifs.is_open())
            ERROR("Can't open file \""+p_selector->selected_cfgs_fnm+"\" for reading configurations");

        // combine all files in one to make the number of configurations more equal on processors
        if (mpi.size > 1)
        {
            string tmp_fnm = p_selector->sampled_cfgs_fnm.substr(0, p_selector->sampled_cfgs_fnm.length()-mpi.fnm_ending.length());
            for (int rnk=0; rnk<mpi.size; rnk++)
            {
                if (rnk == mpi.rank)
                {
                    ofstream ofs(tmp_fnm, ios::binary | ios::app);
                    ofs << ifs.rdbuf();
                }
                MPI_Barrier(mpi.comm);
            }

            ifs.close();
            ifs.open(tmp_fnm, ios::binary);
        }

        int cntr = 0;
        for (Configuration cfg; cfg.Load(ifs); cntr++)
            if (cntr % mpi.size == mpi.rank)
            {
                if (cfg.features.find("is_new") == cfg.features.end())   //
                    cfg.features["is_new"] = "yes";
                else
                    ERROR("Feature \"new\" is used as a mark involved in internal routines.\
                           configurations should not have it in LOTF.");
                p_selector->Process(cfg);
            }

        if ((cntr % mpi.size != 0) && mpi.rank >= (cntr % mpi.size))
        {
            Configuration cfg;  // empty cfg submitted. Important for parallel selection 
            p_selector->Process(cfg);
        }
    }
    else
    {
        for (auto& cfg : p_selector->cfgs_queue)
            if (cfg.features.find("is_new") == cfg.features.end())   //
                cfg.features["is_new"] = "yes";
            else
                ERROR("Feature \"new\" is used as a mark involved in internal routines.\
                           configurations should not have it in LOTF.");
    }

    p_selector->Select();

    new_cfgs.clear();
    for (Configuration& cfg : p_selector->selected_cfgs)
    {
        if (cfg.features["is_new"] == "yes")
        {
            new_cfgs.push_back(cfg);
            new_cfgs.back().features.erase("is_new");
        }
        cfg.features.erase("is_new");
    }

    if (!p_selector->sampled_cfgs_fnm.empty())
        ofstream ofs(p_selector->sampled_cfgs_fnm);  // cleaning sampling files

    sampled_count=0;

    Message("LOTF: Selection complete. "+to_string(new_cfgs.size())+" new configurations selected");
}

void LOTF::CalcAbInitio()
{
    if (typeid(*p_abinitio) == typeid(ExternalPotential)) // enables parallel abinitio treatment of configurations
    {
        ExternalPotential*p_external = (ExternalPotential*)p_abinitio;

        { if (mpi.rank==0) ofstream ofs(p_external->input_filename); } // clean output files
        { if (mpi.rank==0) ofstream ofs(p_external->output_filename); } // clean output files

        // writing new configurations to a file
        for (int rnk=0; rnk<mpi.size; rnk++)
        {
            if (mpi.rank == rnk)
                for (Configuration& cfg : new_cfgs)
                {
                    ofstream ofs(p_external->input_filename, ios::binary|ios::app);
                    if (!ofs.is_open())
                        ERROR("Can't open output file \"" + tmp_cfg_fnm + "\"");
                    cfg.Save(ofs);
                }

            MPI_Barrier(mpi.comm);
        }

        int pot_calcs_count_global, pot_calcs_count_local=(int)new_cfgs.size();
        MPI_Allreduce(&pot_calcs_count_local, &pot_calcs_count_global, 1, MPI_INT, MPI_SUM, mpi.comm);

        vector<int> cfgs_per_rank(mpi.size);
        MPI_Allgather(&pot_calcs_count_local, 1, MPI_INT, cfgs_per_rank.data(), 1, MPI_INT, mpi.comm);

        Message("LOTF: Starting abinitio calculations for " +
            to_string(pot_calcs_count_global) +
            " configuratuions");
        pot_calcs_count += pot_calcs_count_global;

        // starting external potential
        if (mpi.rank == 0)
        {
            int res = system(p_external->start_command.c_str());
            if (res != 0)
                Message("LOTF: Abinitio calculation script exits with code " + to_string(res));
        }

        MPI_Barrier(mpi.comm);

        // loading configuration from output file
        ifstream ifs(p_external->output_filename, ios::binary);
        if (!ifs.is_open())
            ERROR("Can not open file \"" + p_external->output_filename + "\" for reading configurations");

        if (!abinitio_changes_cfg)
        {
            Configuration cfg_tmp;
            for (int rnk=0; rnk<mpi.rank; rnk++)
                for (int i=0; i<cfgs_per_rank[rnk]; i++)
                    if (!cfg_tmp.Load(ifs))
                        ERROR("Cannot read configuration");

            for (Configuration& cfg : new_cfgs)
            {
                if (!cfg_tmp.Load(ifs))
                    ERROR("Cannot read configuration");

                string err_fnm = "ERROR.cfg_" + to_string(step_count) + "_" + to_string(mpi.rank);
                if (!p_external->CheckCfgNotChanged(cfg, cfg_tmp, err_fnm))
                    ERROR("Configuration changed after calculation on abinitio model");

                p_external->CopyEFS(cfg_tmp, cfg);
                for (auto feature : cfg_tmp.features)
                    cfg.features[feature.first] = feature.second;
            }
        }
        else
        {
            new_cfgs.clear();
            new_cfgs = MPI_LoadCfgs(p_external->output_filename);
            int pot_calcs_count_global2, pot_calcs_count_local = (int)new_cfgs.size();
            MPI_Allreduce(&pot_calcs_count_local, &pot_calcs_count_global2, 1, MPI_INT, MPI_MAX, mpi.comm);
            if (pot_calcs_count_global != pot_calcs_count_global2)
                ERROR("The number of configurations changed after abinitio calculation");
        }
        
        Message("LOTF: Reading abinitio data complete");
    }
    else
    {
        for (auto& item : new_cfgs)
        {
            p_abinitio->CalcEFS(item);
            pot_calcs_count++;
        }

        MPI_Barrier(mpi.comm);
    }
}

void LOTF::Retrain()
{
    Message("LOTF: Updating MLIP's parameters");

    if (replace_ts)
        train_set = vector<Configuration>(p_selector->selected_cfgs.begin(), p_selector->selected_cfgs.end());
    else
    {
        for (Configuration& cfg : new_cfgs)
            train_set.push_back(cfg);
    }

    if (!tmp_cfg_fnm.empty()) // saving and loading the training set to make the configuration distrubuted more equally among processors. Otherwise istribution of configurations among processors is not computationally efficient  
    {
        string orig_tmp_cfg_fnm = tmp_cfg_fnm;
        if (save_history)

        { if (mpi.rank == 0) ofstream ofs(tmp_cfg_fnm); } // clean up output file

        for (int rnk=0; rnk<mpi.size; rnk++)
        {
            if (mpi.rank == rnk)
            {
                ofstream ofs(tmp_cfg_fnm, ios::binary | ios::app);
                if (!ofs.is_open())
                    ERROR("Can't open output file \"" + tmp_cfg_fnm + "\"");
                for (Configuration& cfg : train_set)
                    cfg.Save(ofs);
            }

            MPI_Barrier(mpi.comm);
        }

        train_set.clear();
        train_set = MPI_LoadCfgs(tmp_cfg_fnm);

        tmp_cfg_fnm = orig_tmp_cfg_fnm;
    }

    p_trainer->Train(train_set);

    train_count++;

    Message("LOTF: MLIP's parameters update complete");
}

void LOTF::UpdateSelectionState()
{
    Message("LOTF: Updating selection state");

    new_cfgs.clear();

    p_selector->Reset();

    for (auto& cfg : train_set)
        p_selector->Process(cfg);

    p_selector->Select();

    if (save_history)
        p_selector->Save(p_trainer->mlip_fitted_fnm + "." + to_string(step_count));
    else
        p_selector->Save(p_trainer->mlip_fitted_fnm);

    Message("LOTF: Selection state updated");
}

void LOTF::WriteCfgToFile(Configuration& cfg, string filename)
{
    if (cfg.is_mpi_splited && (p_selector->MV_ene_cmpnts_weight!=0 || p_selector->MV_str_cmpnts_weight!=0))
        cfg.AppendToFile(p_selector->selected_cfgs_fnm);
    else
    {
        // Do not use AppendToFile. It involve MPI communication for splitted configuration
        ofstream ofs(filename, ios::binary | ios::app);
        if (mpi.size == 1)
            cfg.Save(ofs);
        else
            cfg.Save(ofs, Configuration::SAVE_GHOST_ATOMS);
    }
}

void LOTF::UpdateMLIP()
{
    Select();

    int local_new_size = (int)new_cfgs.size();
    int global_new_size = 0;
    MPI_Allreduce(&local_new_size, &global_new_size, 1, MPI_INT, MPI_SUM, mpi.comm);
    if (global_new_size == 0)
        return;

    CalcAbInitio();
    Retrain();
    UpdateSelectionState();
}

void LOTF::CalcEFS(Configuration & cfg)
{
    step_count++;

    if (cfg.is_mpi_splited && p_selector->sampled_cfgs_fnm.empty())
        ERROR("Invalid settings: \"sample:save_extrapolative_to\" must be specified for MPI splitted configurations");

    cfg.features["step"] = to_string(step_count);

    double grade = p_selector->Grade(cfg);
    double global_max_grade;
    MPI_Allreduce(&grade, &global_max_grade, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);

    if (grade < p_selector->threshold_sample)
        interp_cfg_count++;
    else
    {
        sampled_count++;
        extrap_cfg_count++;
        if (!p_selector->sampled_cfgs_fnm.empty())
            WriteCfgToFile(cfg, p_selector->sampled_cfgs_fnm);
        else
            p_selector->AddForSelection(cfg);
    }

    if (global_max_grade < p_selector->threshold_break)
    {
        p_mlip->CalcEFS(cfg);
        MTP_calcs_count++;
    }
    else
    {
        Message("LOTF: Maximal grade exceeded at step#" + to_string(step_count) + 
                ". Breaking calculations");
        UpdateMLIP();
        Message("LOTF: Mlip update is complete. Resuming calculations");

        // calculate EFS via mlip if necesary
        if (EFS_via_abinitio)
        {
            p_mlip->CalcEFS(cfg);
            MTP_calcs_count++;
        }
    }

    MPI_Allreduce(&sampled_count, &global_sampled_count, 1, MPI_INT, MPI_SUM, mpi.comm);
    if (global_sampled_count > max_sampled_count)
    {
        Message("LOTF: Sampled configuration limit exceeded at step#" + to_string(step_count) +
                ". Breaking calculations");
        UpdateMLIP();
        Message("LOTF: Mlip update is complete. Resuming calculations");
    }

    std::stringstream logstrm1;
    logstrm1<< "\tLOTF: cfg# "      << p_selector->cfg_eval_count
            << "\tAbinitio_calcs "  << pot_calcs_count
            << "\tMTP_calcs "       << MTP_calcs_count
            << "\tSampled "         << extrap_cfg_count
            << "\tAccumulated "     << sampled_count
            << "\tTrained "         << train_count
            << std::endl;
    MLP_LOG(tagname, logstrm1.str());
}

