/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "cfg_sampling.h"
#include <sstream>
#include <numeric>


using namespace std;


const char* CfgSampling::tagname = {"sample"};


CfgSampling::CfgSampling(const CfgSelection& selector, const Settings& _settings) :
    CfgSelection(selector)
{
    InitSettings();
    ApplySettings(_settings);
    PrintSettings();

    if (cfg_added_count > 0)
        ERROR("Selection object has configurations accumulated for selection but not selected");

    if (threshold_sample < 0.0)
        ERROR("Invalid threshold specified");

    if (!slct_log.empty() &&
        slct_log != "stdout" &&
        slct_log != "stderr")
        slct_log += mpi.fnm_ending;

    // clean output files
    if (!slct_log.empty() && slct_log!="stdout" && slct_log!="stderr")
        ofstream ofs(slct_log);

    if (!sampled_cfgs_fnm.empty())
    {
        { ofstream ofs(sampled_cfgs_fnm); }
        { ofstream ofs(sampled_cfgs_fnm + mpi.fnm_ending); }
        remove(sampled_cfgs_fnm.c_str());
        remove((sampled_cfgs_fnm + mpi.fnm_ending).c_str());
    }

    SetTagLogStream("sample", slct_log);
}

CfgSampling::CfgSampling(AnyLocalMLIP* _p_mlip, 
                         std::string load_state_from, 
                         const Settings & settings)
{
    p_mlip = _p_mlip;

    if (!load_state_from.empty())
        Load(load_state_from);

    InitSettings();
    ApplySettings(settings);
    PrintSettings();

    if (threshold_sample < 0.0)
        ERROR("Invalid threshold specified");

    if (!slct_log.empty() &&
        slct_log != "stdout" &&
        slct_log != "stderr")
        slct_log += mpi.fnm_ending;
    
    // clean output files
    if (!slct_log.empty() && slct_log!="stdout" && slct_log!="stderr")
        ofstream ofs(slct_log);

    if (!sampled_cfgs_fnm.empty())
    {
        { ofstream ofs(sampled_cfgs_fnm); }
        { ofstream ofs(sampled_cfgs_fnm + mpi.fnm_ending); }
        remove(sampled_cfgs_fnm.c_str());
        remove((sampled_cfgs_fnm + mpi.fnm_ending).c_str());
    }

    SetTagLogStream("sample", slct_log);
}

CfgSampling::~CfgSampling()
{
}

double CfgSampling::Grade(Neighborhood& nbh)
{
    if (MV_ene_cmpnts_weight != 0.0 &&
        MV_nbh_cmpnts_weight != 1.0 &&
        MV_frc_cmpnts_weight != 0.0 &&
        MV_str_cmpnts_weight != 0.0)
        ERROR("An attempt of inapropriate calculation of neighborhood grade detected");

    if (nbh.count > 0)
    {
        vector<double> tmp(p_mlip->CoeffCount());
        p_mlip->CalcSiteEnergyGrad(nbh, tmp);
        memcpy(&maxvol.B(0, 0), &tmp[0], tmp.size()*sizeof(double));
        maxvol.B.resize(0, p_mlip->CoeffCount());

        return maxvol.CalcSwapGrade();
    }
    else
        return 0.0;
}

double CfgSampling::Grade(Configuration& cfg)
{
    if (cfg.nbh_cutoff != p_mlip->CutOff())
        cfg.InitNbhs(p_mlip->CutOff());

    if (cfg_added_count > 0)
    {
        Warning("Performing multiple selection prior to selection of single configuration");
        Select();
    }

#ifdef MLIP_DEBUG
    if (cfg_added_count != 0 ||
        cfgs_queue.size() != 0 ||
        maxvol.B.size1 != 0 ||
        eqn2cfg_links.size() != p_mlip->CoeffCount())
        ERROR("Inconsistent state detected");
#endif // MLIP_DEBUG

    int row_count = PrepareMatrix(cfg);

    double max_local_grade, max_global_grade, max_grade;

    vector<double> grades;
    if (cfg.nbhs.size() == 0)
        max_grade = max_local_grade = 0.0;
    else if (MV_nbh_cmpnts_weight != 0)
    {
        eqn_counts.resize(cfg.nbhs.size(), 1);
        max_grade = max_local_grade = maxvol.CalcSwapGrade(&eqn_counts, &grades);
    }
    else
        max_grade = max_local_grade = maxvol.CalcSwapGrade();
    cfg_eval_count++;

    if (cfg.is_mpi_splited)
    {
        MPI_Allreduce(&max_local_grade, &max_global_grade, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);

        if (MV_ene_cmpnts_weight!=0 || MV_str_cmpnts_weight!=0)
            max_grade = max_global_grade;
    }

    if (add_grade_to_cfg)
    {
        cfg.features["MV_grade"] = to_string(max_grade);

        if (MV_nbh_cmpnts_weight != 0)
        {
            cfg.has_nbh_grades(true);
            memset(&cfg.nbh_grades(0), 0, cfg.size()*sizeof(double));
            for (int i=0; i<cfg.nbhs.size(); i++)
                cfg.nbh_grades(cfg.nbhs[i].my_ind) = grades[i];
        }
    }

    std::stringstream logstrm1;
    logstrm1 << "Sampling: cfg# " << cfg_eval_count
             << "\tgrade: " << max_grade
             << endl;
    MLP_LOG(tagname, logstrm1.str());

    maxvol.B.resize(0, p_mlip->CoeffCount());

    return max_grade;
}

void CfgSampling::Evaluate(Configuration & cfg)
{
    double max_grade = Grade(cfg);

    if (max_grade > threshold_sample && !sampled_cfgs_fnm.empty())
        if (cfg.is_mpi_splited && (MV_ene_cmpnts_weight!=0 || MV_str_cmpnts_weight!=0))
            cfg.AppendToFile(sampled_cfgs_fnm);
        else
        {
            // Do not use AppendToFile. It involve MPI communication for splitted configuration
            ofstream ofs(sampled_cfgs_fnm + mpi.fnm_ending, ios::binary | ios::app);
            if (mpi.size == 1)
                cfg.Save(ofs);
            else
                cfg.Save(ofs, Configuration::SAVE_GHOST_ATOMS);
        }

    if (threshold_break > threshold_sample)
    {
        if (cfg.is_mpi_splited)
        {
            double max_global_grade;
            MPI_Allreduce(&max_grade, &max_global_grade, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);

            if (max_global_grade > threshold_break)   // break on all processors
            {
                MPI_Barrier(mpi.comm);
                string mess;
                if (max_grade > threshold_break)
                    mess += "Breaking threshold exceeded (MV-grade: "+to_string(max_grade)+") on " +
                    to_string(mpi.rank) + " processor";
                throw MlipException(mess);
            }
        }
        else if (max_grade > threshold_break)                    // break only on the current process (unpredictable behavior when run on >1 cores)
            throw MlipException("Breaking threshold exceeded (MV-grade: "+to_string(max_grade)+")");
    }
}
