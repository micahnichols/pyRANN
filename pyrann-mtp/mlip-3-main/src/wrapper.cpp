/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Evgeny Podryabinkin
 */

#include <sstream>

#include "wrapper.h"
#include "mtp.h"
#include "linear_regression.h"
#include "pair_potentials.h"
#include "vasp_potential.h"
#include "lammps_potential.h"
#include "external_potential.h"
#include "mtpr_trainer.h"
#include "sw.h"


using namespace std;


const char* Wrapper::tagname = {"mlip"};

void Wrapper::SetUpMLIP(const Settings& settings)
{
    MLMTPR* p_mtpr = nullptr;
    LinearRegression* p_linreg = nullptr;
    StillingerWeberRadialBasis* p_sw = nullptr;

    // incorect scenarios
    if (!enable_mlip && !enable_abinitio && !enable_cfg_writing)
        ERROR("Nothing to be done");

    if (enable_abinitio)
        p_abinitio = p_external = new ExternalPotential(settings.ExtractSubSettings("abinitio"));
    else
        p_abinitio = new VoidPotential(); // trick for compatibility

    if (enable_mlip)
    {
        if (enable_select && enable_learn && !enable_lotf)
            ERROR("Configuration selection and MLIP fitting cannot be activated at the same time");
        if (enable_sampling && enable_learn && !enable_lotf)
            ERROR("Configuration sampling and MLIP fitting cannot be activated at the same time");
        if (enable_EFScalc && enable_learn && !enable_lotf)
            ERROR("EFS calculation and MLIP fitting cannot be activated at the same time");
        if (enable_sampling && enable_select && !enable_lotf)
            ERROR("Configuration sampling and selection cannot be activated at the same time");
        if (!enable_EFScalc && !enable_learn && !enable_sampling && !enable_select && !enable_lotf)
            ERROR("No mlip routines activated");
        if (enable_lotf && !(enable_abinitio && enable_EFScalc && enable_learn && enable_sampling && enable_select))
            ERROR("EFS calculation, fitting, selection and sampling must be enabled in lotf scenario");
        if (mlip_fnm.empty())
            ERROR("MLIP filename is not specified");
        if (enable_EFScalc && enable_abinitio && !enable_lotf)
            Warning("abinitio calculation will be overwriten by MLIP calculation");

        ifstream ifs(mlip_fnm, ios::binary);
        if (!ifs.is_open())
            ERROR("Can not open file \""+mlip_fnm+"\" for MLIP loading");
        ifs >> mlip_type;
        ifs.close();

        //cout << "mlip_fnm = " << mlip_fnm << endl;
        if (mlip_type == "MTP")
        {
            Message("MLIP type is non-linear MTP");
            p_mlip = p_mtpr = new MLMTPR(mlip_fnm);
        }
        else
            ERROR("unrecognized MLIP type \""+mlip_type+"\" detected in \""+mlip_fnm+"\"");

        if (!enable_lotf)
        {
            if (enable_sampling)
                p_sampler = new CfgSampling(p_mlip, 
                                            mlip_fnm, 
                                            settings.ExtractSubSettings("extrapolation_control"));

            if (enable_select)
                p_selector = new CfgSelection(p_mlip, settings.ExtractSubSettings("select"));

            if (enable_learn)
                p_learner = new MTPR_trainer(p_mtpr, settings.ExtractSubSettings("fit"));
        }
        else
        {
            try // attempting to read selection state from the file
            {
                p_sampler = new CfgSampling(p_mlip, 
                                            mlip_fnm, 
                                            settings.ExtractSubSettings("extrapolation_control"));

                CfgSelection tmp_sel(p_mlip, settings.ExtractSubSettings("select"));
                p_sampler->batch_size = tmp_sel.batch_size;
                p_sampler->swap_limit = tmp_sel.swap_limit;
                p_sampler->selected_cfgs_fnm = tmp_sel.selected_cfgs_fnm;
                p_sampler->state_save_fnm = tmp_sel.state_save_fnm;
                p_sampler->slct_log = tmp_sel.slct_log;
            }
            catch (MlipException& exception) // attempting to initilize selection from scratch
            {
                if (exception.message == "No selection data provided with MLIP")
                {
                    CfgSelection tmp_sel(p_mlip, settings.ExtractSubSettings("select"));
                    p_sampler = new CfgSampling(tmp_sel, settings.ExtractSubSettings("extrapolation_control"));
                }
                else
                    ERROR("Unable to initilaize LOTF.\n" + exception.message);
            }
            p_learner = new MTPR_trainer(p_mtpr, settings.ExtractSubSettings("fit"));
            p_lotf = new LOTF(p_external,
                              p_sampler,
                              p_learner,
                              settings.ExtractSubSettings("lotf"));
        }
    }
    else
        enable_lotf = enable_learn = enable_select = enable_sampling = enable_EFScalc = false;

    if (enable_err_check)
        p_errmon = new ErrorMonitor(settings.ExtractSubSettings("check_errors"));

    // checking and cleaning output configurations file (it is better to crash with error in the beginning)
    if (enable_cfg_writing)
    {
#ifdef MLIP_MPI
        string cfgs_fnm_many(cfgs_fnm);
        if (!cfgs_fnm.empty() &&
            cfgs_fnm != "stdout" &&
            cfgs_fnm != "stderr")
            cfgs_fnm_many += mpi.fnm_ending;
        
        ifstream ifs(cfgs_fnm_many);
        if (ifs.is_open())
        { // clean the content
            ifs.close();
            ofstream ofs_tmp(cfgs_fnm_many);
            if (!ofs_tmp.is_open())
                ERROR("Can't open .cfgs file \"" + cfgs_fnm_many + "\" for output");
			remove(cfgs_fnm_many.c_str());
        }
        if (mpi.rank == 0)
        {
            ifstream ifs(cfgs_fnm);
            if (ifs.is_open())
            { // clean the content
                ofstream ofs_tmp(cfgs_fnm);
                if (!ofs_tmp.is_open())
                    ERROR("Can't open .cfgs file \"" + cfgs_fnm + "\" for output");
            }
        }
#else
        ofstream ofs_tmp(cfgs_fnm);
        if (!ofs_tmp.is_open())
            ERROR("Can't open .cfgs file \"" + cfgs_fnm + "\" for output");
#endif
    }
}

Wrapper::Wrapper(const Settings& settings)
{
    InitSettings();
    ApplySettings(settings);

    Message("Wrapper initialization");
    
    if ( 0 == mpi.rank )
        PrintSettings();

    if (!log_output.empty() &&
        log_output != "stdout" &&
        log_output != "stderr")
            log_output += mpi.fnm_ending;

    if (!enable_lotf)
        log_output.clear();

    SetTagLogStream("mlip", log_output); // 

    SetUpMLIP(settings);

    Message("Wrapper initialization complete");
}

Wrapper::~Wrapper()
{
    Release();

    std::stringstream logstrm1;
    logstrm1 << endl;   // to keep previous messages printed with '\r'
    MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");

    if (p_errmon != nullptr)
        delete p_errmon;

    if (p_lotf != nullptr)
        delete p_lotf;

    if (p_selector != nullptr)
        delete p_selector;

    if (p_sampler != nullptr)
        delete p_sampler;

    if (p_learner != nullptr)
        delete p_learner;

    if (p_mlip != nullptr)
        delete p_mlip;

    if (p_abinitio != nullptr)
        delete p_abinitio;

    Message("Wrapper object has been destroyed");
}

void Wrapper::Process(Configuration & cfg)
{
    if (!enable_lotf || enable_err_check)
        p_abinitio->CalcEFS(cfg);

    if (enable_err_check)
        cfg_valid = cfg; // This placed before CalcEFS(cfg) for correct work with VoidPotential

    if (enable_lotf)
        p_lotf->CalcEFS(cfg);
    else
    {
        if (enable_sampling)
            p_sampler->Evaluate(cfg);

        if (enable_select)
            p_selector->Process(cfg);

        if (enable_learn)
        {
            training_set.push_back(cfg);
            learn_count++;
        }

        if (enable_EFScalc)
        {
            p_mlip->CalcEFS(cfg);
            MTP_count++;
        }
    }

    if (enable_err_check)
        p_errmon->AddToCompare(cfg_valid, cfg);

    if (!cfgs_fnm.empty() && (call_count % (skip_saving_N_cfgs+1) == 0))
    {
        if (mpi.LammpsCallbackComm != nullptr)
        {
            double* comm_arr = (cfg.has_forces() ? &cfg.force(0, 0) : nullptr);
            mpi.LammpsCallbackComm(cfg.p_void_pair, comm_arr);

            for (int i=0; i<cfg.size(); i++)
                if (cfg.ghost_inds.count(i) != 0)
                {
                    cfg.force(i, 0) = 0.0;
                    cfg.force(i, 1) = 0.0;
                    cfg.force(i, 2) = 0.0;
                }

            cfg.AppendToFile(cfgs_fnm);
        }
        else 
            if (cfg.is_mpi_splited)
                cfg.AppendToFile(cfgs_fnm);
            else
                if (cfg.size() > 0)
                    cfg.AppendToFile(cfgs_fnm + mpi.fnm_ending);
    }

    call_count++;
    std::stringstream logstrm1;
    logstrm1 << "MLIP: processed "  << call_count 
            << "\tAbinitio_calcs "  << abinitio_count
            << "\tMTP_calcs "       << MTP_count
            << "\tcfg_learned "     << learn_count
            << endl;
    MLP_LOG(tagname,logstrm1.str());
}

// function that does work configured according settings file (for example may perform fitting, EFS calculation, error caclulation, LOTF, etc.)

void Wrapper::Release()
{
    if (enable_lotf)
    {
        p_lotf->UpdateMLIP();
        return;
    }

    if (enable_learn)
    {
        
        int global_learn_count = 0;
        MPI_Allreduce(&learn_count, &global_learn_count, 1, MPI_INT, MPI_SUM, mpi.comm);
        if (global_learn_count > 0)
            p_learner->Train(training_set);
        learn_count = 0;
        if (mpi.rank == 0)
            p_learner->p_mlip->Save(p_learner->mlip_fitted_fnm);
    }

    if (enable_select)
    {
        p_selector->Select((p_selector->swap_limit>0) ? p_selector->swap_limit : HUGE_INT);
    }

    if (enable_err_check)
    {
        if (mpi.rank==0)
            p_errmon->GetReport();
    }

    MPI_Barrier(mpi.comm);
}
