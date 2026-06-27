#include <fstream>
#include <iostream>
#include <algorithm>


#include "../../src/common/utils.h"
#include "../../src/wrapper.h"
#include "../../src/configuration.h"
#include "../../src/drivers/basic_drivers.h"
#include "../../src/mtpr_trainer.h"

  
using namespace std;


#define FAIL(...) \
{std::string str("" __VA_ARGS__); \
std::cout << " failed!" << std::endl; \
if(!str.empty()) std::cout << std::endl << "Error message: " << str << std::endl; \
return false;}


// "MLIP_wrapper test1 (Recording cfgs, EFS via abinitio)"
bool utest_mw1()
{
    {ofstream ofs("temp/Track1.cfg"); }

    Settings settings;
    settings["mlip"] = "false";
    settings["write_cfgs"] = "true";
    settings["write_cfgs:to"] = "temp/Track1.cfg";
    Wrapper wrp(settings);

    CfgReader reader("Li_T450.cfgs", &wrp);
    reader.Run();

    ifstream ifs1("Li_T450.cfgs", ios::binary);
#ifndef MLIP_MPI
    ifstream ifs2("temp/Track1.cfg", ios::binary);
    Configuration cfg1, cfg2;
    while (cfg1.Load(ifs1))
    {
        cfg2.Load(ifs2);
        if (cfg1 != cfg2)
            FAIL();
    }
#else
    string fnm = "temp/Track1.cfg" + mpi.fnm_ending;
    ifstream ifs2(fnm, ios::binary);
    Configuration cfg1, cfg2;
    for( int cntr=0; cfg1.Load(ifs1); cntr++)
    {
        if (cntr%mpi.size == mpi.rank) {
        cfg2.Load(ifs2);
        if (cfg1 != cfg2)
            FAIL();
        }
    }
#endif
    return true;
}

// "MLIP_wrapper test2 (EFS via MTP, Recording cfgs, Error Monitoring)"
bool utest_mw2()
{
    {ofstream ofs("temp/Track2.cfg"); }
    {ofstream ofs("temp/delete_me.tmp"); }

    // update to the latest MTP-format
    //MTP mtp("MTP100_3all.mtp");
    //mtp.Save("MTP100_3all.mtp");

    Settings settings;
    settings["mlip"] = "true";
    settings["mlip:load_from"] = "MTP100_3all.mtp";
    settings["calculate_efs"] = "TRUE";
    settings["write_cfgs"] = "true";
    settings["write_cfgs:to"] = "temp/Track2.cfg";
    settings["check_errors"] = "true";
    settings["check_errors:report_to"] = "temp/errors.txt";
    settings["check_errors:log"] = "temp/errors.log";

    Wrapper wrp(settings);

    CfgReader reader("Li_T450.cfgs", &wrp);
    reader.max_cfg_cnt = 10;
    reader.Run();

    //cout << wrp.errmon.epa_aveabs() << endl << wrp.errmon.frc_aveabs() << endl << wrp.errmon.str_aveabs();
    if (
//      (wrp.errmon.cfg_cntr != 10) ||
        (fabs(wrp.p_errmon->epa_aveabs() - 0.000261901) > 1.0e-9) ||
        (fabs(wrp.p_errmon->frc_aveabs() - 0.00449916) > 1.0e-8) ||
        (fabs(wrp.p_errmon->str_aveabs() - 0.292422) > 1.0e-6)
        )
        FAIL();

    return true;
}

// "MLIP_wrapper test3 (Learning, Recording cfgs, EFS from ab-initio)"
bool utest_mw3()
{
    {ofstream ofs("temp/Track3.cfg"); }
    {ofstream ofs("temp/MTP9fitted.mtp"); }

    {
        Settings settings;
        settings["mlip"] = "true";
        settings["mlip:load_from"] = "MTP9.mtp";
        settings["fit"] = "TRUE";
        settings["fit:save_to"] = "temp/MTP9fitted.mtp";
        settings["fit:energy_weight"] = "1.0";
        settings["fit:force_weight"] = "0.001";
        settings["fit:stress_weight"] = "0.1";
        settings["fit:iteration_limit"] = "10";
        settings["fit:log"] = "temp/fitlog1.txt";
        settings["write_cfgs"] = "true";
        settings["write_cfgs:to"] = "temp/Track3.cfg";

        Wrapper wrp(settings);

        CfgReader reader("5cfgs.cfg", &wrp);
        reader.Run();
    }

    MLMTPR mtp("MTP9.mtp");
    Settings settings;
    settings["energy_weight"] = "1.0";
    settings["force_weight"] = "0.001";
    settings["stress_weight"] = "0.1";
    settings["iteration_limit"] = "10";
    settings["log"] = "temp/fitlog2.txt";

    MTPR_trainer lr(&mtp, settings);
    Configuration cfg;
    int cntr = 0;
    auto ts = MPI_LoadCfgs("5cfgs.cfg");
    lr.Train(ts);
    mtp.Save("temp/MTP9fitted.mtp2");

    ifstream ifs("5cfgs.cfg", ios::binary);
    cfg.Load(ifs);
    cfg.Load(ifs);
    cfg.Load(ifs);
    cfg.Load(ifs);    
    cfg.Load(ifs);
    Configuration cfg1(cfg), cfg2(cfg);
//    mtp.Load("temp/MTP9fitted.mtp2");
    mtp.CalcEFS(cfg1);

    MLMTPR mtp2("temp/MTP9fitted.mtp");
    mtp2.CalcEFS(cfg2);

    if (fabs(cfg1.energy - cfg2.energy) > 1.0e-10)
        FAIL("e1="+to_string(cfg1.energy) + ", e2="+to_string(cfg2.energy));
    if ((cfg1.stresses - cfg2.stresses).MaxAbs() > 1.0e-10)
        FAIL("sdiff="+to_string((cfg1.stresses - cfg2.stresses).MaxAbs()));

    return true;
} 

// "MLIP_wrapper test4 (Selection)"
bool utest_mw4()
{
    {ofstream ofs("temp/TS_9-1.cfg"); }

    Settings settings;
    settings["mlip"] = "true";
    settings["mlip:load_from"] = "MTP9fitted.mtp";
    settings["select"] = "TRUE";
    settings["select:threshold"] = "1.001";
    settings["select:batch_size"] = "999";
    settings["select:swap_limit"] = "10";
    settings["select:site_en_weight"] = "0.0";
    settings["select:energy_weight"] = "1.0";
    settings["select:force_weight"] = "0.0";
    settings["select:stress_weight"] = "0.0";
    settings["select:save_selected_to"] = "temp/TS_9-1.cfg";
    settings["select:weight_scaling"] = "0";

    {
        Wrapper wrp(settings);

        CfgReader reader("126cfgs1type.cfg", &wrp);
        reader.Run();
    }

    MLMTPR mtp("MTP9fitted.mtp");
    settings["threshold"] = "1.001";
    settings["batch_size"] = "999";
    settings["site_en_weight"] = "0.0";
    settings["energy_weight"] = "1.0";
    settings["force_weight"] = "0.0";
    settings["stress_weight"] = "0.0";
    settings["weight_scaling"] = "0";
    CfgSelection mv(&mtp, settings);

    Configuration cfg;
    int cntr = 0;
    for (ifstream ifs("126cfgs1type.cfg", ios::binary); cfg.Load(ifs); cntr++)
        if (cntr%mpi.size==mpi.rank)
            mv.Process(cfg);
    Configuration foo;
    if (cntr%mpi.size!=0 && mpi.rank>=cntr%mpi.size)
        mv.Process(foo);
    mv.Select(10);
    mv.SaveSelected("temp/TS_9-2.cfg");

    if (mpi.rank == 0)
    {
        ifstream ifs1("temp/TS_9-1.cfg", ios::binary);
        ifstream ifs2("temp/TS_9-2.cfg", ios::binary);
        Configuration cfgg1, cfgg2;
        while (cfgg1.Load(ifs1))
        {
            cfgg2.Load(ifs2);
            mtp.CalcEFS(cfgg1);
            mtp.CalcEFS(cfgg2);
            if (cfgg1.energy != cfgg2.energy)
                FAIL();
        }
    }
    return true;
}

// "MLIP_wrapper test5 (Loading selection + Sampling)"
bool utest_mw5()
{
    { ofstream ofs("temp/TS61-1.cfg"+mpi.fnm_ending, ios::binary); }
    { ofstream ofs("temp/TS61-3.cfg"+mpi.fnm_ending, ios::binary); }

    int cntr = 0;
    Configuration cfg;

    Settings settings;
    MLMTPR mtp("MTP61.mtp");
    CfgSelection mv2(&mtp, settings); // default selection settings
    mv2.LoadSelected("TS61-1.cfgs");
    mv2.Save("temp/test.mvs");
    settings["save_extrapolative_to"] = "temp/TS61-3.cfg";
    settings["threshold_save"] = "2.1";
    settings["threshold_break"] = "9999999999.";
    CfgSampling mv(mv2, settings);

    auto cfgs = MPI_LoadCfgs("Ti4MVtest.cfg");
    for (auto& cfg : cfgs)
        mv.Evaluate(cfg);

    {
        Settings settings;
        settings["mlip"] = "true";
        settings["mlip:load_from"] = "temp/test.mvs";
        settings["extrapolation_control"] = "TRUE";
        settings["extrapolation_control:save_extrapolative_to"] = "temp/TS61-1.cfg";
        settings["extrapolation_control:threshold_save"] = "2.1";
        settings["extrapolation_control:threshold_break"] = "9999999999.";

        Wrapper wrp(settings);
        CfgReader reader("Ti4MVtest.cfg", &wrp);
        reader.Run();
    }

#ifdef MLIP_MPI
    ifstream ifs1("temp/TS61-3.cfg"+mpi.fnm_ending, ios::binary);
    ifstream ifs2("temp/TS61-1.cfg"+mpi.fnm_ending, ios::binary);
#else
    ifstream ifs1("temp/TS61-3.cfg", ios::binary);
    ifstream ifs2("temp/TS61-1.cfg", ios::binary);
#endif
    Configuration cfgg1, cfgg2;
    for (int count=1; cfgg1.Load(ifs1); count++) 
    {
        cfgg2.Load(ifs2);
        if (cfgg1 != cfgg2)
            FAIL();
    }
    return true;
}

// "MLIP_wrapper test6 (Breaking treshold test)"
bool utest_mw6()
{
    {ofstream ofs("temp/TS_break.cfgs"); }
    {ofstream ofs("temp/test2.mvs"); }
    //{ofstream ofs("temp/MTP9lotf.mtp"); }
    //{ofstream ofs("temp/delete_me.tmp"); }
    //{ofstream ofs("temp/TS9lotf.cfg"); }

    MLMTPR mtp("MTP9fitted.mtp");

    Settings settings;
    //settings["select:threshold"] = "1.1"; 

    CfgSelection mv(&mtp, settings.ExtractSubSettings("select"));
    ////CfgSelection mv(&mtp, 0.000001, 1.1, 1.000001);
    mv.LoadSelected("B-1.cfgs");
    mv.Save("temp/test2.mvs");

    ////mv.threshold_select = 2.0;
    ////mv.Save("test2.mvs");
    ////mv.SaveSelected("temp/test7.cfg");
    int proc_err = 0;

    settings["mlip:load_from"] = "temp/test2.mvs";
    settings["mlip"] = "true";
    settings["calculate_efs"] = "TRUE";                             // Enables/disables EFS calculation by MTP (disabled learning may be usefull in pure learning/selection regime for best performance)
    settings["extrapolation_control"] = "TRUE";                                    // Activates/deactivates selection (active learning engine)
    settings["extrapolation_control:threshold_save"] = "4.0";                           // Selection threshold - maximum allwed extrapolation level
    settings["extrapolation_control:threshold_break"] = "30.0";                    // Selection threshold - maximum allwed extrapolation level
    settings["extrapolation_control:save_extrapolative_to"] = "temp/TS_break.cfgs";      // Selected configurations will be saved in this file after selection is complete. No configuration saved if not specified
    settings["extrapolation_control:add_grade_feature"] = "true";
    settings["abinitio"] = "FALSE";

    Wrapper wrp(settings);

    CfgReader reader("B.cfgs", &wrp);
    try
    {
        reader.Run();
    }
    catch (MlipException& excp)
    {
        if (excp.What() != (string)"Breaking threshold exceeded (MV-grade: 29.213068)")
        {
            cout << endl << excp.What() << "|" << endl;
            cout << "Breaking threshold exceeded (MV-grade: 29.213068)" << "|" << endl;
            proc_err=1;
        }
    }

    auto cfgs = LoadCfgs("temp/TS_break.cfgs"+mpi.fnm_ending);

    for (int i=0; i<cfgs.size(); i++)
    {
        if (stod(cfgs[i].features["MV_grade"]) < stod(settings["extrapolation_control:threshold_save"]))
            proc_err=1;
        if (stod(cfgs[i].features["MV_grade"]) > stod(settings["extrapolation_control:threshold_break"]) &&
            cfgs[i].features["MV_grade"] != "29.213068")
            proc_err=1;
    }

    int error = proc_err;

    MPI_Allreduce(&proc_err, &error, 1, MPI_INT, MPI_MAX, mpi.comm);

    if (error>0)
        FAIL();

    return true;
}

// "MLIP_wrapper test7 (LOTF)"
bool utest_mw7()
{
    {ofstream ofs("temp/test2.mvs"+mpi.fnm_ending); }
    {ofstream ofs("temp/MTP9lotf.mtp"); }
    {ofstream ofs("temp/ts_temp.cfg"); }
    {ofstream ofs("temp/lotf_sampled.cfg"+mpi.fnm_ending); }
    {ofstream ofs("temp/MTP9lotf_check.mtp"); }
    {ofstream ofs("temp/TS9lotf_check.cfgs"+mpi.fnm_ending); }

    {
        Settings settings;

        settings["mlip"] = "true";
        settings["mlip:load_from"] = "MTP9.mtp";
        settings["calculate_efs"] = "TRUE";                             // Enables/disables EFS calculation by MTP (disabled learning may be usefull in pure learning/selection regime for best performance)
        settings["fit"] = "TRUE";                           // Enables/disables MTP learning
        settings["fit:save_to"] = "temp/MTP9lotf.mtp";            // Output MTP file name (for trained MTP)
        settings["fit:energy_weight"] = "1.0";                      // Weight for energy equation in fitting procedure
        settings["fit:force_weight"] = "0.01";                      // Weight for forces equations in fitting procedure
        settings["fit:stress_weight"] = "0.1";                      // Weight for stresses equations in fitting procedure
        settings["fit:scale_by_force"] = "0.0";                     // Experimental setting. It is recommended to set it 0.0
        settings["fit:iteration_limit"] = "2";                     // Experimental setting. It is recommended to set it 0.0
        settings["select"] = "TRUE";                            // Activates/deactivates selection (active learning engine)
        settings["select:site_en_weight"] = "0.0";                      // Weight for site energy equations in selection procedure
        settings["select:energy_weight"] = "1.0";                       // Weight for energy equation in selection procedure
        settings["select:force_weight"] = "0.0";                        // Weight for forces equations in selection procedure
        settings["select:stress_weight"] = "0.0";                       // Weight for stresses equations in selection procedure
        settings["select:weight_scaling"] = "0.0";                       // Weight for stresses equations in selection procedure
        settings["select:save_selected_to"] = "temp/lotf_selected.cfg";                       // Weight for stresses equations in selection procedure
        settings["extrapolation_control"] = "true";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
        settings["extrapolation_control:threshold_save"] = "2.0";
        settings["extrapolation_control:threshold_break"] = "222.0";
        settings["extrapolation_control:save_extrapolative_to"] = "temp/lotf_sampled.cfg";
        settings["lotf"] = "true";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
        settings["lotf:fit_to_active_set"] = "FALSE";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
        settings["lotf:efs_only_by_mlip"] = "FALSE";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
        settings["lotf:sampling_limit"] = "20";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
        settings["lotf:save_ts_to"] = "temp/ts_temp.cfg";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
        settings["lotf:save_updates"] = "true";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
        settings["abinitio"] = "true";
        settings["abinitio:input_file"] = "foo";
        settings["abinitio:output_file"] = "foo";
        settings["abinitio:start_command"] = "foo";

        Wrapper wrp(settings);
        delete wrp.p_abinitio;
        wrp.p_abinitio = wrp.p_lotf->p_abinitio = new VoidPotential;
        CfgReader reader("B-2.cfgs", &wrp);
        reader.Run();
	}

    MPI_Barrier(mpi.comm); // some process may start reading "temp/MTP9lotf.mtp" until it is written

    MLMTPR mtp1("temp/MTP9lotf.mtp");
    MLMTPR mtp2("MTP9lotf_check.mtp");
    ifstream ifss("B-2.cfgs", ios::binary);
	Configuration cfgg1, cfgg2;
    cfgg1.Load(ifss);
    cfgg2 = cfgg1;
    mtp1.CalcEFS(cfgg1);
    mtp2.CalcEFS(cfgg2);
    if (fabs(cfgg1.energy - cfgg2.energy)/cfgg1.size() > 1.0e-1)
        FAIL();
    
    return true;
}

//// "MLIP_wrapper test8 (LOTF)"
//bool utest_mw8()
//{
//    {ofstream ofs("temp/MTP9state.mtp"); }
//    {ofstream ofs("temp/MTP9lotf.mtp"); }
//    {ofstream ofs("temp/ts_temp.cfg"); }
//    {ofstream ofs("temp/lotf_sampled.cfg"+mpi.fnm_ending); }
//    {ofstream ofs("temp/MTP9lotf_check.mtp"); }
//    {ofstream ofs("temp/TS9lotf_check.cfgs"+mpi.fnm_ending); }
//
//    Settings settings1;
//
//    MLMTPR mtp("MTP9fitted.mtp");
//    settings1["select:site_en_weight"] = "0.0";                      // Weight for site energy equations in selection procedure
//    settings1["select:energy_weight"] = "1.0";                       // Weight for energy equation in selection procedure
//    settings1["select:force_weight"] = "0.0";                        // Weight for forces equations in selection procedure
//    settings1["select:stress_weight"] = "0.0";                       // Weight for stresses equations in selection procedure
//    settings1["select:weight_scaling"] = "0.0";                       // Weight for stresses equations in selection procedure
//    CfgSelection sel(&mtp, settings1.ExtractSubSettings("select"));
//    sel.LoadSelected("B-2.cfgs");
//    sel.Save("temp/MTP9state.mtp");
//
//    Settings settings;
//
//    settings["mlip"] = "true";
//    settings["mlip:load_from"] = "temp/MTP9state.mtp";
//    settings["calculate_efs"] = "TRUE";                             // Enables/disables EFS calculation by MTP (disabled learning may be usefull in pure learning/selection regime for best performance)
//    settings["fit"] = "TRUE";                           // Enables/disables MTP learning
//    settings["fit:energy_weight"] = "1.0";                      // Weight for energy equation in fitting procedure
//    settings["fit:force_weight"] = "0.01";                      // Weight for forces equations in fitting procedure
//    settings["fit:stress_weight"] = "0.1";                      // Weight for stresses equations in fitting procedure
//    settings["fit:scale_by_force"] = "0.0";                     // Experimental setting. It is recommended to set it 0.0
//    settings["fit:iteration_limit"] = "2";                     // Experimental setting. It is recommended to set it 0.0
//    settings["select"] = "TRUE";                            // Activates/deactivates selection (active learning engine)
//    settings["select:batch_size"] = "99999";                            // Activates/deactivates selection (active learning engine)
//    settings["extrapolation_control"] = "true";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
//    settings["extrapolation_control:threshold"] = "2.0";
//    settings["extrapolation_control:threshold_break"] = "222.0";
//    settings["lotf"] = "true";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
//    settings["lotf:fit_to_active_set"] = "FALSE";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
//    settings["lotf:efs_only_by_mlip"] = "FALSE";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
//    settings["lotf:sampling_limit"] = "20";                       // FALSE - pass to driver QM data for EFS when learning occurs, TRUE - only MLIP-calculated data will be passed to driver
//    settings["abinitio"] = "true";
//    settings["abinitio:input_file"] = "foo";
//    settings["abinitio:output_file"] = "foo";
//    settings["abinitio:start_command"] = "foo";
//    settings["check_errors"] = "true";
//    settings["check_errors:report_to"] = "temp/lotf_errep.txt";
//
//    Wrapper wrp(settings);
//    delete wrp.p_abinitio;
//    wrp.p_abinitio = wrp.p_lotf->p_abinitio = new VoidPotential;
//    CfgReader reader("B.cfgs", &wrp);
//    reader.Run();
//    wrp.Release();
//
//    wrp.p_lotf->p_selector->Save("temp/MTP9lotf.mtp");
//
//    MLMTPR mtp1("temp/MTP9lotf.mtp");
//    MLMTPR mtp2("MTP9lotf_check.mtp");
//    ifstream ifss("B-2.cfgs", ios::binary);
//    Configuration cfgg1, cfgg2;
//    cfgg1.Load(ifss);
//    cfgg2 = cfgg1;
//    mtp1.CalcEFS(cfgg1);
//    mtp2.CalcEFS(cfgg2);
//    //if (fabs(cfgg1.energy - cfgg2.energy)/cfgg1.size() > 1.0e-0)
//    //    FAIL();
//
//    return true;
//}

