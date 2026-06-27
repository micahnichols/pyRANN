#include <fstream>
#include <iostream>
#include <algorithm>


#include "../../src/common/utils.h"
#include "../../src/maxvol.h"
#include "../../src/configuration.h"
#include "../../src/mtp.h"
#include "../../src/mtpr.h"
#include "../../src/cfg_sampling.h"
#include "../../src/linear_regression.h"
#include "../../src/lotf_linear.h"
#include "../../src/error_monitor.h"
 

using namespace std;


#define FAIL(...) \
{std::string str("" __VA_ARGS__); \
if(!str.empty()) std::cout << std::endl << "Error message: " << str << std::endl; \
return false;}

// "MaxVol initialization and selection")
bool utest_mv() {
    MaxVol maxvol(10, 1.0e-7);
    maxvol.B.resize(10, 10);

    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            maxvol.B(i, j) = i + j;
    for (int i = 0; i < 10; i++) {
        double norm_vect_row_sq = 0.0;
        for (int j = 0; j < 10; j++)
            norm_vect_row_sq += pow(maxvol.B(i, j), 2);
        for (int j = 0; j < 10; j++)
            maxvol.B(i, j) /= sqrt(norm_vect_row_sq);
    }

    maxvol.MaximizeVol(1.00000001);
    if (maxvol.InitedRowsCount() != 2)
        FAIL()

        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                maxvol.B(i, j) = pow(i + 1, j);
    for (int i = 0; i < 10; i++) {
        double norm_vect_row_sq = 0.0;
        for (int j = 0; j < 10; j++)
            norm_vect_row_sq += pow(maxvol.B(i, j), 2);
        for (int j = 0; j < 10; j++)
            maxvol.B(i, j) /= sqrt(norm_vect_row_sq);
    }

    maxvol.MaximizeVol(1.00000001);
    if (maxvol.InitedRowsCount() != maxvol.A.size1 - 1)
        FAIL();

    double fingerprint = 0.0;
    for (int i = 0; i < 10; i++)
        fingerprint += 10 * maxvol.A(i, i);
    if (fabs(fingerprint - 10.6768) > 1.0e-4)
        FAIL();

        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                maxvol.B(i, j) = (i == j) ? 1 : 0;

    maxvol.MaximizeVol(1.00000001);

    vector<double> rsum(10);
    vector<double> csum(10);
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++) {
            rsum[i] += maxvol.A(i, j);
            csum[j] += maxvol.A(i, j);
        }
    for (int i = 0; i < 10; i++)
        if (fabs(rsum.at(i) - 1.0) > 1e-6 || fabs(csum.at(i) - 1.0) > 1e-6)
            FAIL()
    return true;
}

// "MaxVol evaluation"
bool utest_mve() 
{
    MTP mtp("mlips/unfitted.mtp");
    VoidPotential pot;

    Settings settings;
    settings["select:site_en_weight"] = "0.25"; 
    settings["select:energy_weight"] = "0.25"; 
    settings["select:force_weight"] = "0.25";  
    settings["select:stress_weight"] = "0.25";  
    settings["select:update_active_set"] = "TRUE";

    CfgSampling mlip(CfgSelection(&mtp, settings.ExtractSubSettings("select")), settings.ExtractSubSettings("select"));
     
    //mlip.MV_nbh_cmpnts_weight = 0.25;
    //mlip.MV_ene_cmpnts_weight = 0.25;
    //mlip.MV_frc_cmpnts_weight = 0.25;
    //mlip.MV_str_cmpnts_weight = 0.25;
    std::vector<Configuration> cfgs;
    Configuration cfg;
    std::ifstream ifs("Ti4MVtest.cfg", ios::binary);
#ifndef MLIP_MPI
    while (cfg.Load(ifs))
        mlip.Select(cfg);
#else
//     int ncfg=0;
//     while(cfg.Load(ifs))
//         ncfg++;
//     ifs.close();
//     ifs.open("Ti4MVtest.proc3", ios::binary);
// 
//     for(int icfg=0;icfg < ncfg; ++icfg) {
//           cfg.Load(ifs);
//           if( icfg % mpi_size == mpi_rank )
//               mlip.Select(cfg);
//     }
//     cfg.destroy();
//     if( 0 != ncfg%mpi_size )
//         for(int icfg = ncfg%mpi_size; icfg < mpi_size; ++icfg) {
//           if( icfg % mpi_size == mpi_rank )
//               mlip.Select(cfg);
//         }
// #else
//     int cfg_counter=0;
//     while (cfg.Load(ifs))
//     {
//          if (cfg_counter++ % mpi_size != mpi_rank)
//                 cfg.destroy();
//          mlip.Select(cfg);
// 
//     }
    int cntr = 0;
    Configuration cfg0;
    for (Configuration cfg; cfg.Load(ifs); cntr++)
        if (cntr % mpi.size == mpi.rank)
            mlip.Select(cfg);
    if ((cntr % mpi.size != 0) && mpi.rank >= (cntr % mpi.size))
    {
       Configuration cfg;
       mlip.Select(cfg);
    }

#endif
    return true;

    ifs.close();
    ifs.open("configurations/reading_test.cfg", ios::binary);
    int ncfg=0;
    for (int i = 0; cfg.Load(ifs); i++, ncfg++)
#ifndef MLIP_MPI
        cfgs.push_back(cfg);
#else
        if (i % mpi.size == mpi.rank) 
            cfgs.push_back(cfg);
#endif
    ifs.close();

#ifdef GEN_TESTS
    std::ofstream ofs("test_MV_evaluation.txt");
    ofs.precision(15);

    mlip.MV_nbh_cmpnts_weight = 1.0;
    mlip.MV_ene_cmpnts_weight = 0.0;
    mlip.MV_frc_cmpnts_weight = 0.0;
    mlip.MV_str_cmpnts_weight = 0.0;
    for (int i = 0; i < cfgs.size(); i++)
          ofs << mlip.Grade(cfgs[i]) << std::endl;

    mlip.MV_nbh_cmpnts_weight = 0.0;
    mlip.MV_ene_cmpnts_weight = 1.0;
    mlip.MV_frc_cmpnts_weight = 0.0;
    mlip.MV_str_cmpnts_weight = 0.0;
    for (int i = 0; i < cfgs.size(); i++)
          ofs << mlip.Grade(cfgs[i]) << std::endl;

    mlip.MV_nbh_cmpnts_weight = 0.0;
    mlip.MV_ene_cmpnts_weight = 0.0;
    mlip.MV_frc_cmpnts_weight = 1.0;
    mlip.MV_str_cmpnts_weight = 0.0;
    for (int i = 0; i < cfgs.size(); i++)
          ofs << mlip.Grade(cfgs[i]) << std::endl;

    mlip.MV_nbh_cmpnts_weight = 0.0;
    mlip.MV_ene_cmpnts_weight = 0.0;
    mlip.MV_frc_cmpnts_weight = 0.0;
    mlip.MV_str_cmpnts_weight = 1.0;
    for (int i = 0; i < cfgs.size(); i++)
          ofs << mlip.Grade(cfgs[i]) << std::endl;

    ofs.close();
#endif

    ifs.open("test_MV_evaluation.txt", ios::binary);

    mlip.MV_nbh_cmpnts_weight = 1.0;
    mlip.MV_ene_cmpnts_weight = 0.0;
    mlip.MV_frc_cmpnts_weight = 0.0;
    mlip.MV_str_cmpnts_weight = 0.0;

    for (int i = 0, icfg = 0; i < ncfg; i++) {
        double grade;
        ifs >> grade;
#ifdef MLIP_MPI
        if (i % mpi.size != mpi.rank)
            continue;
#endif
        if (fabs(grade - mlip.Grade(cfgs[icfg])) > 1.0e-7)
            FAIL()
        icfg++;
      }

    mlip.MV_nbh_cmpnts_weight = 0.0;
    mlip.MV_ene_cmpnts_weight = 1.0;
    mlip.MV_frc_cmpnts_weight = 0.0;
    mlip.MV_str_cmpnts_weight = 0.0;
    for (int i = 0, icfg = 0; i < ncfg; i++) {
        double grade;
        ifs >> grade;
#ifdef MLIP_MPI
        if (i % mpi.size != mpi.rank)
            continue;
#endif
        if (fabs(grade - mlip.Grade(cfgs[icfg])) > 1.0e-7)
            FAIL()
        icfg++;
    }
 
    mlip.MV_nbh_cmpnts_weight = 0.0;
    mlip.MV_ene_cmpnts_weight = 0.0;
    mlip.MV_frc_cmpnts_weight = 1.0;
    mlip.MV_str_cmpnts_weight = 0.0;
    for (int i = 0, icfg = 0; i < ncfg; i++) {
        double grade;
        ifs >> grade;
#ifdef MLIP_MPI
        if (i % mpi.size != mpi.rank)
            continue;
#endif
        if (fabs(grade - mlip.Grade(cfgs[icfg])) > 1.0e-7)
            FAIL()
        icfg++;
    }

    mlip.MV_nbh_cmpnts_weight = 0.0;
    mlip.MV_ene_cmpnts_weight = 0.0;
    mlip.MV_frc_cmpnts_weight = 0.0;
    mlip.MV_str_cmpnts_weight = 1.0;
    for (int i = 0, icfg = 0; i < ncfg; i++) {
        double grade;
        ifs >> grade;
#ifdef MLIP_MPI
        if (i % mpi.size != mpi.rank)
            continue;
#endif
        if (fabs(grade - mlip.Grade(cfgs[icfg])) > 1.0e-7)
            FAIL()
        icfg++;
    }

    ifs.close();
    return true;
}

// "Configurations motion while MaxVol selection"
bool utest_mvcm() {
    VoidPotential pot;
    //double MV_trshld = 1.1;

    MTP mtp("mlips/unfitted.mtp");
    LinearRegression linreg(&mtp, Settings());

    Settings settings;
    settings["select:threshold"] = "1.1"; 
    settings["select:weight_scaling"] = "2";  

    CfgSelection MLIP(&mtp, settings.ExtractSubSettings("select"));

    ifstream ifs("Ti4MVtest.cfg", ios::binary);
    Configuration cfg;

    vector<int> swap_cnt_tracker;

    auto& selected = MLIP.selected_cfgs;
    for (int count = 1; cfg.Load(ifs); count++)
    {
        cfg.features["ind"] = to_string(count); 
        swap_cnt_tracker.push_back(MLIP.Select(cfg));
    }

    ifs.close();

#ifdef GEN_TESTS
    {
        ofstream ofs("test_MV_swaps.txt");
        for (int i = 0; i < swap_cnt_tracker.size(); i++)
            ofs << swap_cnt_tracker[i] << '\n';
        ofs.close();
        ofs.open("test_MV_selected.txt");
        for (Configuration& p_cfg : selected)
            ofs << cfg.features["ind"] << '\n';
        ofs.close();
    }
#endif

    ifs.open("test_MV_swaps.txt", ios::binary);
    for (int i = 0; i < (int)swap_cnt_tracker.size(); i++) {
        int foo;
        ifs >> foo;
        if (foo != swap_cnt_tracker[i])
            FAIL("wrong swaps");
    }
    ifs.close();
    ifs.open("test_MV_selected.txt", ios::binary);
    multiset<int> ids, ids_check;
    for (Configuration& cfgg : selected) {
        int foo;
        ifs >> foo;
        ids.insert(foo);
    }
    for (auto& cfgg : selected)
        ids_check.insert(stoi(cfgg.features["ind"]));

    if (ids != ids_check)
        FAIL("wrong ids");

    ifs.close();
    return true;
}

// "Maxvol selection unique cfgs and swap_limit tests"
bool utest_mvsl() 
{
    MTP mtp("MTP61.mtp");

    Settings settings;
    settings["select:threshold"] = "1.00001"; 
    settings["select:site_en_weight"] = "1"; 
    settings["select:energy_weight"] = "0"; 
    settings["select:force_weight"] = "0";  
    settings["select:stress_weight"] = "0"; 

    CfgSelection mv(&mtp, settings.ExtractSubSettings("select"));
    //CfgSelection mv(&mtp, 0.00001, 1.00001, 1.00001, 1, 0, 0, 0);

    if (mv.selected_cfgs.size() != 0)
        FAIL();
    mv.LoadSelected("Li.cfg");
    
    if (mpi.rank == 0)
        if (mv.selected_cfgs.size() != 1)
            FAIL();

    ifstream ifs("B.cfgs", ios::binary);
    Configuration cfg;
    for (int cntr=0; cfg.Load(ifs); cntr++)
    {
        int prev = (int)mv.selected_cfgs.size();
        mv.AddForSelection(cfg);
        mv.Select(1);
        if (mv.selected_cfgs.size() - prev > 1)
            FAIL();
    }
    return true;
}

//  "Multiple selection test#1"
bool utest_ms1() {

    int mpi_rank=0;
    int mpi_size=1;

#ifdef MLIP_MPI
    MPI_Comm_rank(mpi.comm, &mpi_rank);
    MPI_Comm_size(mpi.comm, &mpi_size);
#endif

    MTP mtp("MTP9prev.mtp");

    ifstream ifs("B.cfgs", ios::binary);
    vector<Configuration> cfgs;
    int cntr = 0;
    const int  mxncfg = 20;
#ifndef MLIP_MPI
    for (Configuration cfg; cfg.Load(ifs) && cntr<mxncfg; cntr++)
        cfgs.push_back(cfg);
#else
    for (Configuration cfg; cfg.Load(ifs) && cntr<mxncfg; cntr++)
        if (cntr % mpi_size == mpi_rank)
            cfgs.push_back(cfg);
    if ((cntr % mpi_size != 0) && mpi_rank >= (cntr % mpi_size))
        cfgs.push_back(Configuration());
#endif

    Settings settings;
    settings["select:threshold"] = "1.0000005"; 
    settings["select:update_active_set"] = "TRUE";
    settings["select:site_en_weight"] = "0"; 
    settings["select:energy_weight"] = "1"; 
    settings["select:force_weight"] = "0";  
    settings["select:stress_weight"] = "0"; 

    CfgSelection mv_single(&mtp, settings.ExtractSubSettings("select"));    
    //CfgSelection mv_single(&mtp, 0.0005, 1.0000005, 1.0000005, 0, 1, 0, 0);
    bool swaps = true;
    while (swaps)
    {
        swaps = false;
        for (Configuration& cfg : cfgs)
            swaps |= (mv_single.Select(cfg) > 0);
    }

    CfgSelection mv_mult(&mtp, settings.ExtractSubSettings("select"));
    //CfgSelection mv_mult(&mtp, 0.0005, 1.0000005, 1.0000005, 0, 1, 0, 0);
    for (Configuration& cfg : cfgs)
        mv_mult.AddForSelection(cfg);
    mv_mult.Select();

    vector<int> s1, s2;

    for (Configuration& cfg : mv_single.selected_cfgs)
        if (cfg.size()!=0)
            s1.push_back(stoi(cfg.features["number"]));

    for (Configuration& cfg2 : mv_mult.selected_cfgs)
        if (cfg2.size()!=0)
            s2.push_back(stoi(cfg2.features["number"]));


#ifdef MLIP_MPI
        MPI_Barrier(mpi.comm);
#endif

    std::ofstream ofs_s;
    std::ofstream ofs_m;
        
    for (int i=0;i<mpi_size;i++)
    {
        if (i==mpi_rank)
        {
            if (i==0)
            {
                ofs_s.open("temp/singl");
                ofs_m.open("temp/mult");
            }
            else
            {
                ofs_s.open("temp/singl",ios::app);
                ofs_m.open("temp/mult",ios::app);
            }

            for (int j=0;j<s1.size();j++)
                ofs_s << s1[j] << endl; 

            for (int j=0;j<s2.size();j++)
                ofs_m << s2[j]<< endl;  

            ofs_s.close();
            ofs_m.close();

        }
#ifdef MLIP_MPI
        MPI_Barrier(mpi.comm);
#endif
    }
        
    int total_single=0;
    int total_mult=0;

    int proc_err = 0;

    int s1s = (int)s1.size();
    int s2s = (int)s2.size();

#ifdef MLIP_MPI
MPI_Allreduce(&s1s, &total_single, 1, MPI_INT, MPI_SUM, mpi.comm);
MPI_Allreduce(&s2s, &total_mult, 1, MPI_INT, MPI_SUM, mpi.comm);
#endif

if (mpi_rank==0)
{
        
    if (total_single != total_mult)
        proc_err=1; 

    s1.resize(total_single);
    s2.resize(total_mult);


    std::ifstream ifs_s("temp/singl");
    std::ifstream ifs_m("temp/mult");

    string x;
    for (int j=0;j<total_single;j++)
    {
            ifs_s >> x; 
            s1[j]=std::stoi(x);
            ifs_m >> x;     
            s2[j]=std::stoi(x); 
    }

    ifs_s.close();
    ifs_m.close();

    for (int j=0;j<total_single;j++)
    {
        proc_err+=1;
        for (int k=0;k<total_single;k++)
            if (s1[j]==s2[k])
            {
                proc_err-=1;
                break;
            }
        
    }

    //for (int j=0;j<total_single;j++)
    //  cout << "single=" << s1[j] << endl; 

    //for (int j=0;j<total_single;j++)
    //  cout << "mult=" << s2[j] << endl;   

}

#ifdef MLIP_MPI
        MPI_Barrier(mpi.comm);
#endif

    int error = proc_err;

#ifdef MLIP_MPI
    MPI_Allreduce(&proc_err, &error, 1, MPI_INT, MPI_MAX, mpi.comm);
#endif
    
    if (error>0)
        FAIL();

    return true;
}

// "Multiple selection test#2"
bool utest_ms2() {
    MTP mtp("MTP9prev.mtp");

    ifstream ifs("B.cfgs", ios::binary);
    vector<Configuration> cfgs;
    int cntr = 0;
    const int  mxncfg = 20;
#ifndef MLIP_MPI
    for (Configuration cfg; cfg.Load(ifs) && cntr<mxncfg; cntr++)
        cfgs.push_back(cfg);
#else
    for (Configuration cfg; cfg.Load(ifs) && cntr<mxncfg; cntr++)
        if (cntr % mpi.size == mpi.rank)
            cfgs.push_back(cfg);
    if ((cntr % mpi.size != 0) && mpi.rank >= (cntr % mpi.size))
        cfgs.push_back(Configuration());
#endif

    Settings settings;
    settings["select:threshold"] = "1.000001"; 
    settings["select:update_active_set"] = "TRUE";
    settings["select:site_en_weight"] = "1"; 
    settings["select:energy_weight"] = "0"; 
    settings["select:force_weight"] = "0";  
    settings["select:stress_weight"] = "0"; 

    CfgSelection mv_single(&mtp, settings.ExtractSubSettings("select"));

    //CfgSelection mv_single(&mtp, 0.000001, 1.000001, 1.000001, 1, 0, 0, 0);
    for (Configuration& cfg : cfgs)
        mv_single.Select(cfg);

    CfgSelection mv_mult(&mtp, settings.ExtractSubSettings("select"));
    //CfgSelection mv_mult(&mtp, 0.000001, 1.000001, 1.000001, 1, 0, 0, 0);
    for (Configuration& cfg : cfgs)
    {
        mv_mult.AddForSelection(cfg);
        mv_mult.Select();
    }

    if (mv_single.selected_cfgs.size() != mv_mult.selected_cfgs.size())
        FAIL();

    multiset<int> s1, s2;

    for (auto& cfg : mv_single.selected_cfgs)
        for (int foo : cfg.fitting_items)
            s1.insert(stoi(cfg.features["number"]));
    for (auto& cfg2 : mv_mult.selected_cfgs)
        for (int foo : cfg2.fitting_items)
            s2.insert(stoi(cfg2.features["number"]));

    if (s1 != s2)
        FAIL();

    s1.clear(); s2.clear();

    for (Configuration& cfg : cfgs)
        mv_mult.AddForSelection(cfg);
    mv_mult.Select();

    bool swaps = true;
    while (swaps)
    {
        swaps = false;
        for (Configuration& cfg : cfgs)
            swaps |= (mv_single.Select(cfg)>0);
    }

    for (auto& cfg : mv_single.selected_cfgs)
        for (int foo : cfg.fitting_items)
            s1.insert(stoi(cfg.features["number"]));
    for (auto& cfg2 : mv_mult.selected_cfgs)
        for (int foo : cfg2.fitting_items)
            s2.insert(stoi(cfg2.features["number"]));

    if (s1 != s2)
        FAIL();

    return true;
}

// "MaxVol Saving/Loading selection test#1"
bool utest_mvsl1() 
{
    MTP mtp("MTP61.mtp");

    ifstream ifs("B.cfgs", ios::binary);
    vector<Configuration> cfgs;
    int cntr = 0;
    const int  mxncfg = 2;
#ifndef MLIP_MPI
    for (Configuration cfg; cfg.Load(ifs) && cntr<2; cntr++)
        cfgs.push_back(cfg);
#else
    for (Configuration cfg; cfg.Load(ifs) && cntr<mxncfg; cntr++)
        if (cntr % mpi.size == mpi.rank)
            cfgs.push_back(cfg);
    if ((cntr % mpi.size != 0) && mpi.rank >= (cntr % mpi.size))
        cfgs.push_back(Configuration());
#endif

    Settings settings;
    settings["select:threshold"] = "1.0000001"; 
    settings["select:update_active_set"] = "TRUE";
    settings["select:site_en_weight"] = "1"; 
    settings["select:energy_weight"] = "0"; 
    settings["select:force_weight"] = "0";  
    settings["select:stress_weight"] = "0"; 

    CfgSelection mv(&mtp, settings.ExtractSubSettings("select"));

    //CfgSelection mv(&mtp, 0.0000001, 1.0000001, 1.0000001, 1, 0, 0, 0);
    for (Configuration& cfg : cfgs)
        mv.AddForSelection(cfg);
    mv.Select();

    // 0. Check for errors while saving/loading
    MTP foo("MTP9.mtp");
    settings["select:update_active_set"] = "TRUE";
    settings["select:site_en_weight"] = "16"; 
    settings["select:energy_weight"] = "45"; 
    settings["select:force_weight"] = "123";  
    settings["select:stress_weight"] = "543"; 
    CfgSelection mv_check(&mtp, settings.ExtractSubSettings("select"));
    //CfgSelection mv_check(&foo, 0.0000001, 1.0000001, 1.0000001, 16, 45, 123, 543); // Object to be loaded
    try
    {
        mv.Save("temp/maxvol_state.mvs");
        mv_check.Load("temp/maxvol_state.mvs");
    }
    catch (...)
        FAIL();

    // 1. Check cfg count 
#ifndef MLIP_MPI
    if (mv.selected_cfgs.size() != mv_check.selected_cfgs.size())
        FAIL();
#else
    {
        int size,size_check;
        int cfgsize = (int)mv.selected_cfgs.size();
        MPI_Allreduce(&cfgsize, &size, 1, MPI_INT, MPI_SUM, mpi.comm);
        cfgsize = (int)mv_check.selected_cfgs.size();
        MPI_Allreduce(&cfgsize, &size_check, 1, MPI_INT, MPI_SUM, mpi.comm);
        if (size != size_check)
            FAIL();
    }
#endif

    // 2. Check for selected set content
#ifndef MLIP_MPI
    map<int, Configuration> ordered_cfg_set1;
    map<int, Configuration> ordered_cfg_set2;

    for (Configuration mv_cfg : mv.selected_cfgs)
        ordered_cfg_set1.insert(make_pair(stoi(mv_cfg.features["number"]), mv_cfg));
    for (Configuration mv_cfg : mv_check.selected_cfgs)
        ordered_cfg_set2.insert(make_pair(stoi(mv_cfg.features["number"]), mv_cfg));

    for (auto& item : ordered_cfg_set1)
    {
        if (item.second != ordered_cfg_set2.at(item.first))
            FAIL();
        if (item.second.fitting_items != ordered_cfg_set2.at(item.first).fitting_items)
            FAIL();
    }
#else
    int root = 0;
    int size, *psizes0=nullptr;
    int *displs=nullptr,i,*rcounts=nullptr, *rbuff=nullptr;

    vector<int> vind,vind_check;
    vector<int> vall,vall_check;
    for (Configuration mv_cfg : mv.selected_cfgs)
        vind.push_back(stoi(mv_cfg.features["number"]));
    for (Configuration mv_cfg : mv_check.selected_cfgs)
        vind_check.push_back(stoi(mv_cfg.features["number"]));
    
    if (root == mpi.rank) {
        psizes0 = new int[mpi.size];
        displs =  new int[mpi.size]; 
        rcounts = new int[mpi.size]; 
    }

    size = (int)vind.size();
    MPI_Gather(&size, 1, MPI_INT, psizes0, 1, MPI_INT, root, mpi.comm);

    if (root == mpi.rank) {
        size=0;
        for (i=0; i < mpi.size; ++i) { 
            displs[i] = i*psizes0[i]; 
            rcounts[i] = psizes0[i];
            size += psizes0[i];
        } 
        vall.resize(size);
        rbuff = &vall[0];
    }

    MPI_Gatherv(&vind[0],(int)vind.size(),MPI_INT,rbuff,rcounts,displs,MPI_INT,root,mpi.comm);

    size = (int)vind_check.size();
    MPI_Gather(&size, 1, MPI_INT, psizes0, 1, MPI_INT, root, mpi.comm);

    if (root == mpi.rank) {
        size=0;
        for (i=0; i < mpi.size; ++i) { 
            displs[i] = i*psizes0[i]; 
            rcounts[i] = psizes0[i];
            size += psizes0[i];
        } 
        vall_check.resize(size);
        rbuff = &vall_check[0];
    }

    MPI_Gatherv(&vind_check[0], (int)vind_check.size(), MPI_INT, rbuff, rcounts, displs, MPI_INT,
                root, mpi.comm);

    if (root == mpi.rank) {
        delete [] psizes0;
        delete [] displs;
        delete [] rcounts;
    }

    if (root == mpi.rank) {
        sort(vall.begin(),vall.end());
        sort(vall_check.begin(),vall_check.end());
        if( !equal(vall.begin(), vall.end(), vall_check.begin()) )
            FAIL();
    }

    vind.clear();vind_check.clear();
    vall.clear();vall_check.clear();

#endif

    // 3. Check for maxvol matrices
    if (mv.maxvol.A.size1 != mv_check.maxvol.A.size1 ||
        mv.maxvol.A.size1 != mv_check.maxvol.A.size2)
        FAIL();

    for (int i=0; i<mv.maxvol.A.size1*mv.maxvol.A.size2; i++)
        if (*(&mv.maxvol.A(0, 0) + i) != *(&mv_check.maxvol.A(0, 0) + i))
            FAIL();

    // 4. Check selected set after additional selection
#ifndef MLIP_MPI
    for (Configuration cfg; cfg.Load(ifs); cntr++)
    {
        mv.AddForSelection(cfg);
        mv_check.AddForSelection(cfg);
    }
    mv.Select();
    mv_check.Select();
#else
    cntr = 0;
//     for (Configuration cfg; cfg.Load(ifs); cntr++)
//         if (cntr % mpi_size == mpi_rank) {
//             mv.AddForSelection(cfg);
//             mv_check.AddForSelection(cfg);
//         }
//     int ncfg = cntr; ++ncfg;
//     for( Configuration cfg; cntr < (ncfg/mpi_size)*mpi_size; cntr++)
//         if (cntr % mpi_size == mpi_rank) {
//             mv.AddForSelection(cfg);
//             mv_check.AddForSelection(cfg);
//         }
//     mv.Select();
//     mv_check.Select();
   for (Configuration cfg; cfg.Load(ifs); cntr++)
    {
        mv.AddForSelection(cfg);
        mv_check.AddForSelection(cfg);
    }
    mv.Select();
    mv_check.Select();

#endif

#ifndef MLIP_MPI
    ordered_cfg_set1.clear();
    ordered_cfg_set2.clear();

    for (Configuration& mv_cfg : mv.selected_cfgs)
        ordered_cfg_set1.insert(make_pair(stoi(mv_cfg.features["number"]), mv_cfg));
    for (Configuration& mv_cfg : mv_check.selected_cfgs)
        ordered_cfg_set2.insert(make_pair(stoi(mv_cfg.features["number"]), mv_cfg));

    for (auto& item : ordered_cfg_set1)
    {
        if (item.second != ordered_cfg_set2.at(item.first))
            FAIL();
        if (item.second.fitting_items != ordered_cfg_set2.at(item.first).fitting_items)
            FAIL();
    }
#else
    for (Configuration mv_cfg : mv.selected_cfgs)
        vind.push_back(stoi(mv_cfg.features["number"]));
    for (Configuration mv_cfg : mv_check.selected_cfgs)
        vind_check.push_back(stoi(mv_cfg.features["number"]));
    
    if (root == mpi.rank) {
        psizes0 = new int[mpi.size];
        displs =  new int[mpi.size]; 
        rcounts = new int[mpi.size]; 
    }

    size = (int)vind.size();
    MPI_Gather(&size, 1, MPI_INT, psizes0, 1, MPI_INT, root, mpi.comm);

    if (root == mpi.rank) {
        size=0;
        for (i=0; i < mpi.size; ++i) { 
            displs[i] = i*psizes0[i]; 
            rcounts[i] = psizes0[i];
            size += psizes0[i];
        } 
        vall.resize(size);
        rbuff = &vall[0];
    }

    MPI_Gatherv(&vind[0],(int)vind.size(),MPI_INT,rbuff,rcounts,displs,MPI_INT,root,mpi.comm);

    size = (int)vind_check.size();
    MPI_Gather(&size, 1, MPI_INT, psizes0, 1, MPI_INT, root, mpi.comm);

    if (root == mpi.rank) {
        size=0;
        for (i=0; i < mpi.size; ++i) { 
            displs[i] = i*psizes0[i]; 
            rcounts[i] = psizes0[i];
            size += psizes0[i];
        } 
        vall_check.resize(size);
        rbuff = &vall_check[0];
    }

    MPI_Gatherv(&vind_check[0], (int)vind_check.size(), MPI_INT, rbuff, rcounts, displs, MPI_INT,
                root, mpi.comm);

    if (root == mpi.rank) {
        delete [] psizes0;
        delete [] displs;
        delete [] rcounts;
    }

    if (root == mpi.rank) {
        sort(vall.begin(),vall.end());
        sort(vall_check.begin(),vall_check.end());
        if( !equal(vall.begin(), vall.end(), vall_check.begin()) )
            FAIL();
    }

#endif

    return true;
}

//TEST("MaxVol Saving/Loading selection test#2") {
//  MLMTPR mtp("PdAg_20g_str.mtp");
//
//  ifstream ifs("20g_Selected.cfgs", ios::binary);
//  vector<Configuration> cfgs;
//  int cntr = 0;
//  for (Configuration cfg; cfg.Load(ifs) && cntr<2; cntr++)
//      cfgs.push_back(cfg);
//
//  CfgSelection mv(&mtp, 0.0000001, 0.1, 0.0000001, 0, 1, 0, 0);
//  for (Configuration& cfg : cfgs)
//      mv.AddForSelection(cfg);
//  mv.Select();
//
//  MLMTPR foo("PdAg_20g_str.mtp");
//  CfgSelection mv_check(&foo, 0.0000001, 0.0000001, 0.0000001, 16, 45, 123, 543);
//  try
//  {
//      mv.Save("temp/maxvol_state.mvs");
//      mv_check.Load("temp/maxvol_state.mvs");
//  }
//  catch (...)
//      FAIL();
//
//  for (int i=0; i<mv.selected_cfgs.size(); i++)
//      if (mv.selected_cfgs[i] != mv_check.selected_cfgs[i])
//          FAIL();
//
//  if (mv.UniqueCfgCount() != mv_check.UniqueCfgCount())
//      FAIL();
//
//  if (mv.maxvol.A.size1 != mv_check.maxvol.A.size1 ||
//      mv.maxvol.A.size1 != mv_check.maxvol.A.size2)
//      FAIL();
//
//  for (int i=0; i<mv.maxvol.A.size1*mv.maxvol.A.size2; i++)
//      if (*(&mv.maxvol.A(0, 0) + i) != *(&mv_check.maxvol.A(0, 0) + i))
//          FAIL();
//
//  for (Configuration cfg; cfg.Load(ifs); cntr++)
//  {
//      mv.AddForSelection(cfg);
//      mv_check.AddForSelection(cfg);
//  }
//  mv.Select();
//  mv_check.Select();
//
//  for (int i=0; i<mv.selected_cfgs.size(); i++)
//      if (mv.selected_cfgs[i] != mv_check.selected_cfgs[i])
//          FAIL();
//
//} END_TEST;

// "parallel selection test"
bool utest_prlsel()
{
    int mpi_rank=0;
    int mpi_size=1;
#ifdef MLIP_MPI
    MPI_Comm_rank(mpi.comm, &mpi_rank);
    MPI_Comm_size(mpi.comm, &mpi_size);

    auto cfgs = MPI_LoadCfgs("B.cfgs");
#else
    auto cfgs = LoadCfgs("B.cfgs");
#endif
    MTP mtp("MTP61.mtp");

    Settings settings;
    settings["select:update_active_set"] = "TRUE";
    settings["select:threshold"] = "1.1"; 
    settings["select:site_en_weight"] = "1"; 
    settings["select:energy_weight"] = "0"; 
    settings["select:force_weight"] = "0";  
    settings["select:stress_weight"] = "0"; 

    CfgSelection select_serial1(&mtp, settings.ExtractSubSettings("select"));
    CfgSelection select_serial2(&mtp, settings.ExtractSubSettings("select"));
    CfgSelection select_serial3(&mtp, settings.ExtractSubSettings("select"));
    CfgSelection select_parall(&mtp, settings.ExtractSubSettings("select"));

    //CfgSelection select_serial1(&mtp, 1.0e-7, 1.1, 1.0001, 1, 0, 0, 0);
    //CfgSelection select_serial2(&mtp, 1.0e-7, 1.1, 1.0001, 1, 0, 0, 0);
    //CfgSelection select_serial3(&mtp, 1.0e-7, 1.1, 1.0001, 1, 0, 0, 0);
    //CfgSelection select_parall(&mtp, 1.0e-7, 1.1, 1.0001, 1, 0, 0, 0);

    ifstream ifs("B.cfgs", ios::binary);

    for (int i=0; i<cfgs.size(); i++)
    {
        select_parall.Select(cfgs[i]);

        Configuration cfg2;
        for (int j=0; j<mpi_size; j++)
            if (cfg2.Load(ifs))
            {
                if (j % mpi_size == mpi_rank)
                    select_serial1.AddForSelection(cfg2);
                if (mpi_rank == 0)
                    select_serial2.AddForSelection(cfg2);
                select_serial3.AddForSelection(cfg2);
            }
        select_serial1.Select();
        select_serial2.Select();
        select_serial3.Select();

        multiset<int> ss1;
        multiset<int> ss2;
        multiset<int> ss3;
        multiset<int> sp;

        for (auto& cfg : select_serial1.selected_cfgs)
            ss1.insert(stoi(cfg.features.at("number")));
        for (auto& cfg : select_serial2.selected_cfgs)
            ss2.insert(stoi(cfg.features.at("number")));
        for (auto& cfg : select_serial3.selected_cfgs)
            ss3.insert(stoi(cfg.features.at("number")));
        for (auto& cfg : select_parall.selected_cfgs)
            sp.insert(stoi(cfg.features.at("number")));

        vector<int> sizes1(mpi_size);
        vector<int> sizes2(mpi_size);
        vector<int> sizes3(mpi_size);
        vector<int> sizes4(mpi_size);
#ifdef MLIP_MPI
        {
            int tmp = (int)ss1.size();
            MPI_Allgather(&tmp, 1, MPI_INT, sizes1.data(), 1, MPI_INT, mpi.comm);
            vector<int> send_buff(sizes1[mpi_rank]);
            int sum_sizes = 0;
            vector<int> displs(mpi_size);
            displs[0] = 0;
            for (int i=1; i<mpi_size; i++)
                displs[i] = (sum_sizes += sizes1[i-1]);
            sum_sizes += sizes1[mpi_size-1];
            vector<int> recv_buff(sum_sizes);
            multiset<int>::iterator itr=ss1.begin();
            for (int i=0; i<ss1.size(); i++)
                send_buff[i] = *(itr++);
            MPI_Allgatherv(send_buff.data(), (int)send_buff.size(), MPI_INT, recv_buff.data(),
                sizes1.data(), displs.data(), MPI_INT, mpi.comm);
            ss1.clear();
            for (int el : recv_buff)
                ss1.insert(el);
        }
        {
            int tmp = (int)ss2.size();
            MPI_Allgather(&tmp, 1, MPI_INT, sizes2.data(), 1, MPI_INT, mpi.comm);
            vector<int> send_buff(sizes2[mpi_rank]);
            int sum_sizes = 0;
            vector<int> displs(mpi_size);
            displs[0] = 0;
            for (int i=1; i<mpi_size; i++)
                displs[i] = (sum_sizes += sizes2[i-1]);
            sum_sizes += sizes2[mpi_size-1];
            vector<int> recv_buff(sum_sizes);
            multiset<int>::iterator itr=ss2.begin();
            for (int i=0; i<ss2.size(); i++)
                send_buff[i] = *(itr++);
            MPI_Allgatherv(send_buff.data(), (int)send_buff.size(), MPI_INT, recv_buff.data(),
                sizes2.data(), displs.data(), MPI_INT, mpi.comm);
            ss2.clear();
            for (int el : recv_buff)
                ss2.insert(el);
        }
        {
            int tmp = (int)ss3.size();
            MPI_Allgather(&tmp, 1, MPI_INT, sizes3.data(), 1, MPI_INT, mpi.comm);
            vector<int> send_buff(sizes3[mpi_rank]);
            int sum_sizes = 0;
            vector<int> displs(mpi_size);
            displs[0] = 0;
            for (int i=1; i<mpi_size; i++)
                displs[i] = (sum_sizes += sizes3[i-1]);
            sum_sizes += sizes3[mpi_size-1];
            vector<int> recv_buff(sum_sizes);
            multiset<int>::iterator itr=ss3.begin();
            for (int i=0; i<ss3.size(); i++)
                send_buff[i] = *(itr++);
            MPI_Allgatherv(send_buff.data(), (int)send_buff.size(), MPI_INT, recv_buff.data(),
                sizes3.data(), displs.data(), MPI_INT, mpi.comm);
            ss3.clear();
            for (int el : recv_buff)
                ss3.insert(el);
        }
        {
            int tmp = (int)sp.size();
            MPI_Allgather(&tmp, 1, MPI_INT, sizes4.data(), 1, MPI_INT, mpi.comm);
            vector<int> send_buff(sizes4[mpi_rank]);
            int sum_sizes = 0;
            vector<int> displs(mpi_size);
            displs[0] = 0;
            for (int i=1; i<mpi_size; i++)
                displs[i] = (sum_sizes += sizes4[i-1]);
            sum_sizes += sizes4[mpi_size-1];
            vector<int> recv_buff(sum_sizes);
            multiset<int>::iterator itr=sp.begin();
            for (int i=0; i<sp.size(); i++)
                send_buff[i] = *(itr++);
            MPI_Allgatherv(send_buff.data(), (int)send_buff.size(), MPI_INT, recv_buff.data(),
                sizes4.data(), displs.data(), MPI_INT, mpi.comm);
            sp.clear();
            for (int el : recv_buff)
                sp.insert(el);
        }
#endif // MLIP_MPI
        if (ss1 != sp)
            FAIL();
        if (ss2 != sp)
            FAIL();
        if (ss3 != sp)
            FAIL();
    }

    return true;
}

// "Lotf test"
bool utest_lotf()
{
    MTP mtp("MTP9prev.mtp");
    MTP mtp_4lotf("MTP9prev.mtp");

    Settings settings;
    settings["select:threshold"] = "1.1"; 
    settings["select:site_en_weight"] = "1"; 
    settings["select:energy_weight"] = "0"; 
    settings["select:force_weight"] = "0";  
    settings["select:stress_weight"] = "0"; 

    CfgSelection select(&mtp, settings.ExtractSubSettings("select"));
    CfgSelection select_4lotf(&mtp, settings.ExtractSubSettings("select"));

    //CfgSelection select(&mtp, 1.0e-7, 1.1, 1.0001, 1, 0, 0, 0);
    //CfgSelection select_4lotf(&mtp_4lotf, 1.0e-7, 1.1, 1.0001, 1, 0, 0, 0);

    Settings setup;
    setup["force_weight"] = to_string(0.01);
    LinearRegression lr(&mtp, setup);
    LinearRegression lr_4lotf(&mtp_4lotf, setup);

    VoidPotential vp;
    LOTF_linear lotf(&vp, &select_4lotf, &lr_4lotf, true);
    lotf.ignore_entry_count = true;

    ErrorMonitor errmon(setup);
    ErrorMonitor errmon_lotf(setup);

#ifdef MLIP_MPI
    auto cfgs = MPI_LoadCfgs("B-2.cfgs");
#else
    auto cfgs = LoadCfgs("B-2.cfgs");
#endif // MLIP_MPI

    int cntr = 0;
    for (auto& cfg : cfgs)
    {
        select.Select(cfg);
        vector<Configuration> cfgs_vec(select.selected_cfgs.begin(), select.selected_cfgs.end());
        lr.Train(cfgs_vec);
        Configuration cfg2(cfg);
        mtp.CalcEFS(cfg2);

        Configuration cfg3(cfg);
        lotf.CalcEFS(cfg3);

        //vector<Configuration> cfgs_vec2(lotf.p_selector->selected_cfgs.begin(), lotf.p_selector->selected_cfgs.end());
        if (select.selected_cfgs.size() != lotf.p_selector->selected_cfgs.size())
            FAIL("diff_size");

        errmon.Compare(cfg2, cfg3);

        if (fabs(errmon.epa_cfg.delta) > 1.0e-10)
            FAIL("epa_cfg " + to_string(errmon.epa_cfg.delta) + "  cntr=" + to_string(cntr));
        if (errmon.frc_cfg.sum.dltsq > 1.0e-10)
            FAIL("frc_cfg");
        if (errmon.str_cfg.dltsq > 1.0e-10)
            FAIL("str_cfg");

        if (++cntr>20)  // to accelerate test
            break;
    }

    return true;
}

// "MaxVol learning on the fly"
bool utest_mvslf() {
#ifdef MLIP_MPI
    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(mpi.comm, &mpi_rank);
    MPI_Comm_size(mpi.comm, &mpi_size);
#endif

    VoidPotential pot;
    //double MV_trshld = 1+1.0e-7;

    MTP mtp("mlips/unfitted.mtp");
    Settings setup;
    setup["energy_weight"] = to_string(500);
    setup["force_weight"] = to_string(1);
    setup["stress_weight"] = to_string(10);
    LinearRegression regr(&mtp, setup);

    Settings settings;
    settings["select:update_active_set"] = "TRUE";
    settings["select:threshold"] = "1.0000001"; 
    settings["select:site_en_weight"] = "0"; 
    settings["select:energy_weight"] = "1"; 
    settings["select:force_weight"] = "0";  
    settings["select:stress_weight"] = "0"; 

    CfgSelection al(&mtp, settings.ExtractSubSettings("select"));

    //CfgSelection al(&mtp, 1.0e-7, MV_trshld, MV_trshld, 0, 1, 0, 0);
    LOTF_linear MLIP(&pot, &al, &regr, true);

    std::ifstream ifs("Ti4MVtest.cfg", ios::binary);
    Configuration cfg;
    Configuration cfg4valid;
    ErrorMonitor err_mon(setup);

    for (int count = 0; count < 10; count++)
    {
        if (!cfg.Load(ifs))
            FAIL();
#ifdef MLIP_MPI
        if ( count % mpi_size != mpi_rank  )
            cfg.destroy();
#endif
        cfg.features["number"] = to_string(count);

        al.Select(cfg);
    }

    vector<Configuration> selcfg = vector<Configuration>(al.selected_cfgs.begin(), al.selected_cfgs.end());
    regr.Train(selcfg);

    for (int count = 10; cfg.Load(ifs); count++) 
    {
#ifdef MLIP_MPI
        if ( count % mpi_size != mpi_rank  )
            cfg.destroy();
#endif

        cfg.features["number"] = to_string(count);
        cfg4valid = cfg;
        MLIP.CalcEFS(cfg);
        err_mon.AddToCompare(cfg4valid, cfg);
    }

    ifs.close();

#ifdef GEN_TESTS
    {
        std::ofstream ofs("learn_on_the_fly.txt");
        ofs << err_mon.ene_aveabs() << '\n'
            << err_mon.frc_rmsrel() << '\n'
            << err_mon.str_rmsrel() << '\n';
        ofs.close();
    }
#endif

    ifs.open("learn_on_the_fly.txt");

    double ref;
    ifs >> ref;

    if (abs(err_mon.ene_aveabs() - ref) > 2e-5)
        FAIL()
        ifs >> ref;
    if (abs(err_mon.frc_rmsrel() - ref) > 2e-5)
        FAIL()
        ifs >> ref;
    if (abs(err_mon.str_rmsrel() - ref) > 2e-5)
        FAIL()
        ifs >> ref;

    ifs.close();

    return true;
}

// Selection weighting test
bool utest_wtgsel()
{
    const int ncut=15;
    const int nend=16;
    const char *cfgfile="TSB100.cfgs";
    const char *cfgselfile="temp/TSB100Sel.cfgs";
    // vibration
    {
        MTP mtp0("MTP9.mtp");

        Settings settings;
        settings["select:threshold"] = "1.0000001"; 
        settings["select:site_en_weight"] = "0"; 
        settings["select:energy_weight"] = "1"; 
        settings["select:force_weight"] = "1";  
        settings["select:stress_weight"] = "1"; 

        CfgSelection mv(&mtp0, settings.ExtractSubSettings("select"));

        //CfgSelection mv(&mtp0, 0.0000001, 1.0000001, 1.0000001, 0, 1, 1, 1);

#ifndef MLIP_MPI
        auto cfgs = LoadCfgs(cfgfile,ncut);
#else
        auto cfgs = MPI_LoadCfgs(cfgfile,ncut);
#endif

        mv.wgt_scale_power = 1;
        for (Configuration& cfg : cfgs)
            mv.AddForSelection(cfg);
        mv.Select();

        Configuration cfg1;
        Configuration cfg2;
        ifstream ifs(cfgfile, ios::binary);
        for (int i = 0; i < nend && cfg1.Load(ifs); i++) {
            if (i < ncut) continue;
            cfg2=cfg1;
            cfg2.ReplicateUnitCell(2,2,2);
        }

        CfgSampling smv(mv, settings);

        smv.MV_ene_cmpnts_weight=1.0;
        smv.MV_frc_cmpnts_weight=0.0;
        smv.MV_str_cmpnts_weight=0.0;

//         printf("\n%.10f %.10f %.15f\n",mv.Grade(cfg1),mv.Grade(cfg2),fabs(mv.Grade(cfg1) - mv.Grade(cfg2)));
        if(fabs(sqrt(8)*smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();
        smv.MV_ene_cmpnts_weight=0.0;
        smv.MV_frc_cmpnts_weight=1.0;
        smv.MV_str_cmpnts_weight=0.0;
        if(fabs(smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();
        smv.MV_ene_cmpnts_weight=0.0;
        smv.MV_frc_cmpnts_weight=0.0;
        smv.MV_str_cmpnts_weight=1.0;
        if(fabs(sqrt(8)*smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();

    }
    // structures
    {
        MTP mtp0("MTP9.mtp");
        Settings settings;
        settings["select:update_active_set"] = "TRUE";
        settings["select:threshold"] = "1.0000001"; 
        settings["select:site_en_weight"] = "0"; 
        settings["select:energy_weight"] = "1"; 
        settings["select:force_weight"] = "1";  
        settings["select:stress_weight"] = "1"; 
        CfgSelection mv(&mtp0, settings.ExtractSubSettings("select"));
        //CfgSelection mv(&mtp0, 0.0000001, 1.0000001, 1.0000001, 0, 1, 1, 1);

#ifndef MLIP_MPI
        auto cfgs = LoadCfgs(cfgfile,ncut);
#else
        auto cfgs = MPI_LoadCfgs(cfgfile,ncut);
#endif

        mv.wgt_scale_power = 2;
        for (Configuration& cfg : cfgs)
            mv.AddForSelection(cfg);
        mv.Select();

        Configuration cfg1;
        Configuration cfg2;
        ifstream ifs(cfgfile, ios::binary);
        for (int i = 0; i < nend && cfg1.Load(ifs); i++) {
            if (i < ncut) continue;
            cfg2=cfg1;
            cfg2.ReplicateUnitCell(2,2,2);
        }
        
        CfgSampling smv(mv, settings);

        smv.MV_ene_cmpnts_weight=1.0;
        smv.MV_frc_cmpnts_weight=0.0;
        smv.MV_str_cmpnts_weight=0.0;

//       printf("\n%.10f %.10f %.15f\n",mv.Grade(cfg1),mv.Grade(cfg2),fabs(mv.Grade(cfg1) - mv.Grade(cfg2)));
        if (fabs(smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();
        smv.MV_ene_cmpnts_weight=0.0;
        smv.MV_frc_cmpnts_weight=1.0;
        smv.MV_str_cmpnts_weight=0.0;
        if(fabs(smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();
        smv.MV_ene_cmpnts_weight=0.0;
        smv.MV_frc_cmpnts_weight=0.0;
        smv.MV_str_cmpnts_weight=1.0;
        if(fabs(smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();
    }

    // molecules
    {
        MTP mtp0("MTP9.mtp");
        Settings settings;
        settings["select:update_active_set"] = "TRUE";
        settings["select:threshold"] = "1.0000001"; 
        settings["select:site_en_weight"] = "0"; 
        settings["select:energy_weight"] = "1"; 
        settings["select:force_weight"] = "1";  
        settings["select:stress_weight"] = "1"; 
        CfgSelection mv(&mtp0, settings.ExtractSubSettings("select"));
        //CfgSelection mv(&mtp0, 0.0000001, 1.0000001, 1.0000001, 0, 1, 1, 1);

#ifndef MLIP_MPI
        auto cfgs = LoadCfgs(cfgfile,ncut);
#else
        auto cfgs = MPI_LoadCfgs(cfgfile,ncut);
#endif

        mv.wgt_scale_power = 0;
        for (Configuration& cfg : cfgs)
            mv.AddForSelection(cfg);
        mv.Select();

        Configuration cfg1;
        Configuration cfg2;
        ifstream ifs(cfgfile, ios::binary);
        for (int i = 0; i < nend && cfg1.Load(ifs); i++) {
            if (i < ncut) continue;
            cfg2=cfg1;
            cfg2.ReplicateUnitCell(2,2,2);
        }

        CfgSampling smv(mv, settings);

        smv.MV_ene_cmpnts_weight=1.0;
        smv.MV_frc_cmpnts_weight=0.0;
        smv.MV_str_cmpnts_weight=0.0;

//       printf("\n%.10f %.10f %.15f\n",mv.Grade(cfg1),mv.Grade(cfg2),fabs(mv.Grade(cfg1) - mv.Grade(cfg2)));
        if(fabs(8*smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();
        smv.MV_ene_cmpnts_weight=0.0;
        smv.MV_frc_cmpnts_weight=1.0;
        smv.MV_str_cmpnts_weight=0.0;
        if(fabs(smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();
        smv.MV_ene_cmpnts_weight=0.0;
        smv.MV_frc_cmpnts_weight=0.0;
        smv.MV_str_cmpnts_weight=1.0;
        if(fabs(8*smv.Grade(cfg1) - smv.Grade(cfg2)) > 1.0e-10 )
          FAIL();
    }
    return true;
}

// Local environments saving test
bool utest_locenv()
{
    { ofstream ofs("temp/selstate.mvs"); }
    { ofstream ofs("temp/savesampled.cfg"); }

    MLMTPR mtp("MTP9.mtp");
    
    Settings settings;
    settings["select"] = "TRUE";
    settings["select:site_en_weight"] = "1"; 
    settings["select:energy_weight"] = "0"; 

    CfgSelection sel(&mtp, settings.ExtractSubSettings("select"));

    sel.LoadSelected("B-1.cfgs");
//    sel.Save("temp/selstate.mvs");

    settings["extrapolation_control"] = "true";
    settings["extrapolation_control:threshold_save"] = "1.1";
    settings["extrapolation_control:save_extrapolative_to"] = "temp/savesampled.cfg";
    settings["extrapolation_control:add_grade_feature"] = "true";

    CfgSampling sam(sel, settings.ExtractSubSettings("extrapolation_control"));

    auto cfgs = MPI_LoadCfgs("B-2.cfgs");
    for (auto& cfg : cfgs)
        sam.Evaluate(cfg);

    ifstream ifs("temp/savesampled.cfg" + mpi.fnm_ending);
    if (ifs.is_open())
    {
        ifs.close();
        auto cfgs2 = LoadCfgs("temp/savesampled.cfg" + mpi.fnm_ending);
        for (auto& cfg : cfgs2)
            for (auto& cfgg : cfgs)
                if (cfgg.features["number"] == cfg.features["number"])
                {
                    if (cfgg.nbh_grades(0) - cfg.nbh_grades(0) >  1.0e-4 ||
                        cfgg.nbh_grades(cfgg.size()-1) - cfg.nbh_grades(cfg.size()-1)  >  1.0e-4)
                    {
                        cout << cfgg.features["number"] << " " << cfg.features["number"] << " " <<
                            cfgg.nbh_grades(0) <<  " " << cfg.nbh_grades(0);
                        FAIL();
                    }
                    break;
                }
    }

    return true;
}


