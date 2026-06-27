#include "../common/comm.h"
#include "../common/utils.h"
#include <chrono>

using namespace std;


#ifdef MLIP_NO_SELFTEST
bool RunAllTests(bool is_parallel) { return true; }
#else

// 0
// "Configurations reading"
bool utest_cr();
// "Binary and normal configurations reading"
bool utest_bnr();
// "Loading configuration database to a vector" 
bool utest_ldv();
// "Binary configurations file format and old format r/w"
bool utest_rw();
// "New format configurations reading/writing test" 
bool utest_cfgrw();
// "Configuration with ghost atoms reading/writing test" 
bool utest_cfgrwgst();
// "Identical configurations reading test" 
bool utest_icrt();
//"Testing Configuration periodic extension"
bool utest_pe();
// "Testing reading configuration with fractional coordinates"
bool utest_frac();
// "Testing Configuration::features"
bool utest_ftrs();
// "VASP OUTCAR iterations count"
bool utest_vaspr();
// "BLAS_test"
bool utest_blas();
// "Errorneus configurations reading test"
bool utest_errr();
// "Construction of neighborhoods for non-periodic (i.e. with ghost atoms) configuration" 
bool utest_gstnbh();
// "MTP file format v1.1.0 test" 
bool utest_mtp();
// "MTP file reading 1"
bool utest_mtpfr1();
// "MTP file reading 2"
bool utest_mtpfr2();
// "MTP file reading 3" 
bool utest_mtpfr3();
// "MTP learning" 
bool utest_mlpl();

// 1
// "Linesearch"
bool utest_ls();
// "Checks minimal distance in configurations"
bool utest_cmd();
// "Supercell rotation test"
bool utest_sr();
// "Relaxation driver test"
bool utest_rd();
// "Robustness of relaxation with changing potential"
bool utest_rr();
// "repulsion in relaxation"
bool utest_reprel();
// "Another MTP learning + ErrorMonitor"
bool utest_mtpl();
// "MTP fitting on insufficient data"
bool utest_mtpf();
// "Check radial basis derivatives with finite differences")
bool utest_rbd();
// Linear Regression weighting test
bool utest_lrw0();

// 2
// "MaxVol initialization and selection")
bool utest_mv();
// "MaxVol evaluation"
bool utest_mve();
// "Configurations motion while MaxVol selection"
bool utest_mvcm();
// "Maxvol selection unique cfgs and swap_limit tests"
bool utest_mvsl();
//  "Multiple selection test#1"
bool utest_ms1();
//  "Multiple selection test#2"
bool utest_ms2();
// "MaxVol Saving/Loading selection test#1"
bool utest_mvsl1();
// "Parallel selection test"
bool utest_prlsel();
// "Lotf test"
bool utest_lotf();
// "MaxVol learning on the fly"
bool utest_mvslf();
// Selection weighting test 
bool utest_wtgsel();
// Local environments saving test
bool utest_locenv();

// 3
// "MLIP_wrapper test1 (Recording cfgs, EFS via abinitio)"
bool utest_mw1();
// "MLIP_wrapper test2 (EFS via MTP, Recording cfgs, Error Monitoring)"
bool utest_mw2();
// "MLIP_wrapper test3 (Learning, Recording cfgs, EFS from ab-initio)"
bool utest_mw3();
// "MLIP_wrapper test4 (Learning + Selection, EFS from ab-initio)"
bool utest_mw4();
// "MLIP_wrapper test5 (Loading selection + Selection)"
bool utest_mw5();
// "MLIP_wrapper test6 (Breaking treshold test)"
bool utest_mw6();
// "MLIP_wrapper test7 (LOTF with files)"
bool utest_mw7();
//// "MLIP_wrapper test8 (Loading selection + LOTF in memory, track changes)"
//bool utest_mw8();

// 4
// "EAMSimple CalcEnergyGrad test by finite difference"
bool utest_eamsceg();
// "EAMSimple CalcEFS check with finite differences"
bool utest_eamscefs();
// "EAMSimple AddLossGrad test by finite difference"
bool utest_eamsalg();
// "EAMMult CalcEFS check with finite differences"
bool utest_eammefs();
// "EAMMult AddLossGrad test by finite difference"
bool utest_eammalg();
// "Stillinger-Weber CalcEFS check with finite differences"
bool utest_swcefs();
// "Stillinger-Weber CalcEFSvsCalcEFSGrad EFS comparison"
bool utest_swcefsc();
// "Stillinger-Weber CalcEFSGrad check with central differences"
bool utest_swcefscd();
// "Stillinger-Weber CalcEnergyGradvsCalcEFSGrad energy grad comparison"
bool utest_swcefsgc();
// "Lennard-Jones CalcEFS check with finite differences"
bool utest_ljcefs();
// "Lennard-Jones AddLossGrad test by finite difference"
bool utest_ljagl();
// "NonlinearLennard-Jones CalcEFS check with finite differences"
bool utest_nlljcefs();
// "NonlinearLennard-Jones AddLossGrad test by finite difference"
bool utest_nlljalg();
// "PairPot AddLossGrad test by finite difference"
bool utest_ppalg();
// "MTPR CalcEFS (check forces correctness by using of central differences)"
bool utest_mtprcefs();
// "MTPR CalcEFSGrads (check grads correctness by using of central differences)"
bool utest_mtprcefsgrads();
// "MTPR CalcEFSGrads2 (Check energies, forces and stresses gradients using central differences)"
bool utest_mtprcefsgrads2();
// Non-Linear Regression weighting test
bool utest_nlrw0();

// 5
// "Training PairPotentialNonlinear"
bool utest_ppnl();
// "EAMSimple fitting test"
bool utest_eams1();
// "Lennard-Jones fitting test"
bool utest_ljp1();
// "NonlinearLennard-Jones fitting test"
bool utest_nlljp1();
// "MTPR calc EFS with absolute/relative atom numeration test"
bool utest_mtprp21();
// "MTPR species_count extension test"
bool utest_mtprp11();
// "MTPR fitting test"
bool utest_mtprp1();
// "EAMMult fitting test"
bool utest_eamm1();
// "MTPR loss gradient test"
bool utest_mtprp2();


bool run_test(const char* pName, bool (*ptest)())
{
    std::cout.flush();
    MPI_Barrier(mpi.comm);

    static int test_num = 1;
    string run_mess  = "\tTest "            + to_string(test_num) + "\t...    ";
    string pass_mess  = "\r\tTest "         + to_string(test_num) + "\tpassed ";
    string fail_mess  = "\r!!! --> Test "   + to_string(test_num) + "\tfailed ";
    if (0 == mpi.rank)
    {
        std::cout << run_mess << pName;
        std::cout.flush();
    }
    
    if( (*ptest)() ) 
    { 
        if ( 0 == mpi.rank )
            std::cout << pass_mess << pName << endl;
        test_num++;
        return true;
    }
    else 
    {
        std::cout << fail_mess << pName << endl;
        test_num++;
        return false;
    }
}

bool RunAllTests(bool is_parallel)
{
    bool result = true;

    try {
          // 0
          result &= run_test("Configurations reading",&utest_cr);
          result &= run_test("Binary and normal configurations reading",&utest_bnr);
          result &= run_test("Loading configuration database to a vector",&utest_ldv);
          result &= run_test("Binary configurations file format and old format r/w",&utest_rw);
          result &= run_test("New format configurations reading/writing test",&utest_cfgrw);
          result &= run_test("Configuration with ghost atoms reading/writing test", &utest_cfgrwgst);
          result &= run_test("Identical configurations reading test",&utest_icrt);
          result &= run_test("Testing Configuration periodic extension",&utest_pe);
          result &= run_test("Testing reading configuration with fractional coordinates",&utest_frac);
          result &= run_test("Testing Configuration::features",&utest_ftrs);
          result &= run_test("VASP OUTCAR iterations count",&utest_vaspr);
          result &= run_test("BLAS_test",&utest_blas);
          result &= run_test("Errorneus configurations reading test",&utest_errr);
          result &= run_test("Construction of neighborhoods for non-periodic (i.e. with ghost atoms) configuration",&utest_gstnbh);
          result &= run_test("MTP file format v1.1.0 test",&utest_mtp);
          result &= run_test("MTP file reading 1",&utest_mtpfr1);
          result &= run_test("MTP file reading 2",&utest_mtpfr2);
          result &= run_test("MTP file reading 3",&utest_mtpfr3);
          result &= run_test("MTP learning",&utest_mlpl);
          // 1
          result &= run_test("Linesearch",&utest_ls);
          result &= run_test("Checks minimal distance in configurations",&utest_cmd);
          result &= run_test("Supercell rotation test",&utest_sr);
          result &= run_test("Relaxation driver test",&utest_rd);
          result &= run_test("Robustness of relaxation with changing potential",&utest_rr);
          result &= run_test("Repulsion in relaxation", &utest_reprel);
          result &= run_test("Another MTP learning + ErrorMonitor",&utest_mtpl);
          result &= run_test("MTP fitting on insufficient data",&utest_mtpf);
          result &= run_test("Check radial basis derivatives with finite differences",&utest_rbd);
          result &= run_test("Linear Regression weighting test",&utest_lrw0);
          // 2
          result &= run_test("MaxVol initialization and selection",&utest_mv);
          result &= run_test("MaxVol evaluation",&utest_mve);
          result &= run_test("Configurations motion while MaxVol selection",&utest_mvcm);
          result &= run_test("Maxvol selection unique cfgs and swap_limit tests",&utest_mvsl);
          result &= run_test("Multiple selection test#1",&utest_ms1);
          result &= run_test("Multiple selection test#2",&utest_ms2);
          result &= run_test("MaxVol Saving/Loading selection test#1",&utest_mvsl1);
          result &= run_test("Parallel selection test", &utest_prlsel);
          result &= run_test("MaxVol learning on the fly",&utest_mvslf);
          result &= run_test("Lotf test", &utest_lotf);
          result &= run_test("Selection weighting test", &utest_wtgsel);
          result &= run_test("Local environments saving test", &utest_locenv);
          // 3
          result &= run_test("MLIP_wrapper test1 (Recording cfgs, EFS via abinitio)",&utest_mw1);
          result &= run_test("MLIP_wrapper test2 (EFS via MTP, Recording cfgs, Error Monitoring)",&utest_mw2);
          result &= run_test("MLIP_wrapper test3 (Learning, Recording cfgs)",&utest_mw3);
          result &= run_test("MLIP_wrapper test4 (Selection, EFS from ab-initio)",&utest_mw4);
          result &= run_test("MLIP_wrapper test5 (Loading selection + Sampling)",&utest_mw5);
          result &= run_test("MLIP_wrapper test6 (Breaking treshold test)", &utest_mw6);
          result &= run_test("MLIP_wrapper test7 (LOTF with file writing)", &utest_mw7);
//          result &= run_test("MLIP_wrapper test8 (Loading selection + LOTF through memory, track errors)", &utest_mw8);
          // 4
          result &= run_test("EAMSimple CalcEnergyGrad test by finite difference",&utest_eamsceg);
          result &= run_test("EAMSimple CalcEFS check with finite differences",&utest_eamscefs);
          result &= run_test("EAMSimple AddLossGrad test by finite difference",&utest_eamsalg);
          result &= run_test("EAMMult CalcEFS check with finite differences",&utest_eammefs);
          result &= run_test("EAMMult AddLossGrad test by finite difference",&utest_eammalg);
          result &= run_test("Stillinger-Weber CalcEFS check with finite differences",&utest_swcefs);
          result &= run_test("Stillinger-Weber CalcEFSvsCalcEFSGrad EFS comparison",&utest_swcefsc);
          result &= run_test("Stillinger-Weber CalcEFSGrad check with central differences",&utest_swcefscd);
          result &= run_test("Stillinger-Weber CalcEnergyGradvsCalcEFSGrad energy grad comparison",&utest_swcefsgc);
          result &= run_test("Lennard-Jones CalcEFS check with finite differences",&utest_ljcefs);
          result &= run_test("Lennard-Jones AddLossGrad test by finite difference",&utest_ljagl);
          result &= run_test("NonlinearLennard-Jones CalcEFS check with finite differences",&utest_nlljcefs);
          result &= run_test("NonlinearLennard-Jones AddLossGrad test by finite difference",&utest_nlljalg);
          result &= run_test("PairPot AddLossGrad test by finite difference",&utest_ppalg);
          result &= run_test("MTPR CalcEFS (check forces correctness by using of central differences)",&utest_mtprcefs);
          result &= run_test("MTPR CalcEFSGrads (Compare gradients calculated by potential and pointer)",&utest_mtprcefsgrads);
          result &= run_test("MTPR CalcEFSGrads2 (Check energies, forces and stresses gradients using central differences)",&utest_mtprcefsgrads2);
          result &= run_test("Non-Linear Regression weighting test",&utest_nlrw0);
          // 5
          result &= run_test("Training PairPotentialNonlinear",&utest_ppnl);
          result &= run_test("EAMSimple fitting test",&utest_eams1);
          result &= run_test("Lennard-Jones fitting test",&utest_ljp1);
          result &= run_test("NonlinearLennard-Jones fitting test",&utest_nlljp1);  
          result &= run_test("MTPR calc EFS with absolute/relative atom numeration test", &utest_mtprp21);
          result &= run_test("MTPR species_count extension test", &utest_mtprp11);
          result &= run_test("MTPR fitting test",&utest_mtprp1);
          result &= run_test("MTPR loss gradient test",&utest_mtprp2);
          result &= run_test("EAMMult fitting test",&utest_eamm1);

    }
    catch(const MlipException& e) { 
        printf("\nERROR: MLIP Exception caught: %s\n",(char*)e.What());
        return false; 
    }
    return result;
}

#endif // MLIP_NO_SELFTEST




