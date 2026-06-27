/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin, Alexander Shapeev, Ivan Novikov
 */

#include "relaxation.h"
#include "../pair_potentials.h"
#include <sstream>


using namespace std;


const char* Relaxation::tagname = {"relax"};

void Relaxation::WriteLog()
{
    std::stringstream logstrm1;
    logstrm1    << "cfg " << struct_cntr
                << "\titr " << step
                << "\t dx " << max_step
                << "\t md " << curr_mindist
                << "\te/a " << cfg.energy/cfg.size()
                << "\tent " << f
                << "\tfrc " << max_force
                << "\tstr " << max_stress
//                << "\tmagmom " << max_en_der_magmom
                << std::endl;
    MLP_LOG(tagname,logstrm1.str());
};

std::string Relaxation::MindistStr(std::set<int>& typeset, std::map<std::pair<int, int>, double>& md_map)
{
    string outstr;

    for (int i : typeset)
    {
        for (int j : typeset)
            if (j >= i)
                outstr += "md(" + to_string(i) + ',' + to_string(j) + ")=" +
                          to_string(md_map[make_pair(i, j)]) + ";\t";
    }
    outstr = outstr.substr(0, outstr.length() - 2);

    return outstr;
}

void Relaxation::GetFixedAtoms(std::string strind)
{
    while (!strind.empty())
    {
        size_t commapos = strind.find(',');
        if (commapos == -1)
        {
            fixed_atoms.insert(stoi(strind));
            return;
        }
        int ind = stoi(strind.substr(0, commapos));
        fixed_atoms.insert(ind);
        strind = strind.substr(commapos+1);
    }
}

void Relaxation::EFStoFuncGrad()
{
    double vol = fabs(cfg.lattice.det());

    f = cfg.energy + (pressure/eVA3_to_GPa) * vol;

    Matrix3 Stress = cfg.stresses - Matrix3::Id() * (vol * pressure / eVA3_to_GPa);
    max_stress = Stress.MaxAbs() * eVA3_to_GPa / vol;
    if (relax_cell_flag)
    {
        Matrix3 dEdL = cfg.lattice.inverse().transpose() * Stress;
        for (int a=0; a<3; a++)
            for (int b=0; b<3; b++)
                g[3*a+b] = -dEdL[a][b];
    }

    max_force = 0.0;
    if (relax_atoms_flag)
    {
        for (int i=0; i<cfg.size(); i++)
        {
            Vector3 frac_force = cfg.lattice * cfg.force(i);
            g[offset+3*i+0] = -frac_force[0];
            g[offset+3*i+1] = -frac_force[1];
            g[offset+3*i+2] = -frac_force[2];
            max_force = __max(max_force, cfg.force(i).Norm());
        }

        for (int ind : fixed_atoms)
        {
            g[offset+3*ind+0] = 0.0;
            g[offset+3*ind+1] = 0.0;
            g[offset+3*ind+2] = 0.0;
        }
    }
    else // for correct output in log
        for (int i=0; i<cfg.size(); i++)
            max_force = __max(max_force, cfg.force(i).Norm());

    max_en_der_magmom = 0.0;
    if (relax_magmoms_flag)
    {
        for (int i=0; i<cfg.size(); i++)
        {
            //cout << i+1 << " " << cfg.en_der_wrt_magmom(i)[0] << " " << cfg.magmom(i)[0] << endl;
            g[offset+offset2+i] = cfg.en_der_wrt_magmom(i)[0];
            max_en_der_magmom = __max(max_en_der_magmom, cfg.en_der_wrt_magmom(i)[0]);
        }
    }
    /*else // for correct output in log
        for (int i=0; i<cfg.size(); i++)
            max_en_der_magmom = __max(max_en_der_magmom, cfg.en_der_wrt_magmom(i)[0]);*/
}

void Relaxation::XtoLatPosMagmom()
{
    max_step = 0.0;

    if (relax_cell_flag)
    {
        if (!freeze_small_grads || max_stress > tol_stress || bfgs.is_in_linesearch())
        {
            // Calculation of the new lattice in Cartesian coordinates
            Matrix3 dfrm = cfg.lattice.inverse();
            dfrm *= Matrix3(x[0], x[1], x[2],
                            x[3], x[4], x[5],
                            x[6], x[7], x[8]);
            Matrix3 new_lattice(cfg.lattice*dfrm);

            if (relax_atoms_flag)
                cfg.lattice = new_lattice;
            else
                cfg.Deform(dfrm);

            // max_step calculation
            Vector3 delta_l1(x[0]-x_prev[0], x[1]-x_prev[1], x[2]-x_prev[2]);
            Vector3 delta_l2(x[3]-x_prev[3], x[4]-x_prev[4], x[5]-x_prev[5]);
            Vector3 delta_l3(x[6]-x_prev[6], x[7]-x_prev[7], x[8]-x_prev[8]);
            max_step = __max(__max(delta_l1.Norm(), delta_l2.Norm()), delta_l3.Norm());
        }
        else if (freeze_small_grads && !bfgs.is_in_linesearch()) // && (max_stress > tol_stress)
        {
            for (int a=0; a<3; a++)
                for (int b=0; b<3; b++)
                    x[3*a+b] = cfg.lattice[a][b];
            bfgs.Set_x(x);
        }
    }

    if (relax_atoms_flag)
    {
        // Calculation of the new atomic positions in Cartesian coordinates
        Matrix3 LT(cfg.lattice.transpose());
        if (freeze_small_grads && !bfgs.is_in_linesearch())
        {
            for (int i=0; i<cfg.size(); i++)                // small forces are not relaxed 
            {
                Vector3 frac_coord(x[offset+3*i+0], x[offset+3*i+1], x[offset+3*i+2]);
                Vector3 period(floor(frac_coord[0]), floor(frac_coord[1]), floor(frac_coord[2]));
                frac_coord -= period;
                if (!freeze_small_grads || cfg.force(i).Norm()>tol_force)        // move atom if force is large
                    cfg.pos(i) = LT*frac_coord;
                else                                        // keep atom position and fix x vector  
                {
                    x[offset+3*i+0] = x_prev[offset+3*i+0];
                    x[offset+3*i+1] = x_prev[offset+3*i+1];
                    x[offset+3*i+2] = x_prev[offset+3*i+2];
                }
            }

            bfgs.Set_x(x);                                  // update bfgs.x with not moved atoms
        }
        else
            for (int i=0; i<cfg.size(); i++)
            {
                Vector3 frac_coord(x[offset+3*i+0], x[offset+3*i+1], x[offset+3*i+2]);
                Vector3 period(floor(frac_coord[0]), floor(frac_coord[1]), floor(frac_coord[2]));
                frac_coord -= period;
                cfg.pos(i) = LT*frac_coord;
            }

        // max_step calculation
        for (int i=0; i<cfg.size(); i++)
        {
            Vector3 delta_x(x[offset+3*i+0] - x_prev[offset+3*i+0],
                            x[offset+3*i+1] - x_prev[offset+3*i+1],
                            x[offset+3*i+2] - x_prev[offset+3*i+2]);
            max_step = __max(max_step, (delta_x * cfg.lattice).Norm());
        }
    }

    if (relax_magmoms_flag) {
        for (int i=0; i<cfg.size(); i++) {
            cfg.magmom(i)[0] = x[offset+offset2+i];
            max_step = __max(max_step, fabs(x[offset+offset2+i]-x_prev[offset+offset2+i]));
        }
    }
}

void Relaxation::LatPosMagmomToX()
{
    if (relax_cell_flag)
    {
        //Lattice relaxation in Cartesian coordinates
        for (int a=0; a<3; a++)
            for (int b=0; b<3; b++)
                x[3*a+b] = cfg.lattice[a][b];
    }
    if (relax_atoms_flag)
    {
        Matrix3 invLT(cfg.lattice.inverse().transpose());
        for (int i=0; i<cfg.size(); i++)
        {
            Vector3 frac_coord = cfg.direct_pos(i);
            x[offset+3*i+0] = frac_coord[0];
            x[offset+3*i+1] = frac_coord[1];
            x[offset+3*i+2] = frac_coord[2];
        }
    }
    if (relax_magmoms_flag)
    {
        for (int i=0; i<cfg.size(); i++)
            x[offset+offset2+i] = cfg.magmom(i)[0];
    }
}

void Relaxation::Init()
{
    // setting flags
    relax_atoms_flag = (tol_force > 0);
    relax_cell_flag = (tol_stress > 0);
    relax_magmoms_flag = (tol_en_der_magmom > 0);
    if (!relax_atoms_flag && !relax_cell_flag && !relax_magmoms_flag)
        ERROR("No relaxation activated");
    offset = relax_cell_flag ? 9 : 0;

    cfg.ClearNbhs();
    // Checking and repairing initial configuration
    if (correct_cell)
        cfg.CorrectSupercell();
    else        
        cfg.MoveAtomsIntoCell();
    
    offset2 = relax_atoms_flag ? 3*cfg.size() : 0;
    // initalization of BFGS argument array 'x'
    int new_size = offset + offset2 + (relax_magmoms_flag ? cfg.size() : 0);
    x.resize(new_size);
    x_prev.resize(new_size);
    g.resize(new_size);
    LatPosMagmomToX();
    bfgs.Set_x(x.data(), new_size);
    bfgs.Restart();                     // Cleaning Hessian
    max_step = 0.0;

    // setting the initial Hessian
    const Matrix3 LT = cfg.lattice * cfg.lattice.transpose();
    // setting force constants;
    if (relax_atoms_flag) 
    {
        const Matrix3 force_const_inv = (1.0 / forces_elasticity_scale) * LT.inverse();
        for (int i=0; i<cfg.size(); i++) 
            for (int a=0; a<3; a++)
                for (int b=0; b<3; b++)
                    bfgs.inv_hess(offset + 3*i + a, offset + 3*i + b) = force_const_inv[a][b];
    }
    // setting the elastic constants
    if (relax_cell_flag) 
    {
        const Matrix3 el_const_inv = (cfg.lattice.transpose() * cfg.lattice) *
                                     (eVA3_to_GPa / stress_elasticity_scale / fabs(cfg.lattice.det()));
        for (int i=0; i<3; i++) 
            for (int a=0; a<3; a++)
                for (int b=0; b<3; b++)
                    bfgs.inv_hess(3*i + a, 3*i + b) = el_const_inv[a][b];
    }
    bfgs.no_Hessian_update = use_GD;
}

void Relaxation::CorrectMinDist()
{
    Init();

    curr_mindist = cfg.MinDist();
    //cfg.features["mindist"] = to_string(curr_mindist);

    repulsive_pot.power = 1000;
    
    auto typeset = cfg.GetTypes();
    
    if (init_mindist == -1)
    {
        auto md_map = cfg.GetTypeMindists();
        double delta = HUGE_DOUBLE;
        for (int i : typeset)
            for (int j : typeset)
                delta = __min(delta, md_map[make_pair(i,j)] - (mindist_cores[i]+mindist_cores[j]));

        if (delta >= 0)
            return;

        for (int type : typeset)
            if (repulsive_pot.cutoffs.find(type) == repulsive_pot.cutoffs.end())
                ERROR("mindist core for type " + to_string(type) + " is not specified");
    }
    else
    {
        if (curr_mindist >= init_mindist)
            return;

        auto type_set = cfg.GetTypes();
        for (int type : type_set)
            repulsive_pot.cutoffs[type] = init_mindist + maxstep_constraint;
    }

    {
        std::stringstream logstrm1("Relaxation: Correcting minimal distance in cfg#" +
            to_string(struct_cntr) + "\n");
        MLP_LOG(tagname, logstrm1.str());
        logstrm1.str("");
    }

    bfgs.no_Hessian_update = true;

    double press_bckp = pressure;
    pressure = 0;

    vector<Vector3> fixed_pos_backup;
    for (int ind : fixed_atoms)
        fixed_pos_backup.push_back(cfg.pos(ind));

    // Run main loop
    curr_mindist = cfg.MinDist();
    for (step=0; step<9999; step++)
    {
        cfg.ClearNbhs();
        repulsive_pot.CalcEFS(cfg);
        EFStoFuncGrad();

        auto md_map = cfg.GetTypeMindists();
        curr_mindist = cfg.MinDist(); // TODO: Remove

        if (init_mindist != -1 && curr_mindist > init_mindist)
        {
            std::stringstream logstrm1;
            logstrm1 << "Relaxation: Mindist for cfg#" << struct_cntr << " fixed\n";
            MLP_LOG(tagname, logstrm1.str());

            break;
        }
        else if (init_mindist == -1)
        {
            double delta = HUGE_DOUBLE;
            for (int i : typeset)
                for (int j : typeset)
                    delta = __min(delta, md_map[make_pair(i, j)] - (mindist_cores[i]+mindist_cores[j]));

            if (delta >= 0)
            {
                std::stringstream logstrm1;
                logstrm1 << "Relaxation: Mindist for cfg#" << struct_cntr << " fixed\n";
                MLP_LOG(tagname, logstrm1.str());

                break;
            }
        }

        {
            std::stringstream logstrm1;
            logstrm1    << "cfg " << struct_cntr
                << "\tstep " << step
                << "\tdx " << max_step
                << "\t" << MindistStr(typeset, md_map)
                //                << "\tmagmom " << max_en_der_magmom
                << std::endl;
            MLP_LOG(tagname, logstrm1.str());
        }

        // backuping previous x
        memcpy(x_prev.data(), x.data(), x.size()*sizeof(double));

        // Do BFGS iteration
        try { bfgs.Iterate(f, g); x = bfgs.x(); }
        catch (const MlipException& excep) {
            Warning("Relaxation: BFGS crashed. Error message:\n" + excep.message);
            bfgs.Restart();
            LatPosMagmomToX();
            bfgs.Set_x(x.data(), (int)x.size());
        }

        // updating lattice and atomic positions
        XtoLatPosMagmom();

        while (max_step > maxstep_constraint)
        {
            x = bfgs.ReduceStep(0.5);
            XtoLatPosMagmom();
        }

        if (max_step < minstep_constraint)
        {
            bfgs.Set_x(x);  // Reseting linesearch
            XtoLatPosMagmom();
        }

        if (!fixed_atoms.empty())
        {
            int ind2 = 0;
            for (int ind : fixed_atoms)
                cfg.pos(ind) = fixed_pos_backup[ind2++];

            bfgs.Set_x(x);  // Reseting linesearch
            XtoLatPosMagmom();
        }
    }

    pressure  = press_bckp;

    if (correct_cell)
        cfg.CorrectSupercell();
    cfg.has_energy(false);
    cfg.has_forces(false);
    cfg.has_stresses(false);
    cfg.has_site_energies(false);
    //cfg.features["mindist"] = to_string(cfg.MinDist());
}

Relaxation::Relaxation( AnyPotential* _pot,
                        const Configuration& _cfg,
                        double _tol_force,
                        double _tol_stress, 
                        double _tol_en_der_magmom)
{
    p_potential = _pot;
    cfg = _cfg;
    tol_force = _tol_force;
    tol_stress = _tol_stress;
    tol_en_der_magmom = _tol_en_der_magmom;

//  SetLogStream(log_output);
}

Relaxation::Relaxation(const Settings& settings, Wrapper* _p_mlip, AnyPotential* _p_potential)
{
    InitSettings();
    ApplySettings(settings);
    
    Message("Using internal relaxation driver");

    p_mlip = _p_mlip;
    p_potential = _p_potential;

    if (!log_output.empty() &&
        log_output != "stdout" &&
        log_output != "stderr")
        log_output += mpi.fnm_ending;

    if ( mpi.rank == 0 )
    PrintSettings();

    bfgs.wolfe_c1 = bfgs_wolfe_c1;
    bfgs.wolfe_c2 = bfgs_wolfe_c2;

    if (minstep_constraint >= maxstep_constraint)
        ERROR("Incorrect setings: min_step > max_step");
//  SetLogStream(log_output);

    if (!mindist_string.empty())
    {
        if (mindist_string[0] == '{')
        {
            try
            {
                mindist_string = mindist_string.substr(1);
                while (!mindist_string.empty())
                {
                    int endpos = (int)mindist_string.find(':', 0);
                    int spec = stoi(mindist_string.substr(0, endpos));
                    mindist_string = mindist_string.substr(endpos+1);
                    if ((endpos = (int)mindist_string.find(',', 0)) != -1)
                    {
                        double rad = stod(mindist_string.substr(0, endpos));
                        mindist_string = mindist_string.substr(endpos+1);
                        mindist_cores[spec] = rad;
                    }
                    else if ((endpos = (int)mindist_string.find('}', 0)) != -1)
                    {
                        double rad = stod(mindist_string.substr(0, endpos));
                        mindist_cores[spec] = rad;
                        break;
                    }
                    else
                        ERROR("Incorrect setting for\'init_mindist\'");
                }

                for (auto& core : mindist_cores)
                {
                    repulsive_pot.cutoffs[core.first] = core.second + maxstep_constraint;
                    repulsive_pot.max_cutoff = __max(repulsive_pot.max_cutoff, repulsive_pot.cutoffs[core.first]);
                }

                init_mindist = -1;
            }
            catch (...) { ERROR("unrecognized format of \'init_mindis\'t setting"); }
        }
        else
        {
            try { init_mindist = stod(mindist_string); }
            catch (...) { ERROR("unrecognized format of \'init_mindis\'t setting"); }
        }
    }

    SetTagLogStream(tagname, log_output);

    Message("Relaxation initialization complete");
}

Relaxation::~Relaxation()
{
}

bool Relaxation::CheckConvergence()
{
    return  (cfg.energy < tol_energy) || 
            ((max_force < tol_force || !relax_atoms_flag) &&
             (max_stress < tol_stress || !relax_cell_flag) &&
             (max_en_der_magmom < tol_en_der_magmom || !relax_magmoms_flag));
}

bool Relaxation::FixStep()
{
    while (max_step > maxstep_constraint)
    {
        x = bfgs.ReduceStep(0.5);
        if (max_step < minstep_constraint)
            return false;
    }

    return true;
}

void Relaxation::Run()
{
    if (cfg.features.count("relax:breaking_energy") > 0)
        tol_energy = stod(cfg.features["relax:breaking_energy"]);
    else
        tol_energy = -9.9e99;

    fixed_atoms.clear();
    if (cfg.features.count("relax:fixed_atoms") > 0)
    {
        GetFixedAtoms(cfg.features["relax:fixed_atoms"]);
        correct_cell = false;
    }

    std::stringstream logstrm1;

    struct_cntr++;

    if (correct_cell)
        cfg.CorrectSupercell();

    CorrectMinDist();
    curr_mindist = cfg.MinDist();
    if (curr_mindist < mindist_constraint)
        ERROR("Relaxation: Minimal distance in the intial configuration is too short. " +
            to_string(curr_mindist));

    if (cfg.size()>0 && itr_limit>0)
    {
        logstrm1 << "Relaxation: starting relaxation cfg#" << to_string(struct_cntr) << '\n';
        MLP_LOG(tagname, logstrm1.str());
        logstrm1.str("");
        cfg.ClearNbhs();
        Init();
        if (pressure != 0.0)
            cfg.features["external_pressure,[GPa]"] = to_string(pressure);
    }
    else
        return; // e.g. correction mindist scenario

    bool skip_ls_reset = true;  // required for first iterataion and to prevent line search reset twice
    // Run main loop
    for (step=0; step<itr_limit; step++)
    {
        cfg.ClearNbhs();
        CalcEFS(cfg);
        EFStoFuncGrad();

        WriteLog();

        //  check convergence
        if (CheckConvergence())
            break;

        // backuping previous x
        memcpy(x_prev.data(), x.data(), x.size()*sizeof(double));
        
        // Do BFGS iteration
        try { bfgs.Iterate(f, g); x = bfgs.x(); }
        catch (const MlipException& excep) {
            Warning("Relaxation: BFGS crashed. Error message:\n" + excep.message );
            bfgs.Restart();
            LatPosMagmomToX();
            bfgs.Set_x(x.data(), (int)x.size());
        }

        // updating lattice and atomic positions
        XtoLatPosMagmom();

        if (max_step < minstep_constraint)
        {
            if (!skip_ls_reset)
            {
                logstrm1 << "Relaxation: Too small steps detected. Breaking linesearch\n";
                MLP_LOG(tagname, logstrm1.str());
                logstrm1.str("");
                bfgs.Set_x(x);  // Reseting linesearch
                XtoLatPosMagmom();
            }
            skip_ls_reset = !skip_ls_reset;
        }

        while (max_step > maxstep_constraint)
        {
            x = bfgs.ReduceStep(0.5);
            XtoLatPosMagmom();
        }

        curr_mindist = cfg.MinDist();
        //cfg.features["mindist"] = to_string(curr_mindist);
    }

    if (correct_cell)
        cfg.CorrectSupercell();

    cfg.GetTypeMindists();

    if (p_mlip == nullptr && p_potential == nullptr) // mindist fix scenario
    {
        cfg.has_energy(false);
        cfg.has_site_energies(false);
        cfg.has_stresses(false);
        cfg.has_forces(false);

        cfg.features["from"] = "fix_mindist_OK";
        return;
    }
 
    cfg.ClearNbhs();
    CalcEFS(cfg);

    //  check convergence
    if (CheckConvergence())
    {
        if (curr_mindist >= mindist_constraint)
        {
            cfg.features["from"] = "relaxation_OK";
            if (pressure != 0.0)
            {
                double PV = pressure * fabs(cfg.lattice.det()) / eVA3_to_GPa;
                cfg.features["PV"] = to_string(PV);
                cfg.features["enthalpy"] = to_string(cfg.energy + PV);
            }
            logstrm1 << "Relaxation of cfg#" + to_string(struct_cntr) + " completed successfully\n\n";
            MLP_LOG(tagname, logstrm1.str());
            logstrm1.str("");
            return;
        }
        else
        {
            cfg.features["from"] = "relaxation_MINDIST_ERROR";
            logstrm1 << "Relaxation of cfg#" + to_string(struct_cntr) + " completed with mindist error\n\n";
            MLP_LOG(tagname, logstrm1.str());
            logstrm1.str("");
            throw MlipException("Final configuration does not satisfy mindist constraint");
        }
    }
    else // not converged
    {
        if (curr_mindist >= mindist_constraint)
        {
            cfg.features["from"] = "relaxation_NOT_CONVERGED";
            if (pressure != 0.0)
            {
                double PV = pressure * fabs(cfg.lattice.det()) / eVA3_to_GPa;
                cfg.features["PV"] = to_string(PV);
                cfg.features["enthalpy"] = to_string(cfg.energy + PV);
            }
            logstrm1 << "Relaxation of cfg#" + to_string(struct_cntr) + " convergence has not been achieved\n\n";
            MLP_LOG(tagname, logstrm1.str());
            logstrm1.str("");
        }
        else
        {
            cfg.features["from"] = "relaxation_MINDIST_ERROR";
            logstrm1 << "Relaxation of cfg#" + to_string(struct_cntr) + " failed\n";
            MLP_LOG(tagname, logstrm1.str());
            logstrm1.str("");
            throw MlipException("Relaxation failed");
        }
    }
}

void Relaxation::PrintDebug(const char * fnm)   // for testing purposes
{
    std::ofstream ofs(fnm, std::ios::app);
    cfg.MoveAtomsIntoCell();
    cfg.ClearNbhs();
    for (int i = 0; i < cfg.size(); i++)
        ofs << cfg.pos(i, 0) << ' ' << cfg.pos(i, 1) << ' ' << cfg.pos(i, 2) << '\t';
    ofs << std::endl;
    ofs.close();
}
