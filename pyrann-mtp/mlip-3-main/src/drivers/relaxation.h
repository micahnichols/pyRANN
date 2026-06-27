/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin, Alexander Shapeev, Ivan Novikov
 */

#ifndef MLIP_RELAXATION_H
#define MLIP_RELAXATION_H


#include "basic_drivers.h"
#include "../common/bfgs.h"
#include "../pair_potentials.h"


class Relaxation : public AnyDriver, protected InitBySettings
{
private:
    int offset = 0;     // indicates initial position of atom coordinates in array 'x' (AWA forces in array 'grad'). In case of relaxation of cell and atoms ofset =6, otherwise =0
    int offset2 = 0;    // indicates initial position of magnetic moments in array 'x' (AWA energy derivatives w.r.t. magnetic moments in array 'grad'). In case of relaxation of cell and magmoms set =6, otherwise =0
    std::vector<double> x_prev;

    std::map<int, double> mindist_cores;

    std::set<int> fixed_atoms;

    std::string MindistStr(std::set<int>& typeset, std::map<std::pair<int, int>, double>& md_map);
    void GetFixedAtoms(std::string strind);

    void InitSettings()
    {
        MakeSetting(pressure,               "pressure");
        MakeSetting(tol_stress,             "stress_tolerance");
        MakeSetting(tol_force,              "force_tolerance");
        MakeSetting(freeze_small_grads,     "freeze_small_grads");
//        MakeSetting(tol_en_der_magmom,      "energy_der_wrt_magmom");
        MakeSetting(maxstep_constraint,     "max_step");
        MakeSetting(minstep_constraint,     "min_step");
        MakeSetting(mindist_constraint,     "mindist");
        MakeSetting(bfgs_wolfe_c1,          "bfgs_wolfe_c1");
        MakeSetting(bfgs_wolfe_c2,          "bfgs_wolfe_c2");
        MakeSetting(itr_limit,              "iteration_limit");
        MakeSetting(use_GD,                 "use_gd_method");
        MakeSetting(correct_cell,           "correct_cell");
        MakeSetting(mindist_string,         "init_mindist");
        MakeSetting(forces_elasticity_scale,"forces_elasticity_scale");
        MakeSetting(stress_elasticity_scale,"stress_elasticity_scale");
        MakeSetting(log_output,             "log");
    };

protected:
    static const char* tagname;             // tag name of object
    bool relax_atoms_flag;                  // indicator of atom relaxation
    bool relax_magmoms_flag = false;        // indicator of magmom relaxation
    bool relax_cell_flag;                   // indicator of cell relaxation

    Array1D x;                              // argument-vector holder for BFGS
    Array1D g;                              // gradient-vector holder for BFGS
    double f;                               // function-value holder for BFGS

    // temporal variables
    double curr_mindist;                    // current minimal distance
    double max_force;                       // maximal force magnitude on the current iteration
    double max_en_der_magmom;               // maximal energy derivative w.r.t. magnetic moment magnitude on the current iteration
    double max_stress;                      // maximal deviation of stress tensor from target (maximal stress, if external pressure is 0) on the current iteration
    double max_step;                        // maximal change in atoms displacement and/or lattice vectors displacements on the last iteration
    
    const double eVA3_to_GPa = 160.21766208;

    RepulsivePotential repulsive_pot;

    void EFStoFuncGrad();                   // transfers of EFS from configuration to f, g (function and gradient)
    void XtoLatPosMagmom();                 // transfers from x-vector to lattice vectors, atomic positions and magnetic moments. Also calculates max_atom_travel
    void LatPosMagmomToX();                 // transfers lattice vectors, atomic positions and magnetic moments to x-vector
    bool CheckConvergence();                // returns true if forces and stresses are less than tolerance
    bool FixStep();                         // trying to reduce step to achieve agreement with maxstep_constraint and mindist_constraint constraints

    void Init();                            // initiates relaxation process. Called by Run() before iterations begun
    void CorrectMinDist();                  // corrects configuration using GD method and repulsive pair potentail. Minimal distance in the corracted cfg is (mindist_constraint + maxstep_constraint)

    void WriteLog();
    
public:
    BFGS bfgs;                              // BFGS minimization algorithm
    Configuration cfg;                      // configuration being relaxed
    
    // Settings
    double pressure = 0.0;                  // external pressure in GPa
    double tol_force = 1e-4;                // maximal force tolerance for convergence. If <= 0 then fractional coordinates are held (no atoms relaxation)
    double tol_stress = 1e-3;               // maximal stress tolerance for convergence, in GPa. If <= 0 then lattice vectors are held (no supercell relaxation)
    double tol_en_der_magmom = 0;           // maximal energy derivative w.r.t. magnetic moment tolerance for convergence                 
    double tol_energy = -999999999;         // to_string(HUGE_DOUBLE) looks ugly on the screen 
    double maxstep_constraint = 0.05;       // maximal distance atoms and lattice vectors are allowed to travel in one iteration (in Angstrom)
    double minstep_constraint = 1.0e-8;     // minimal distance atoms and lattice vectors are allowed to travel in one iteration (in Angstrom)
    double mindist_constraint = 0.5;        // in angstroms. Required for prevention of MTP extrapolations
    int itr_limit = HUGE_INT;               // iteration limit
    bool use_GD = false;                    // Gradient descent instead of BFGS
    bool correct_cell = false;              // rotate cell to make the lattice matrix lower triangular, correct skewness
    double init_mindist = 0.0;              // if minimal distance in the initial configuration is less than tis value a repulsive potential will be applied to correct it. The corrected minimal distance will be equal this value
    double forces_elasticity_scale = 10;    // typical force constant in eV/A^2
    double stress_elasticity_scale = 500;   // typical (slightly overestimated) elast. constant in GPa
    bool freeze_small_grads = false;        // if true, freeze the degrees of freedom which the correspondent gradients are less than tolerance
    std::string log_output = "";            // log
    double bfgs_wolfe_c1 = 0.1;
    double bfgs_wolfe_c2 = 0.8;
    std::string mindist_string = "";

    int struct_cntr = 0;                    // Relaxed structures counter
    int step = 0;                           // EFS calcs counter for a given structure
    
    Relaxation( AnyPotential* _p_potential, 
                const Configuration& _cfg, 
                double _tol_force = 1e-4,   // If _tol_force <= 0 fractional coordinates are hold(no atoms relaxation)
                double _tol_stress = 1e-3,  // If _tol_stress <= 0 lattice vectors are hold (no cupercell relaxation)
                double _tol_en_der_magmom = 0); // If _tol_en_der_magmom <= 0 magnetic moments are hold (no magnetic moments relaxation)
    Relaxation( const Settings& _settings, 
                Wrapper* _p_mlip=nullptr, 
                AnyPotential* _p_potential=nullptr);
    ~Relaxation();

    void Run() override;                    // Run relaxation process. 

    void PrintDebug(const char* fnm);       // Prints configuration to file in debug purposes
};

#endif //#ifndef MLIP_RELAXATION_H
