/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "basic_potentials.h"

void AnyPotential::ResetEFS(Configuration & cfg) //!< Declaring and cleaning EFS
{
    cfg.has_energy(true);
    cfg.has_forces(true);
    cfg.has_stresses(true);
    cfg.energy = 0.0;
    memset(&cfg.force(0, 0), 0, cfg.size() * sizeof(Vector3));
    memset(&cfg.stresses[0][0], 0, sizeof(Matrix3));
}

// Cheks equality f_i = -df/dr_i by finite differences. Returns true if deviation between FD-calculated and exact force is less than control_delta, 

bool AnyPotential::CheckEFSConsistency_debug(Configuration cfg, double displacement, double control_delta)
{
    Configuration cfg1, cfg2;
    double err_max = 0.0;
    double err_max_force = 0.0;
    double err_max_magmom = 0.0;

    cfg.ClearNbhs();
    CalcEFS(cfg);

    //std::cout << "forces: " << std::endl;
    // f_i = -dE/dr_i check
    for (int i = 0; i < cfg.size(); i++)
    {
        Vector3 fd_force;
        for (int a=0; a<3; a++)
        {
            cfg1 = cfg;
            cfg2 = cfg;
            cfg1.ClearNbhs();
            cfg2.ClearNbhs();
            cfg1.pos(i)[a] += displacement;
            cfg2.pos(i)[a] -= displacement;
            CalcEFS(cfg1);
            CalcEFS(cfg2);
            fd_force[a] = (cfg2.energy - cfg1.energy) / (2 * displacement);
            if (fabs(fd_force[a] - cfg.force(i)[a]) > control_delta)
                std::cout << fd_force[a] << " " << cfg.force(i)[a] << " " << fabs(fd_force[a] - cfg.force(i)[a]) << " " << i << " " << a << std::endl; 
        }
        double err = (cfg.force(i)-fd_force).Norm();
        err_max_force = __max(err, err_max_force);
    }

    //cout << err_max_force << endl;
    //if (err_max_force > control_delta)
    //  return false;

    if (cfg.has_magmom()) {
        //std::cout << "energy derivatives w.r.t. magnetic moments: " << std::endl;
        for (int i = 0; i < cfg.size(); i++)
        {
            double fd_magmom;
            cfg1 = cfg;
            cfg2 = cfg;
            cfg1.ClearNbhs();
            cfg2.ClearNbhs();
            cfg1.magmom(i)[0] += displacement;
            cfg2.magmom(i)[0] -= displacement;
            CalcEFS(cfg1);
            CalcEFS(cfg2);
            fd_magmom = (cfg1.energy - cfg2.energy) / (2 * displacement);
            if (fabs(fd_magmom - cfg.en_der_wrt_magmom(i)[0]) > control_delta)
                std::cout << fd_magmom << " " << cfg.en_der_wrt_magmom(i)[0] << " " << fabs(fd_magmom - cfg.en_der_wrt_magmom(i)[0]) << " " << i << std::endl; 
            double err = fabs(cfg.en_der_wrt_magmom(i)[0]-fd_magmom);
            err_max_magmom = __max(err, err_max_magmom);
        }
    }

    // L^{-T} * stresses = dE/dL check
    //std::cout << "stresses: " << std::endl;
    Matrix3 dEdL = -cfg.lattice.inverse().transpose() * cfg.stresses;
    Matrix3 fd_dEdL(0,0,0,0,0,0,0,0,0);
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++)
        {
            cfg1 = cfg;
            cfg2 = cfg;

            Matrix3 deform = cfg.lattice;

            deform[a][b] += displacement;
            cfg1.Deform(cfg1.lattice.inverse()*deform);
            cfg1.ClearNbhs();

            deform[a][b] -= 2*displacement;
            cfg2.Deform(cfg2.lattice.inverse()*deform);
            cfg2.ClearNbhs();

            CalcEFS(cfg1);
            CalcEFS(cfg2);

            fd_dEdL[a][b] = (cfg1.energy-cfg2.energy) / (cfg1.lattice[a][b]-cfg2.lattice[a][b]);
            //std::cout << dEdL[a][b] << " " << fd_dEdL[a][b] << std::endl;
        }
    }
    
    if (cfg.has_magmom()) err_max = __max(sqrt((fd_dEdL - dEdL).NormFrobeniusSq()), __max(err_max_force, err_max_magmom));
    else err_max = __max(sqrt((fd_dEdL - dEdL).NormFrobeniusSq()), err_max_force);

    return (err_max < control_delta);
}
