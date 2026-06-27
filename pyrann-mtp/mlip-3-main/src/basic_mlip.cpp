/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */


#include "basic_mlip.h"


using namespace std;


void AnyLocalMLIP::InitMagMomBasis(Configuration& cfg)
{
    //cfg_magnetic_basis_val.set(0);
    //cfg_magnetic_basis_der.set(0);
    if (p_MagneticBasis != nullptr && cfg.has_magmom()) {
        cfg.has_en_der_wrt_magmom(true);
        memset(&cfg.en_der_wrt_magmom(0, 0), 0, cfg.size() * sizeof(Vector3));
        cfg_magnetic_basis_val.resize((int)cfg.nbhs.size(), p_MagneticBasis->size);
        cfg_magnetic_basis_der.resize((int)cfg.nbhs.size(), p_MagneticBasis->size);
        for (int i = 0; i < cfg.nbhs.size(); i++) {
            p_MagneticBasis->Calc(cfg.magmom(i)[0]);
            p_MagneticBasis->CalcDers(cfg.magmom(i)[0]);
            for (int j = 0; j < p_MagneticBasis->size; j++) {
                cfg_magnetic_basis_val(i, j) = p_MagneticBasis->vals[j];
                cfg_magnetic_basis_der(i, j) = p_MagneticBasis->ders[j];
            }
        }
    }
    else {
        cfg_magnetic_basis_val.resize((int)cfg.nbhs.size(), 1);
        cfg_magnetic_basis_der.resize((int)cfg.nbhs.size(), 1);
        for (int i = 0; i < cfg.nbhs.size(); i++) {
            cfg_magnetic_basis_val(i, 0) = 1;
            cfg_magnetic_basis_der(i, 0) = 0;
        }
    }
}

void AnyLocalMLIP::Save(const std::string& filename)        //!< Saves MLIP's data to file
{
    Warning("AnyLocalMLIP: Save method is not implemented for this MLIP");
}

void AnyLocalMLIP::Load(const std::string& filename)        //!< Loads MLIP from file 
{
    Warning("AnyLocalMLIP: Load method is not implemented for this MLIP");
}

double AnyLocalMLIP::SiteEnergy(const Neighborhood & nbh)   //!< 
{
    CalcSiteEnergyDers(nbh);
    return buff_site_energy_;
}

// Must update buff_site_energy_ and buff_site_energy_ders_
void AnyLocalMLIP::CalcSiteEnergyDers(const Neighborhood& nbh)
{
    buff_site_energy_ = 0;
    FillWithZero(buff_site_energy_ders_);    
    AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0.0, nullptr);
}

//! Must calculate gradient of site energy w.r.t. coefficients and add it to second argument array
void AnyLocalMLIP::AddSiteEnergyGrad(const Neighborhood& nbh, 
                                     std::vector<double>& out_se_grad_accumulator) 
{
    AccumulateCombinationGrad(nbh, out_se_grad_accumulator, 1.0, nullptr);
}

// Calculate gradient of site energy w.r.t. coefficients and write them in second argument
void AnyLocalMLIP::CalcSiteEnergyGrad(const Neighborhood& nbh, std::vector<double>& out_se_grad)
{
    out_se_grad.resize(CoeffCount());
    FillWithZero(out_se_grad);

    AddSiteEnergyGrad(nbh, out_se_grad);
}

void AnyLocalMLIP::CalcForcesGrad(Configuration& cfg, Array3D& out_frc_grad)
{
    // calculating EFS components
    if (cfg.nbh_cutoff != CutOff())
        cfg.InitNbhs(CutOff());

    out_frc_grad.resize((int)cfg.size(), 3, CoeffCount());
    out_frc_grad.set(0);

    InitMagMomBasis(cfg);

    std::vector<Vector3> dir;

    for (Neighborhood& nbh : cfg.nbhs)
    {
        dir.resize(nbh.count);

        for (int j=0; j<nbh.count; j++)
            for (int a=0; a<3; a++)
            {
                FillWithZero(dir);
                dir[j][a] = 1;
                FillWithZero(tmp_grad_accumulator_);
                AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0, &dir[0]);
                for (int k=0; k<CoeffCount(); k++)
                {
                    out_frc_grad(nbh.my_ind, a, k) += tmp_grad_accumulator_[k];
                    out_frc_grad(nbh.inds[j], a, k) -= tmp_grad_accumulator_[k];
                }
            }
    }

    // Not optimal implementation from the performance point of view
    // it is caused by LAMMPS limitation on the array size for reverse communication.
    // now this limit is set to comm_reverse=3 (i.e. array size limit is 3*cfg.size())
    // However we need in communication of the array with comm_reverse=3*CoeffCount() (i.e. array size limit is 3*cfg.size()*CoeffCount())
    // It is nessessary to think and study how to _properly_ avoid such inefficient way
    if (mpi.LammpsCallbackComm != nullptr)
    {
        Array2D tmp((int)cfg.nbhs.size(), 3);

        for (int k=0; k<CoeffCount(); k++)
        {
            for (int i=0; i<cfg.size(); i++)
                for (int a=0; a<3; a++)
                    tmp(i, a) = out_frc_grad(i, a, k);

            mpi.LammpsCallbackComm(cfg.p_void_pair, (cfg.size()==0) ? nullptr : &tmp(0, 0));

            for (int i=0; i<cfg.size(); i++)
                for (int a=0; a<3; a++)
                    out_frc_grad(i, a, k) = tmp(i, a);
        }
    }
}

void AnyLocalMLIP::CalcStressesGrads(Configuration & cfg, Array3D & out_str_grad)
{
    out_str_grad.resize(3, 3, CoeffCount());
    out_str_grad.set(0);

    // calculating EFS components
    if (cfg.nbh_cutoff != CutOff())
        cfg.InitNbhs(CutOff());

    InitMagMomBasis(cfg);

    vector<Vector3> dir;

    for (Neighborhood& nbh : cfg.nbhs)
    {
        dir.resize(nbh.count);

        for (int j = 0; j < nbh.count; j++)
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                {
                    FillWithZero(dir);
                    dir[j][a] = 1;
                    FillWithZero(tmp_grad_accumulator_);
                    AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0, &dir[0]);

                    for (int k=0; k<CoeffCount(); k++)
                        out_str_grad(a, b, k) -= tmp_grad_accumulator_[k] * nbh.vecs[j][b];
                }
    }
}

void AnyLocalMLIP::CalcEFSGrads(Configuration & cfg, 
                                Array1D & out_ene_grad, 
                                Array3D & out_frc_grad, 
                                Array3D & out_str_grad)
{
    // calculating EFS components
    if (cfg.nbh_cutoff != CutOff())
        cfg.InitNbhs(CutOff());

    out_ene_grad.resize(CoeffCount());
    FillWithZero(out_ene_grad);
    out_frc_grad.resize((int)cfg.size(), 3, CoeffCount());
    out_frc_grad.set(0);
    out_str_grad.resize(3, 3, CoeffCount());
    out_str_grad.set(0);

    vector<Vector3> dir;

    InitMagMomBasis(cfg);

    for (Neighborhood& nbh : cfg.nbhs)
    {
        dir.resize(nbh.count);
        AccumulateCombinationGrad(nbh, out_ene_grad, 1, nullptr);
        
        for (int j=0; j<nbh.count; j++)
        {
            for (int a=0; a<3; a++)
            {
                FillWithZero(dir);
                dir[j][a] = 1;
                FillWithZero(tmp_grad_accumulator_);
                AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0, &dir[0]);
                for (int k=0; k<CoeffCount(); k++)
                {
                    out_frc_grad(nbh.my_ind, a, k) += tmp_grad_accumulator_[k];
                    out_frc_grad(nbh.inds[j], a, k) -= tmp_grad_accumulator_[k];
                    
                    for (int b=0; b<3; b++)
                        out_str_grad(a, b, k) -= tmp_grad_accumulator_[k] * nbh.vecs[j][b];
                }
            }
        }
    }

    // Not optimal implementation from the performance point of view
    // it is caused by LAMMPS limitation on the array size for reverse communication.
    // now this limit is set to comm_reverse=3 (i.e. array size limit is 3*cfg.size())
    // However we need in communication of the array with comm_reverse=3*CoeffCount() (i.e. array size limit is 3*cfg.size()*CoeffCount())
    // It is nessessary to think and study how to _properly_ avoid such inefficient way
    if (mpi.LammpsCallbackComm != nullptr)
    {
        Array2D tmp((int)cfg.size(), 3);

        for (int k=0; k<CoeffCount(); k++)
        {
            for (int i=0; i<cfg.size(); i++)
                for (int a=0; a<3; a++)
                    tmp(i, a) = out_frc_grad(i, a, k);

            mpi.LammpsCallbackComm(cfg.p_void_pair, (cfg.size()==0) ? nullptr : &tmp(0, 0));

            for (int i=0; i<cfg.size(); i++)
                for (int a=0; a<3; a++)
                    out_frc_grad(i, a, k) = tmp(i, a);
        }
    }
}

//! Function implementing soft constraints on coefficients (e.g., so that the norm of the radial function is 1) or regularization implementing as penalty to training procedure
void AnyLocalMLIP::AddPenaltyGrad(  const double coeff, 
                                    double & out_penalty_accumulator, 
                                    Array1D * out_penalty_grad_accumulator) 
{
}

AnyLocalMLIP::AnyLocalMLIP(double _min_val, double _max_val, int _radial_basis_size)
{
    p_RadialBasis = new Basis_Shapeev(_min_val, _max_val, _radial_basis_size);
    //p_RadialBasis = new RB_Chebyshev(_min_val, _max_val, _radial_basis_size);
    //p_RadialBasis = new Basis_Chebyshev(_min_val, _max_val, _radial_basis_size);
    rb_owner = true;
}

AnyLocalMLIP::AnyLocalMLIP(AnyBasis* p_radial_basis, AnyBasis* p_magnetic_basis)
{
    p_RadialBasis = p_radial_basis;
    p_MagneticBasis = p_magnetic_basis;
    rb_owner = false;
}

AnyLocalMLIP::~AnyLocalMLIP()
{
    if (rb_owner && p_RadialBasis != nullptr)
        delete p_RadialBasis;
    if (rb_owner && p_MagneticBasis != nullptr)
        delete p_MagneticBasis;
}

void AnyLocalMLIP::CalcE(Configuration& cfg)
{
    cfg.energy = 0;

    // calculating EFS components
    if (cfg.nbh_cutoff != CutOff())
        cfg.InitNbhs(CutOff());

    InitMagMomBasis(cfg);

    for (const Neighborhood& nbh : cfg.nbhs)
        cfg.energy += SiteEnergy(nbh);

    cfg.has_energy(true);
}

void AnyLocalMLIP::CalcEFS(Configuration& cfg)
{
    ResetEFS(cfg);

    if (cfg.nbh_cutoff != CutOff())
        cfg.InitNbhs(CutOff());

    InitMagMomBasis(cfg);

    // calculating EFS components
    for (Neighborhood& nbh : cfg.nbhs)
    {
        buff_site_energy_ = 0;
        buff_site_energy_ders_.resize(nbh.count);
        CalcSiteEnergyDers(nbh);

        if (cfg.has_site_energies())
            cfg.site_energy(nbh.my_ind) = buff_site_energy_;
        cfg.energy += buff_site_energy_;

        for (int j = 0; j < nbh.count; j++) {
            cfg.force(nbh.my_ind) += buff_site_energy_ders_[j];
            cfg.force(nbh.inds[j]) -= buff_site_energy_ders_[j];
            if (cfg.has_en_der_wrt_magmom()) 
                cfg.en_der_wrt_magmom(nbh.inds[j])[0] += buff_se_ders_wrt_magmom_neigh_[j];
        }
        if (cfg.has_en_der_wrt_magmom()) 
            cfg.en_der_wrt_magmom(nbh.my_ind)[0] += buff_se_ders_wrt_magmom_central_;

        for (int j = 0; j < nbh.count; j++)
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    cfg.stresses[a][b] -= buff_site_energy_ders_[j][a] * nbh.vecs[j][b];
    }

    // Transfer en_der_wrt_magmom from ghost to original atoms (LAMMPS may do it second time)
    if (mpi.LammpsCallbackComm != nullptr && cfg.has_en_der_wrt_magmom())
    {
        double* comm_arr = (cfg.has_en_der_wrt_magmom() ? &cfg.en_der_wrt_magmom(0, 0) : nullptr);
        mpi.LammpsCallbackComm(cfg.p_void_pair, comm_arr);

        // make data on ghost atoms equal zero to prevent errors after second communication that may be attemped by LAMMPS
        for (int i=0; i<cfg.size(); i++)
            if (cfg.ghost_inds.count(i) != 0)
            {
                cfg.en_der_wrt_magmom(i, 0) = 0.0;
                cfg.en_der_wrt_magmom(i, 1) = 0.0;
                cfg.en_der_wrt_magmom(i, 2) = 0.0;
            }
    }
}

// Calculate gradient of energy w.r.t. coefficients and write them in second argument
void AnyLocalMLIP::CalcEnergyGrad(Configuration & cfg, std::vector<double>& out_energy_grad)
{
    out_energy_grad.resize(CoeffCount());
    FillWithZero(out_energy_grad);

    if (cfg.nbh_cutoff != CutOff())
        cfg.InitNbhs(CutOff());

    InitMagMomBasis(cfg);

    for (const Neighborhood& nbh : cfg.nbhs)
        AddSiteEnergyGrad(nbh, out_energy_grad);
}

void AnyLocalMLIP::AccumulateEFSCombinationGrad(Configuration &cfg,
                                                double ene_weight,
                                                const std::vector<Vector3>& frc_weights,
                                                const Matrix3& str_weights,
                                                std::vector<double>& out_grads_accumulator)
{
    out_grads_accumulator.resize(CoeffCount());

    if (cfg.nbh_cutoff != CutOff())
        cfg.InitNbhs(CutOff());

    /*    InitMagMomBasis(cfg); */

    for (Neighborhood& nbh : cfg.nbhs)
    {
        tmp_se_ders_weights_.resize(nbh.count);
        FillWithZero(tmp_se_ders_weights_);

        for (int j = 0; j < nbh.count; j++) {
            tmp_se_ders_weights_[j] += frc_weights[nbh.my_ind];
            tmp_se_ders_weights_[j] -= frc_weights[nbh.inds[j]];
        }

        for (int j = 0; j < nbh.count; j++)
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    tmp_se_ders_weights_[j][a] += str_weights[a][b] * nbh.vecs[j][b];

        bool grad_zero = true;
        for (int i=0; i<3*tmp_se_ders_weights_.size(); i++)
            if (tmp_se_ders_weights_[i/3][i%3] != 0.0)
            {
                grad_zero = false;
                break;
            }

        if (grad_zero)
        {
            AccumulateCombinationGrad(  nbh,
                                        out_grads_accumulator,
                                        ene_weight,
                                        nullptr);
        }
        else
        {
            AccumulateCombinationGrad(  nbh, 
                                        out_grads_accumulator, 
                                        ene_weight, 
                                        &tmp_se_ders_weights_[0]);
        }
    }
}
