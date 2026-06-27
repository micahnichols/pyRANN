/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Ivan Novikov
 */


#ifndef MLIP_SamplePotentials
#define MLIP_SamplePotentials

#include "basic_mlip.h"

//! This is a sample class for a pair potential
//! It is a linear potential, but it pretends it is not
//! (and therefore is fitted with nonlinear optimizer like bfgs)
//! Uses the AnyLocalMLIP class, allowing to just define the site potentials
class PairPotentialNonlinear : public AnyLocalMLIP
{
protected:
    std::vector<double> regression_coeffs;

protected:

    void AddSiteEnergyGrad( const Neighborhood& neighborhood, 
                            std::vector<double>& site_energy_grad) override {}

    void CalcSiteEnergyDers(const Neighborhood& nbh) override
    {
        buff_site_energy_ = 0.0;
        buff_site_energy_ders_.resize(nbh.count);
        FillWithZero(buff_site_energy_ders_);

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            double dist_j = nbh.dists[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(nbh.dists[j]);
            for (int i = 0; i < p_RadialBasis->size; i++) {
                double val = p_RadialBasis->vals[i];
                double der = p_RadialBasis->ders[i];

                // pair potential part
                buff_site_energy_ += val * regression_coeffs[i];
                buff_site_energy_ders_[j] += vec_j * ((der / dist_j) * regression_coeffs[i]);
            }
        }
    }

    void AccumulateCombinationGrad(const Neighborhood& nbh,
        std::vector<double>& out_grad_accumulator,
        const double se_weight = 0.0,
        const Vector3* se_ders_weights = nullptr) override
    {
        out_grad_accumulator.resize(CoeffCount());
        buff_site_energy_ders_.resize(nbh.count);

        FillWithZero(buff_site_energy_ders_);
        buff_site_energy_ = 0.0;

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            double dist_j = nbh.dists[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(nbh.dists[j]);
            for (int i = 0; i < p_RadialBasis->size; i++) {
                double val = p_RadialBasis->vals[i];
                double der = p_RadialBasis->ders[i];

                // pair potential part
                buff_site_energy_ += val * regression_coeffs[i];
                out_grad_accumulator[i] += se_weight * val;
                buff_site_energy_ders_[j] += vec_j * ((der / dist_j) * regression_coeffs[i]);
                if (se_ders_weights != nullptr)
                    out_grad_accumulator[i] += vec_j.dot(se_ders_weights[j]) * (der / dist_j);
            }
        }
    }

public:
    PairPotentialNonlinear(int _radial_basis_count, double _min_val = 1.6, double _max_val = 5.0)
        : AnyLocalMLIP(_min_val, _max_val, _radial_basis_count)
//      , radial_basis_count(_radial_basis_count)
        , regression_coeffs(_radial_basis_count)
    {
    }

    // this may disapear in future:
    int CoeffCount() //!< number of coefficients
    {
        return (int)regression_coeffs.size();
    }
    double* Coeff() //!< coefficients themselves
    {
        return &regression_coeffs[0];
    }
};

class CoupledPotential : public AnyPotential
{
public:
    AnyPotential* first=nullptr;
    AnyPotential* second=nullptr;

    CoupledPotential() : AnyPotential() {};

    void CalcEFS(Configuration& cfg)
    {
        Configuration cfg_tmp(cfg);
        first->CalcEFS(cfg);
        second->CalcEFS(cfg_tmp);
        cfg.energy += cfg_tmp.energy;
        cfg.stresses += cfg_tmp.stresses;
        for (int i=0; i<cfg.size(); i++)
            cfg.force(i) += cfg_tmp.force(i);
    }
};


//! This is a single-component EAM potential
//! Uses the AnyLocalMLIP class, allowing to just define the site potentials
class EAMSimple : public AnyLocalMLIP
{
protected:
    std::vector<double> regression_coeffs;

public:
    int radial_basis_count_; //!< number of radial basis functions used
    int embedding_parameters_count_; //!< number of terms in f(rho) = c_0 + c_2 rho^2 + c rho^3 + ...
    //!< note that c_1 rho is missing intentionally

    EAMSimple(int _radial_basis_count, int embedding_function_parameters_count,
        double _min_val = 1.6, double _max_val = 5.0)
        : AnyLocalMLIP(_min_val, _max_val, _radial_basis_count)
        , radial_basis_count_(_radial_basis_count)
        , embedding_parameters_count_(embedding_function_parameters_count)
        , regression_coeffs(2 * _radial_basis_count + embedding_function_parameters_count)
    {}

    // this may disapear in future:
    int CoeffCount() //!< number of coefficients
    {
        return (int)regression_coeffs.size();
    }
    double* Coeff() //!< coefficients themselves
    {
        return &regression_coeffs[0];
    }

protected:
    void AddPenaltyGrad(const double coeff, double& out_penalty, Array1D* out_penalty_grad) override
    {
        const double* density_coeffs = &regression_coeffs[radial_basis_count_];

        double sum_coeffs2 = 0;
        for (int i = 0; i < radial_basis_count_; i++) 
            sum_coeffs2 += density_coeffs[i]*density_coeffs[i];

        out_penalty += coeff * (sum_coeffs2-1) * (sum_coeffs2-1);

        if (out_penalty_grad != nullptr)
            for (int i = radial_basis_count_; i < 2 * radial_basis_count_; i++) 
                (*out_penalty_grad)[i] += coeff * 4 * (sum_coeffs2 - 1) * density_coeffs[i - radial_basis_count_];
    }

    void AddSiteEnergyGrad(const Neighborhood& neighborhood, 
        std::vector<double>& site_energy_grad) override
    {
        //const double* pair_potentials_coeffs = &regression_coeffs[0];
        const double* density_coeffs = &regression_coeffs[radial_basis_count_];
        const double* embedding_coeffs = &regression_coeffs[2 * radial_basis_count_];

        //std::cout << radial_basis_count_ << std::endl;    

        double density = 0;
        //site_energy_grad.resize(2 * radial_basis_count_ + embedding_parameters_count_);
        //FillWithZero(site_energy_grad);

        for (int j = 0; j < neighborhood.count; j++) {
            const double dist_j = neighborhood.dists[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(dist_j);

            // density calculation
            for (int i = 0; i < radial_basis_count_; i++) {
                const double val = p_RadialBasis->vals[i];
                density += val * density_coeffs[i];
                site_energy_grad[i] += 0.5 * val;
            }
        }

        // embedding function and its derivative WRT density
        double density_pow = density; // will be density^m
        //double embedding = embedding_coeffs[0];
        double embedding_der = 0;
        site_energy_grad[2*radial_basis_count_] += 1;
        for (int m = 2; m < embedding_parameters_count_+1; m++) {
            embedding_der += m * embedding_coeffs[m-1] * density_pow;
            density_pow *= density;
            site_energy_grad[m-1+2*radial_basis_count_] += density_pow;
        }

        for (int j = 0; j < neighborhood.count; j++) {
            const double dist_j = neighborhood.dists[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(dist_j);

            for (int i = radial_basis_count_; i < 2 * radial_basis_count_; i++) {
                const double val = p_RadialBasis->vals[i - radial_basis_count_];
                site_energy_grad[i] += embedding_der * val;
            }
        }
    }

    void CalcSiteEnergyDers(const Neighborhood& nbh) override
    {
        buff_site_energy_ = 0.0;
        buff_site_energy_ders_.resize(nbh.count);
        FillWithZero(buff_site_energy_ders_);

        const double* pair_potentials_coeffs = &regression_coeffs[0];
        const double* density_coeffs = &regression_coeffs[radial_basis_count_];
        const double* embedding_coeffs = &regression_coeffs[2 * radial_basis_count_];

        //std::cout << radial_basis_count_ << std::endl;    

        double density = 0;

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(nbh.dists[j]);

            // pair potential part
            for (int i = 0; i < radial_basis_count_; i++) {
                const double val = p_RadialBasis->vals[i];
                const double der = p_RadialBasis->ders[i];
                buff_site_energy_ += 0.5 * val * pair_potentials_coeffs[i];
                buff_site_energy_ders_[j] += 0.5 * vec_j * ((der / dist_j) * pair_potentials_coeffs[i]);
            }

            // density calculation
            for (int i = 0; i < radial_basis_count_; i++) {
                const double val = p_RadialBasis->vals[i];
                density += val * density_coeffs[i];
            }
        }

        // embedding function and its derivative WRT density
        double density_pow = density; // will be density^m
        double embedding = embedding_coeffs[0];
        double embedding_der = 0;
        for (int m = 2; m < embedding_parameters_count_ + 1; m++) {
            embedding_der += m * embedding_coeffs[m-1] * density_pow;
            density_pow *= density;
            embedding += embedding_coeffs[m-1] * density_pow;
        }

        buff_site_energy_ += embedding;

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(nbh.dists[j]);
            // now we can compute the derivative of the embedding WRT atom positions
            for (int i = 0; i < radial_basis_count_; i++) {
                const double der = p_RadialBasis->ders[i];
                buff_site_energy_ders_[j] += vec_j * ((der / dist_j) * density_coeffs[i]) * embedding_der;
            }
        }
    }

    void AccumulateCombinationGrad(const Neighborhood& nbh,
        std::vector<double>& out_grad_accumulator,
        const double se_weight = 0.0,
        const Vector3* se_ders_weights = nullptr) override
    {
        out_grad_accumulator.resize(CoeffCount());
        buff_site_energy_ders_.resize(nbh.count);

        FillWithZero(buff_site_energy_ders_);
        buff_site_energy_ = 0.0;

        const double* pair_potentials_coeffs = &regression_coeffs[0];
        const double* density_coeffs = &regression_coeffs[radial_basis_count_];
        const double* embedding_coeffs = &regression_coeffs[2 * radial_basis_count_];

        double density = 0;
        std::vector<double> density_grad(radial_basis_count_);
        FillWithZero(density_grad);

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(nbh.dists[j]);

            // pair potential part
            for (int i = 0; i < radial_basis_count_; i++) {
                const double val = p_RadialBasis->vals[i];
                const double der = p_RadialBasis->ders[i];
                buff_site_energy_ += 0.5 * val * pair_potentials_coeffs[i];
                buff_site_energy_ders_[j] += 0.5 * vec_j * ((der / dist_j) * pair_potentials_coeffs[i]);
            }

            // density calculation
            for (int i = 0; i < radial_basis_count_; i++) {
                const double val = p_RadialBasis->vals[i];
                density += val * density_coeffs[i];
                density_grad[i] += val;
            }
        }

        // embedding function and its derivative WRT density
        double density_pow = density; // will be density^m
        double embedding = embedding_coeffs[0];
        double embedding_der = 0;
        double embedding_der2 = 0;
        out_grad_accumulator[2*radial_basis_count_] += se_weight * 1;
        for (int m = 2; m < embedding_parameters_count_+1; m++) {
            embedding_der += m * embedding_coeffs[m-1] * density_pow;
            density_pow *= density;
            embedding += embedding_coeffs[m-1] * density_pow;
            out_grad_accumulator[m-1+2*radial_basis_count_] += se_weight * density_pow;
        }
        density_pow = 1; // will be density^m
        for (int m = 2; m < embedding_parameters_count_+1; m++) {
            embedding_der2 += m * (m - 1) * embedding_coeffs[m-1] * density_pow;
            density_pow *= density;
        }

        buff_site_energy_ += embedding;

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(nbh.dists[j]);
            // now we can compute the derivative of the embedding WRT atom positions
            for (int i = 0; i < radial_basis_count_; i++) {
                const double der = p_RadialBasis->ders[i];
                buff_site_energy_ders_[j] += vec_j * ((der / dist_j) * density_coeffs[i]) * embedding_der;
            }
        }

        // Computing the functional
        //double functional = out_site_energy * se_weight;
        //for (int j = 0; j < nbh.count; j++)
        //functional += out_site_energy_ders[j].dot(se_ders_weights[j]);

        //int coeff_count = CoeffCount();

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];
            const double vec_j_se_ders_inner = vec_j.dot(se_ders_weights[j]);

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(nbh.dists[j]);

            double total_density_coeff_der = 0;
            int i = 0;
            for (; i < radial_basis_count_; i++) {
                const double val = p_RadialBasis->vals[i];
                const double der = p_RadialBasis->ders[i];
                out_grad_accumulator[i] += 0.5 * se_weight * val;
                out_grad_accumulator[i] += 0.5 * vec_j_se_ders_inner * (der / dist_j);
                total_density_coeff_der += density_coeffs[i] * der;
            }
            for (; i < 2 * radial_basis_count_; i++) {
                const double val = p_RadialBasis->vals[i - radial_basis_count_];
                const double der = p_RadialBasis->ders[i - radial_basis_count_];
                out_grad_accumulator[i] += se_weight * embedding_der * val;
                out_grad_accumulator[i] += vec_j_se_ders_inner * embedding_der2
                    * density_grad[i - radial_basis_count_] * total_density_coeff_der / dist_j;
                out_grad_accumulator[i] += vec_j_se_ders_inner * embedding_der * (der / dist_j);
            }
            double density_pou = density; // will be density^m
            for (i = 2 * radial_basis_count_ + 2; i <= CoeffCount(); i++) {
                int idx = i - 2 * radial_basis_count_;
                out_grad_accumulator[i-1] += vec_j_se_ders_inner * idx * density_pou * total_density_coeff_der / dist_j;
                density_pou *= density;
            }
        }
    }
};


class LJSmooth : public AnyLocalMLIP
{
public:
    Array1D reg_coeffs;

    std::map<int, int> type_mapping;    // maps atomic numbers to 0,1,2,...

    Array1D basis_vals;                             //!< Array of the basis functions values calculated for certain neighborhood of the certain configuration
    Array3D basis_ders;                             //!< Array derivatives w.r.t. each atom in neiborhood of the basis functions values calculated for certain neighborhood of the certain configuration

    inline int index(int type_ind1, int type_ind2)  // calculate index in coeffs array (taking into account symmetry index(a,b)=index(b,a))
    {
        if (type_mapping.count(type_ind1)==0 || type_mapping.count(type_ind2)==0)
            ERROR("LJSmooth: Attempting to treat new types");
        type_ind1 = type_mapping[type_ind1]; // reusing of variable type_ind1
        type_ind2 = type_mapping[type_ind2]; // reusing of variable type_ind2
        if (type_ind2 < type_ind1)           // formula valid only to the oppposite case
            std::swap(type_ind2, type_ind1);
        return type_ind1*(type_count-1) - (type_ind1-3)*type_ind1/2 + (type_ind2-type_ind1);
    };

public:
    int type_count = 0;
    double cutoff = 5.0;
    double r_joint = 4.0;

    void SetTypeCount(int n)
    {
        type_count = n;
        reg_coeffs.resize(2*n*(n+1)/2);
    }

    void CalcPair(double r, double c1, double c2, double& E, double& dEdr, double* dEdc=nullptr, double* d2Edrdc=nullptr)
    {
#ifdef MLIP_DEBUG
        if (r <= 0)
            ERROR("Interatomic distance can not be zero or negative");
#endif // MLIP_DEBUG

        if (r <= r_joint)
        {
            double r6 = r*r*r;
            r6 *= r6;
            E = c1/(r6*r6) + c2/r6;
            dEdr = -6 * (E + c1/(r6*r6)) / r;

            if (dEdc != nullptr)
            {
                dEdc[0] = 1.0 / (r6*r6);
                dEdc[1] = 1.0 / r6;
            }
            if (d2Edrdc != nullptr)
            {
                d2Edrdc[0] = -12.0 / (r6*r6 * r);
                d2Edrdc[1] = -6.0 / r6 / r;
            }
        }
        else if (r < cutoff)
        {
            double rc_r0 = cutoff - r_joint;
            double& rj = r_joint;

            double rj6 = rj*rj*rj;
            rj6 *= rj6;
            double rj12 = rj6*rj6;

            double ej = c1/rj12 + c2/rj6;
            double fj = -6 * (ej + c1/rj12) / rj;

            double pc0 = ej;
            double pc1 = fj;
            double pc2 = -(3*ej + 2*fj*rc_r0) / (rc_r0*rc_r0);//-3 * (pc0 + pc1*rc_r0) / (rc_r0*rc_r0);
            double pc3 = (2*ej + fj*rc_r0) / (rc_r0*rc_r0*rc_r0);//2 * (pc0 + pc1*rc_r0) / (rc_r0*rc_r0*rc_r0);
            
            double dr = r - r_joint;

            E = pc0 + pc1*dr + pc2*dr*dr + pc3*dr*dr*dr;
            dEdr = pc1 + 2*pc2*dr + 3*pc3*dr*dr;

            if (dEdc != nullptr)
            {
                dEdc[0] = dr*dr*dr * (-12*rc_r0/(rj12*rj) + 2/rj12) / (rc_r0*rc_r0*rc_r0)
                        + dr*dr*(24*rc_r0/(rj12*rj) - 3/rj12) / (rc_r0*rc_r0)
                        - 12*dr/(rj12*rj) 
                        + 1/rj12;
                dEdc[1] = dr*dr*dr * (-6*rc_r0/(rj6*rj) + 2/rj6) / (rc_r0*rc_r0*rc_r0)
                        + dr*dr*(12*rc_r0/(rj6*rj) - 3/rj6) / (rc_r0*rc_r0)
                        - 6*dr/(rj6*rj) 
                        + 1/rj6;
            }
            if (d2Edrdc != nullptr)
            {
                d2Edrdc[0] = 3*dr*dr * (-12*rc_r0/(rj12*rj) + 2/rj12) / (rc_r0*rc_r0*rc_r0)
                           + 2*dr * (24*rc_r0/(rj12*rj) - 3/rj12) / (rc_r0*rc_r0)
                           - 12 / (rj12*rj);
                d2Edrdc[1] = 3*dr*dr * (-6*rc_r0/(rj6*rj) + 2/rj6) / (rc_r0*rc_r0*rc_r0)
                           + 2*dr * (12*rc_r0/(rj6*rj) - 3/rj6) / (rc_r0*rc_r0)
                           - 6 / (rj6*rj);
            }
        }
        else
        {
            E = 0.0;
            dEdr = 0.0;

            if (dEdc != nullptr)
            {
                dEdc[0] = 0.0;
                dEdc[1] = 0.0;
            }
            if (d2Edrdc != nullptr)
            {
                d2Edrdc[0] = 0.0;
                d2Edrdc[1] = 0.0;
            }
        }
    }

    void CalcEFS(Configuration& cfg)
    {
        ResetEFS(cfg);
        cfg.has_site_energies(true);
        memset(&cfg.site_energy(0), 0, cfg.size()*sizeof(double));

        if (cfg.nbh_cutoff != cutoff)
            cfg.InitNbhs(cutoff);

        for (int i=0; i<cfg.nbhs.size(); i++) 
        {
            Neighborhood& nbh = cfg.nbhs[i];
            for (int j=0; j<nbh.count; j++)
            {
                double r = nbh.dists[j];
                double c1 = reg_coeffs[2*index(nbh.my_type, nbh.types[j])+0];
                double c2 = reg_coeffs[2*index(nbh.my_type, nbh.types[j])+1];

                double E, dEdr;
                CalcPair(r, c1, c2, E, dEdr);

                cfg.site_energy(i) += 0.5*E;

                for (int a=0; a<3; a++)
                {
                    double force_val = 0.5 * dEdr * nbh.vecs[j][a] / r;
                    cfg.force(nbh.my_ind, a) += force_val; 
                    cfg.force(nbh.inds[j], a)-= force_val;
                    for (int b=0; b<3; b++)
                        cfg.stresses[a][b] -= force_val * nbh.vecs[j][b];
                }
            }
            cfg.energy += cfg.site_energy(i);
        }
    }

    void CalcBasisFuncsDers(Neighborhood& nbh)
    {
        ERROR("Not implemented function call");
    }

    void CalcEFSGrads(Configuration & cfg,
                      Array1D & out_ene_grad,
                      Array3D & out_frc_grad,
                      Array3D & out_str_grad) override
    {
        out_ene_grad.resize(CoeffCount());
        out_frc_grad.resize(cfg.size(), 3, CoeffCount());
        out_str_grad.resize(3, 3, CoeffCount());

        FillWithZero(out_ene_grad);
        out_frc_grad.set(0);
        out_str_grad.set(0);

        if (cfg.nbh_cutoff != CutOff())
            cfg.InitNbhs(CutOff());

        for (int i=0; i<cfg.nbhs.size(); i++)
        {
            Neighborhood& nbh = cfg.nbhs[i];

            for (int j=0; j<nbh.count; j++)
            {
                double r = nbh.dists[j];
                double c1 = reg_coeffs[2*index(nbh.my_type, nbh.types[j])+0];
                double c2 = reg_coeffs[2*index(nbh.my_type, nbh.types[j])+1];

                double E, dEdr, dEdc[2], d2Edrdc[2];
                CalcPair(r, c1, c2, E, dEdr, dEdc, d2Edrdc);

                out_ene_grad[0] += 0.5*dEdc[0];
                out_ene_grad[1] += 0.5*dEdc[1];

                for (int a=0; a<3; a++)
                {
                    double dfdc[2];
                    dfdc[0] = 0.5 * d2Edrdc[0] * nbh.vecs[j][a] / r;
                    dfdc[1] = 0.5 * d2Edrdc[1] * nbh.vecs[j][a] / r;
                    out_frc_grad(nbh.my_ind, a, 0) += dfdc[0];
                    out_frc_grad(nbh.my_ind, a, 1) += dfdc[1];
                    out_frc_grad(nbh.inds[j], a, 0) -= dfdc[0];
                    out_frc_grad(nbh.inds[j], a, 1) -= dfdc[1];
                    for (int b=0; b<3; b++)
                    {
                        out_str_grad(a, b, 0) -= dfdc[0] * nbh.vecs[j][b];
                        out_str_grad(a, b, 1) -= dfdc[1] * nbh.vecs[j][b];
                    }
                }
            }
        }
    }

    LJSmooth()
        : AnyLocalMLIP(1.0, 5.0, 0) // dummy
    {
    
    }

    int CoeffCount() override
    {
        return (int)reg_coeffs.size();
    }
    double* Coeff() override
    {
        return reg_coeffs.data();
    }
    void AccumulateCombinationGrad(const Neighborhood& nbh,                                // Must calculate and accumulate for a given atomic neighborhood nbh the gradient w.r.t coefficients of a linear combination site_energy*se_weight + scalar_product(se_ders_weights, site_energy_ders). It must also update buff_site_energy_ and buff_site_energy_ders_
        Array1D& out_grad_accumulator,                          // Gradient of linear combination must be accumulated in out_grad_accumulator. It should not be zeroed before accumulation in this function
        const double se_weight = 0.0,                           // se_weight is a weight of site energy gradients (w.r.t. coefficients) in a linear combination
        const Vector3* se_ders_weights = nullptr) override
    {
        ERROR("Not implemented function call");
    }
};


class LennardJones : public AnyLocalMLIP
{
protected:
    std::vector<double> regression_coeffs;

public:
    const std::string filename;
    int atom_types_count_;

    LennardJones(const std::string filename_, double _min_val = 1.6, double _max_val = 5.0, int _radial_basis_count = 0)
        : AnyLocalMLIP(_min_val, _max_val, _radial_basis_count)
        , filename(filename_)
    {
        Load(filename);
    }

    // this may disapear in future:
    int CoeffCount() //!< number of coefficients
    {
        return (int)regression_coeffs.size();
    }
    double* Coeff() //!< coefficients themselves
    {
        return &regression_coeffs[0];
    }

    void Load(const std::string& filename)
    {
        std::ifstream ifs(filename);

        try {
            if (!ifs.is_open()) throw MlipException("wrong filename!");
        }
        catch(...)
        {
            ERROR("Can't open file \"" + filename + "\" for reading");
        }

        std::string line;

        getline(ifs, line);

        std::string foo_str;
        char foo_char;
        ifs >> foo_str >> foo_char >> atom_types_count_;
        getline(ifs, line);

        regression_coeffs.resize(2*atom_types_count_*atom_types_count_);

        ifs >> foo_str >> foo_char >> foo_char;

        for (int i = 0; i < CoeffCount(); i++) {
            ifs >> regression_coeffs[i] >> foo_char;
        }
    }

    void Save(const std::string& filename)
    {
        std::ofstream ofs(filename);

        ofs << "LJLinear" << std::endl;
    
        ofs << "species_count = " << atom_types_count_ << std::endl;

        ofs.setf(std::ios::scientific);
        ofs.precision(15);

        ofs << "lj_coefficients = {";

        for (int i = 0; i < CoeffCount()-1; i++) {
            ofs << regression_coeffs[i] << ", ";
        }

        ofs << regression_coeffs[CoeffCount()-1] << "}" << std::endl;
    }

protected:
    void CalcSiteEnergyDers(const Neighborhood& nbh) override
    {
        buff_site_energy_ = 0.0;
        buff_site_energy_ders_.resize(nbh.count);
        FillWithZero(buff_site_energy_ders_);

        const double* a_coeffs = &regression_coeffs[0];
        const double* b_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_];

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];
            const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
            const double dist12 = dist6 * dist6;
            const int idx = nbh.types[j]  + nbh.my_type * atom_types_count_;

            buff_site_energy_ += a_coeffs[idx] / dist12 - b_coeffs[idx] / dist6;
            buff_site_energy_ders_[j] += - 12 * vec_j * a_coeffs[idx] / (dist12 * dist_j * dist_j);
            buff_site_energy_ders_[j] += 6 * vec_j * b_coeffs[idx] / (dist6 * dist_j * dist_j);
        }
    }

    void AccumulateCombinationGrad(const Neighborhood& nbh,
                                    std::vector<double>& out_grad_accumulator,
                                    const double se_weight = 0.0,
                                    const Vector3* se_ders_weights = nullptr) override
    {
        out_grad_accumulator.resize(CoeffCount());

        const double* a_coeffs = &regression_coeffs[0];
        const double* b_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_];

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];
            const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
            const double dist12 = dist6 * dist6;
            const int a_idx = nbh.types[j]  + nbh.my_type * atom_types_count_;
            const int b_idx = nbh.types[j]  + nbh.my_type * atom_types_count_ +
                atom_types_count_ * atom_types_count_;

            buff_site_energy_ += a_coeffs[a_idx] / dist12 - b_coeffs[a_idx] / dist6;
            buff_site_energy_ders_[j] -= 12 * vec_j * a_coeffs[a_idx] / (dist12 * dist_j * dist_j);
            buff_site_energy_ders_[j] += 6 * vec_j * b_coeffs[a_idx] / (dist6 * dist_j * dist_j);

            out_grad_accumulator[a_idx] += se_weight / dist12;
            out_grad_accumulator[b_idx] -= se_weight / dist6;
            if (se_ders_weights != nullptr) {
                out_grad_accumulator[a_idx] -= 12 * vec_j.dot(se_ders_weights[j]) / (dist12 * dist_j * dist_j);
                out_grad_accumulator[b_idx] += 6 * vec_j.dot(se_ders_weights[j]) / (dist6 * dist_j * dist_j);
            }
        }
    }
};

class NonlinearLennardJones : public AnyLocalMLIP
{
protected:
    std::vector<double> regression_coeffs;

public:
    const std::string filename;
    int atom_types_count_;

    NonlinearLennardJones(const std::string filename_, double _min_val = 1.6, double _max_val = 5.0, int _radial_basis_count = 0)
        : AnyLocalMLIP(_min_val, _max_val, _radial_basis_count)
        , filename(filename_)
    {
        Load(filename);
    }


    // this may disapear in future:
    int CoeffCount() //!< number of coefficients
    {
        return (int)regression_coeffs.size();
    }
    double* Coeff() //!< coefficients themselves
    {
        return &regression_coeffs[0];
    }

    void Load(const std::string& filename)
    {
        std::ifstream ifs(filename);

        try {
            if (!ifs.is_open()) throw MlipException("wrong filename!");
        }
        catch(...)
        {
            ERROR("Can't open file \"" + filename + "\" for reading");
        }

        std::string line;

        getline(ifs, line);

        std::string foo_str;
        char foo_char;
        ifs >> foo_str >> foo_char >> atom_types_count_;
        getline(ifs, line);

        regression_coeffs.resize(2*atom_types_count_);

        ifs >> foo_str >> foo_char >> foo_char;

        for (int i = 0; i < CoeffCount(); i++) {
            ifs >> regression_coeffs[i] >> foo_char;
        }
    }

    void Save(const std::string& filename)
    {
        std::ofstream ofs(filename);

        ofs << "LJNonlinear" << std::endl;
    
        ofs << "species_count = " << atom_types_count_ << std::endl;

        ofs.setf(std::ios::scientific);
        ofs.precision(15);

        ofs << "lj_coefficients = {";

        for (int i = 0; i < CoeffCount()-1; i++) {
            ofs << regression_coeffs[i] << ", ";
        }

        ofs << regression_coeffs[CoeffCount()-1] << "}" << std::endl;
    }

protected:

    void AddSiteEnergyGrad_old(const Neighborhood& neighborhood, 
        std::vector<double>& site_energy_grad)
    {
        for (int j = 0; j < neighborhood.count; j++) {
            //const Vector3& vec_j = neighborhood.vecs[j];
            const double dist_j = neighborhood.dists[j];
            const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
            const double dist12 = dist6 * dist6;
            const int a_idx = neighborhood.types[j]  + neighborhood.my_type * atom_types_count_;
            const int b_idx = neighborhood.types[j]  + neighborhood.my_type * atom_types_count_ + 
                                atom_types_count_ * atom_types_count_;

            site_energy_grad[a_idx] += 0.5 * 1 / dist12;
            site_energy_grad[b_idx] += 0.5 * -1 / dist6;
        }   
    }

    void CalcSiteEnergyDers_old(const Neighborhood& nbh)
    {
        buff_site_energy_ = 0.0;
        buff_site_energy_ders_.resize(nbh.count);
        FillWithZero(buff_site_energy_ders_);

        const double* a_coeffs = &regression_coeffs[0];
        const double* b_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_];

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];
            const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
            const double dist12 = dist6 * dist6;
            const int idx = nbh.types[j]  + nbh.my_type * atom_types_count_;

            buff_site_energy_ += 0.5 * a_coeffs[idx] / dist12 - 0.5 * b_coeffs[idx] / dist6;
            buff_site_energy_ders_[j] += - 0.5 * 12 * vec_j * a_coeffs[idx] / (dist12 * dist_j * dist_j);
            buff_site_energy_ders_[j] += 0.5 * 6 * vec_j * b_coeffs[idx] / (dist6 * dist_j * dist_j);
        }
    }

    void SiteEnergyDersGrad_old(const Neighborhood& nbh,
        std::vector<double>& out_grad_accumulator, 
        double se_weight, 
        const std::vector<Vector3>& se_ders_weights)
    {
        out_grad_accumulator.resize(CoeffCount());
        buff_site_energy_ = 0.0;
        buff_site_energy_ders_.resize(nbh.count);
        FillWithZero(buff_site_energy_ders_);

        const double* a_coeffs = &regression_coeffs[0];
        const double* b_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_];

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];
            const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
            const double dist12 = dist6 * dist6;
            const int a_idx = nbh.types[j] + nbh.my_type * atom_types_count_;
            const int b_idx = nbh.types[j] + nbh.my_type * atom_types_count_ +
                                    atom_types_count_ * atom_types_count_;

            buff_site_energy_ += 0.5 * a_coeffs[a_idx] / dist12 - 0.5 * b_coeffs[a_idx] / dist6;
            buff_site_energy_ders_[j] -= 0.5 * 12 * vec_j * a_coeffs[a_idx] / (dist12 * dist_j * dist_j);
            buff_site_energy_ders_[j] += 0.5 * 6 * vec_j * b_coeffs[a_idx] / (dist6 * dist_j * dist_j);
            
            out_grad_accumulator[a_idx] += 0.5 * se_weight / dist12;
            out_grad_accumulator[b_idx] -= 0.5 * se_weight / dist6;
            //if (se_ders_weights != nullptr) {
                out_grad_accumulator[a_idx] -= 0.5 * 12 * vec_j.dot(se_ders_weights[j]) / (dist12 * dist_j * dist_j);
                out_grad_accumulator[b_idx] += 0.5 * 6 * vec_j.dot(se_ders_weights[j]) / (dist6 * dist_j * dist_j);
            //}
        }
    }

    void AddSiteEnergyGrad(const Neighborhood& neighborhood, 
        std::vector<double>& site_energy_grad)
    {
        const double* a_coeffs = &regression_coeffs[0];
        const double* b_coeffs = &regression_coeffs[atom_types_count_];

        for (int j = 0; j < neighborhood.count; j++) {
            //const Vector3& vec_j = neighborhood.vecs[j];
            const double dist_j = neighborhood.dists[j];
            const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
            const double dist12 = dist6 * dist6;
            const int a_idx1 = neighborhood.my_type;
            const int a_idx2 = neighborhood.types[j];
            const int b_idx1 = neighborhood.my_type + atom_types_count_;
            const int b_idx2 = neighborhood.types[j] + atom_types_count_;

            /*double a1a2_der_idx1 = 0.5 * sqrt(a_coeffs[a_idx2] / a_coeffs[a_idx1]);
            double a1a2_der_idx2 = 0.5 * sqrt(a_coeffs[a_idx1] / a_coeffs[a_idx2]);
            double b1b2_der_idx1 = 0.5 * sqrt(b_coeffs[a_idx2] / b_coeffs[a_idx1]);
            double b1b2_der_idx2 = 0.5 * sqrt(b_coeffs[a_idx1] / b_coeffs[a_idx2]);*/

            double a1a2_der_idx1 = 2 * a_coeffs[a_idx2] * a_coeffs[a_idx1] * a_coeffs[a_idx2];
            double a1a2_der_idx2 = 2 * a_coeffs[a_idx1] * a_coeffs[a_idx2] * a_coeffs[a_idx1];
            double b1b2_der_idx1 = 2 * b_coeffs[a_idx2] * b_coeffs[a_idx1] * b_coeffs[a_idx2];
            double b1b2_der_idx2 = 2 * b_coeffs[a_idx1] * b_coeffs[a_idx2] * b_coeffs[a_idx1];

            site_energy_grad[a_idx1] += 0.5 * 1 * a1a2_der_idx1/ dist12;
            site_energy_grad[a_idx2] += 0.5 * 1 * a1a2_der_idx2 / dist12;
            site_energy_grad[b_idx1] += 0.5 * -1 * b1b2_der_idx1 / dist6;
            site_energy_grad[b_idx2] += 0.5 * -1 * b1b2_der_idx2 / dist6;
        }   
    }

    void CalcSiteEnergyDers(const Neighborhood& nbh) override
    {
        buff_site_energy_ = 0.0;
        buff_site_energy_ders_.resize(nbh.count);
        FillWithZero(buff_site_energy_ders_);

        const double* a_coeffs = &regression_coeffs[0];
        const double* b_coeffs = &regression_coeffs[atom_types_count_];

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];
            const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
            const double dist12 = dist6 * dist6;
            const int idx1 = nbh.my_type;
            const int idx2 = nbh.types[j];

            //double a1a2 = sqrt(a_coeffs[idx1] * a_coeffs[idx2]);
            //double b1b2 = sqrt(b_coeffs[idx1] * b_coeffs[idx2]);

            double a1a2 = a_coeffs[idx1] * a_coeffs[idx2] * a_coeffs[idx1] * a_coeffs[idx2];
            double b1b2 = b_coeffs[idx1] * b_coeffs[idx2] * b_coeffs[idx1] * b_coeffs[idx2];

            buff_site_energy_ += 0.5 * a1a2 / dist12 - 0.5 * b1b2 / dist6;
            buff_site_energy_ders_[j] += - 0.5 * 12 * vec_j * a1a2 / (dist12 * dist_j * dist_j);
            buff_site_energy_ders_[j] += 0.5 * 6 * vec_j * b1b2 / (dist6 * dist_j * dist_j);
        }
    }

    void AccumulateCombinationGrad( const Neighborhood& nbh,
                                    std::vector<double>& out_grad_accumulator,
                                    const double se_weight = 0.0,
                                    const Vector3* se_ders_weights = nullptr) override
    {
        out_grad_accumulator.resize(CoeffCount());
        buff_site_energy_ = 0.0;
        buff_site_energy_ders_.resize(nbh.count);
        FillWithZero(buff_site_energy_ders_);

        const double* a_coeffs = &regression_coeffs[0];
        const double* b_coeffs = &regression_coeffs[atom_types_count_];

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& vec_j = nbh.vecs[j];
            const double dist_j = nbh.dists[j];
            const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
            const double dist12 = dist6 * dist6;
            const int a_idx1 = nbh.my_type;
            const int a_idx2 = nbh.types[j];
            const int b_idx1 = nbh.my_type + atom_types_count_;
            const int b_idx2 = nbh.types[j] + atom_types_count_;

            /*double a1a2 = sqrt(a_coeffs[a_idx1] * a_coeffs[a_idx2]);
            double b1b2 = sqrt(b_coeffs[a_idx1] * b_coeffs[a_idx2]);

            double a1a2_der_idx1 = 0.5 * sqrt(a_coeffs[a_idx2] / a_coeffs[a_idx1]);
            double a1a2_der_idx2 = 0.5 * sqrt(a_coeffs[a_idx1] / a_coeffs[a_idx2]);
            double b1b2_der_idx1 = 0.5 * sqrt(b_coeffs[a_idx2] / b_coeffs[a_idx1]);
            double b1b2_der_idx2 = 0.5 * sqrt(b_coeffs[a_idx1] / b_coeffs[a_idx2]);*/

            double a1a2 = a_coeffs[a_idx1] * a_coeffs[a_idx2] * a_coeffs[a_idx1] * a_coeffs[a_idx2];
            double b1b2 = b_coeffs[a_idx1] * b_coeffs[a_idx2] * b_coeffs[a_idx1] * b_coeffs[a_idx2];

            double a1a2_der_idx1 = 2 * a_coeffs[a_idx2] * a_coeffs[a_idx1] * a_coeffs[a_idx2];
            double a1a2_der_idx2 = 2 * a_coeffs[a_idx1] * a_coeffs[a_idx2] * a_coeffs[a_idx1];
            double b1b2_der_idx1 = 2 * b_coeffs[a_idx2] * b_coeffs[a_idx1] * b_coeffs[a_idx2];
            double b1b2_der_idx2 = 2 * b_coeffs[a_idx1] * b_coeffs[a_idx2] * b_coeffs[a_idx1];

            buff_site_energy_ += 0.5 * a1a2 / dist12 - 0.5 * b1b2 / dist6;
            buff_site_energy_ders_[j] -= 0.5 * 12 * vec_j * a1a2 / (dist12 * dist_j * dist_j);
            buff_site_energy_ders_[j] += 0.5 * 6 * vec_j * b1b2 / (dist6 * dist_j * dist_j);
            
            out_grad_accumulator[a_idx1] += 0.5 * se_weight * a1a2_der_idx1 / dist12;
            out_grad_accumulator[b_idx1] -= 0.5 * se_weight * b1b2_der_idx1 / dist6;
            out_grad_accumulator[a_idx2] += 0.5 * se_weight * a1a2_der_idx2 / dist12;
            out_grad_accumulator[b_idx2] -= 0.5 * se_weight * b1b2_der_idx2 / dist6;
            if (se_ders_weights != nullptr) {
                out_grad_accumulator[a_idx1] -= 0.5 * 12 * vec_j.dot(se_ders_weights[j]) *
                            a1a2_der_idx1 / (dist12 * dist_j * dist_j);
                out_grad_accumulator[b_idx1] += 0.5 * 6 * vec_j.dot(se_ders_weights[j]) *
                            b1b2_der_idx1 / (dist6 * dist_j * dist_j);
                out_grad_accumulator[a_idx2] -= 0.5 * 12 * vec_j.dot(se_ders_weights[j]) *
                            a1a2_der_idx2 / (dist12 * dist_j * dist_j);
                out_grad_accumulator[b_idx2] += 0.5 * 6 * vec_j.dot(se_ders_weights[j]) *
                            b1b2_der_idx2 / (dist6 * dist_j * dist_j);
            }
        }
    }

};

#endif // MLIP_SamplePotentials
