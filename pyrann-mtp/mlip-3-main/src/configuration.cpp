/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#include <sstream>
#include <fstream>
#include <iomanip>
#include <set>
#include <iostream>
#include <algorithm>
#include "configuration.h"



using namespace std;


// Configuration file format definitions
#define LAT_FORMATTING right << setw(LAT_WIDTH) << setprecision(LAT_PRECISION)
#define IND_FORMATTING right << setw(IND_WIDTH) << setprecision(IND_PRECISION)
#define TYP_FORMATTING right << setw(TYP_WIDTH) << setprecision(TYP_PRECISION)
#define POS_FORMATTING right << setw(POS_WIDTH) << setprecision(POS_PRECISION)
#define SEN_FORMATTING right << setw(SEN_WIDTH) << setprecision(SEN_PRECISION)
#define FRC_FORMATTING right << setw(FRC_WIDTH) << setprecision(FRC_PRECISION)
#define CHG_FORMATTING right << setw(CHG_WIDTH) << setprecision(CHG_PRECISION)
#define GRD_FORMATTING right << setw(GRD_WIDTH) << setprecision(GRD_PRECISION)
#define MAG_FORMATTING right << setw(MAG_WIDTH) << setprecision(MAG_PRECISION)
#define EDM_FORMATTING right << setw(EDM_WIDTH) << setprecision(EDM_PRECISION)
#define ENE_FORMATTING right << setw(ENE_WIDTH) << setprecision(ENE_PRECISION)
#define STR_FORMATTING right << setw(STR_WIDTH) << setprecision(STR_PRECISION)



//! Procedure adds ghost atoms to the configuration according to the periodic extention.
//! The number of atoms in extended configuration is available via pos_.size().
void Configuration::InitNbhs_AddGhostAtoms(int cfg_size, double cutoff)
{
    if (pos_.empty())
        return;

    // computing the bounding box enclosing all the atoms
    Vector3 min_pos = pos_[0];
    Vector3 max_pos = pos_[0];
    for (int i = 1; i < cfg_size; i++) {
        for (int a = 0; a < 3; a++) {
            if (min_pos[a] > pos_[i][a])
                min_pos[a] = pos_[i][a];
            if (max_pos[a] < pos_[i][a])
                max_pos[a] = pos_[i][a];
        }
    }

    // is the configuration periodic in any dimension?
    bool is_periodic[3];
    for (int a = 0; a < 3; a++)
        is_periodic[a] = lattice[a][0] != 0.0 || lattice[a][1] != 0.0 || lattice[a][2] != 0.0;

    // periodically extends the box until no new ghost atoms are added
    bool added_new_points = true;
    for (int l = 1; added_new_points; l++) {
        added_new_points = false;

        // length of extension in 3 directions
        int ll[3];
        for (int a = 0; a < 3; a++)
            ll[a] = is_periodic[a] ? l : 0;

        for (int i = -ll[0]; i <= ll[0]; i++) {
            for (int j = -ll[1]; j <= ll[1]; j++) {
                for (int k = -ll[2]; k <= ll[2]; k++) {
                    if (std::max(std::max(abs(i), abs(j)), abs(k)) == l) {
                        for (int ind = 0; ind < cfg_size; ind++)
                        {
                            Vector3 candidate = pos_[ind];
                            for (int a = 0; a < 3; a++)
                                candidate[a] += i*lattice[0][a] + j*lattice[1][a]
                                + k*lattice[2][a];

                            if ((candidate[0] < min_pos[0] - cutoff) ||
                                (candidate[1] < min_pos[1] - cutoff) ||
                                (candidate[2] < min_pos[2] - cutoff) ||
                                (candidate[0] > max_pos[0] + cutoff) ||
                                (candidate[1] > max_pos[1] + cutoff) ||
                                (candidate[2] > max_pos[2] + cutoff))
                                continue;

                            int jnd;
                            for (jnd = 0;
                                (jnd < cfg_size) && distance(candidate, pos_[jnd]) > cutoff; jnd++)
                                ;

                            if (jnd < cfg_size) { // found a point to add
                                pos_.push_back(candidate);
                                orig_atom_inds.push_back(ind);
                                added_new_points = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

//! Fills vects, dists, etc., for the neighborhoods
//! For atoms from periodical extensions indices of their atom-origins are used
void Configuration::InitNbhs_ConstructNbhs(int cfg_size, double cut_off)
{
    cfg_size -= (int)ghost_inds.size();
    nbhs.resize(cfg_size);

    // Loop over all pairs of atoms in configuration
    for (int i=0; i<cfg_size; i++)
    {
        for (int j=i+1; j<(int)pos_.size(); j++)
        {   // pos_.size() != size after configuration extension
            Vector3 vect_ij = pos_[j] - pos_[i];

            double dist_ij = vect_ij.Norm();

            if ((dist_ij < cut_off) && (i <= orig_atom_inds[j]))
            {
                // for (i > orig_atom_inds[j]) this connection should be already treated
                Neighborhood& neighborhood_i = nbhs[orig_atom_inds[i]];// Neighborhood& neighborhood_i = nbhs[i] - is the same
                
                neighborhood_i.count++;
                neighborhood_i.inds.push_back(orig_atom_inds[j]);
                neighborhood_i.vecs.emplace_back();
                neighborhood_i.vecs.back() = vect_ij;
                neighborhood_i.dists.push_back(dist_ij);
                neighborhood_i.types.push_back(type(orig_atom_inds[j]));

                if (i != orig_atom_inds[j] && orig_atom_inds[j]<cfg_size)
                {   //(i == orig_atom_inds[j]) if distance between atom and its image < CuttOffRad
                    Neighborhood& neighborhood_j = nbhs[orig_atom_inds[j]];

                    neighborhood_j.count++;
                    neighborhood_j.inds.push_back(i);
                    neighborhood_j.vecs.emplace_back();
                    neighborhood_j.vecs.back() = -vect_ij;
                    neighborhood_j.dists.push_back(dist_ij);
                    neighborhood_j.types.push_back(type(i));
                }
            }
        }
        nbhs[i].my_type = type(i);
        nbhs[i].my_ind = i;
    }
}

void Configuration::InitNbhs_RemoveGhostAtoms(int cfg_size)
{
    pos_.resize(cfg_size);
    orig_atom_inds.clear();
}

void Configuration::InitNbhs(const double _cutoff)
{
#ifdef MLIP_DEBUG
    if (_cutoff < 0.0) ERROR("Invalid cutoff");
#endif
    nbhs.clear();

    orig_atom_inds.resize(size());
    for (int i=0; i<size(); i++)
        orig_atom_inds[i] = i;

    const int cfg_size = size();
    if (ghost_inds.empty())
        InitNbhs_AddGhostAtoms(cfg_size, _cutoff);
    InitNbhs_ConstructNbhs(cfg_size, _cutoff);
    if (ghost_inds.empty())
        InitNbhs_RemoveGhostAtoms(cfg_size);

    nbh_cutoff = _cutoff;
}

//!< neighborhood initialization procedure

void Configuration::ClearNbhs()
{
    nbhs.clear();
    nbh_cutoff = 0.0;
}

void Configuration::destroy()
{
    lattice[0][0] = lattice[0][1] = lattice[0][2]
        = lattice[1][0] = lattice[1][1] = lattice[1][2]
        = lattice[2][0] = lattice[2][1] = lattice[2][2] = 0.0;
    types_.clear();
    pos_.clear();
    forces_.clear();
    magmom_.clear();
    en_der_wrt_magmom_.clear();
    site_energies_.clear();
    charges_.clear();
    has_energy_ = false;
    has_stresses_ = false;
    features.clear();
    ClearNbhs();
    ghost_inds.clear();
}

void Configuration::NearestNeighborDistance(vector<double>& nearest_neigh_dist) {

    //define number of atomic types in configuration
    vector<int> at_type;
    at_type.push_back(types_[0]);

    for (int i = 1; i < size(); i++) {
        bool flag = true;
        for (int j = 0; j < at_type.size(); j++) {
            if (types_[i] == at_type[j]) {
                flag = false;
                break;
            }
        }
        if (flag) at_type.push_back(types_[i]);
    }

    //calculate nearest neighbor distance for each atomic type
    nearest_neigh_dist.resize(at_type.size());

    for (int i = 0; i < at_type.size(); i++)  
        nearest_neigh_dist[i] = 9e99;

    for (int i = 0; i < size(); i++) {
        for (int j = 0; j < i; j++) {
            if (types_[i] == types_[j]) {
                double dist = (pos_[i][0]-pos_[j][0])*(pos_[i][0]-pos_[j][0]);
                dist += (pos_[i][1]-pos_[j][1])*(pos_[i][1]-pos_[j][1]);
                dist += (pos_[i][2]-pos_[j][2])*(pos_[i][2]-pos_[j][2]);
                dist = sqrt(dist);
                if (dist < nearest_neigh_dist[types_[i]]) nearest_neigh_dist[types_[i]] = dist;
            }
        }
    }

}

bool Configuration::operator==(const Configuration& cfg) const
{
    const double TOL = 1.0e-14;

    if (lattice != cfg.lattice) return false;

    for (int i = 0; i < pos_.size(); ++i) {
        if (pos_.size() != cfg.pos_.size()) return false;
        const Vector3& p1 = pos_[i];
        const Vector3& p2 = cfg.pos_[i];
        for (int j = 0;  j < 3; ++j) {
            double d = p1[j] - p2[j];
            if (abs(d) > TOL) return false;
        }
    }
//  if (pos_ != cfg.pos_) return false;

    if (has_energy() != cfg.has_energy()) return false;
    if (has_energy()) {
        double d = energy - cfg.energy;
        if (abs(d) > TOL) return false;
    }
//  if (has_energy() && energy != cfg.energy) return false;

    if (has_stresses() != cfg.has_stresses()) return false;
    if (has_stresses() && stresses != cfg.stresses) return false;

    if (has_forces() != cfg.has_forces()) return false;

    if (has_forces()) {
        if (forces_.size() != cfg.forces_.size()) return false;
        for (int i = 0; i < forces_.size(); ++i) {
            const Vector3& p1 = forces_[i];
            const Vector3& p2 = cfg.forces_[i];
            for (int j = 0;  j < 3; ++j) {
                double d = p1[j] - p2[j];
                if (abs(d) > TOL) return false;
                }
        }
    }
//  if (has_forces() && forces_ != cfg.forces_) return false;

    if (has_magmom() != cfg.has_magmom()) return false;
    if (has_magmom()) {
        if (magmom_.size() != cfg.magmom_.size()) return false;
        for (int i = 0; i < magmom_.size(); ++i) {
            const Vector3& p1 = magmom_[i];
            const Vector3& p2 = cfg.magmom_[i];
            for (int j = 0;  j < 3; ++j) {
                double d = p1[j] - p2[j];
                if (abs(d) > TOL) return false;
                }
        }
    }

//  if (has_magmom() && magmom_ != cfg.magmom_) return false;

    if (has_en_der_wrt_magmom() != cfg.has_en_der_wrt_magmom()) return false;
    if (has_en_der_wrt_magmom()) {
        if (en_der_wrt_magmom_.size() != cfg.en_der_wrt_magmom_.size()) return false;
        for (int i = 0; i < en_der_wrt_magmom_.size(); ++i) {
            const Vector3& p1 = en_der_wrt_magmom_[i];
            const Vector3& p2 = cfg.en_der_wrt_magmom_[i];
            for (int j = 0;  j < 3; ++j) {
                double d = p1[j] - p2[j];
                if (abs(d) > TOL) return false;
                }
        }
    }

//  if (has_en_der_wrt_magmom() && en_der_wrt_magmom_ != cfg.en_der_wrt_magmom_) return false;

    if (has_site_energies() != cfg.has_site_energies()) return false;
    if (has_site_energies()) {
        if (site_energies_.size() != cfg.site_energies_.size()) return false;
        for (int i = 0; i < site_energies_.size(); ++i) {
            double d = site_energies_[i] - cfg.site_energies_[i];
            if (abs(d) > TOL) return false;
        }
    }
//  if (has_site_energies() && site_energies_ != cfg.site_energies_) return false;

    if (has_charges() != cfg.has_charges()) return false;
    if (has_charges()) {
        if (charges_.size() != cfg.charges_.size()) return false;
        for (int i = 0; i < charges_.size(); ++i) {
            double d = charges_[i] - cfg.charges_[i];
            if (abs(d) > TOL) return false;
        }
    }

//  if (has_charges() && charges_ != cfg.charges_) return false;

    if (types_ != cfg.types_) return false;

    return true;
}

bool Configuration::operator!=(const Configuration& cfg) const
{
    return !operator==(cfg);
}

void Configuration::Deform(const Matrix3& mapping)
{
    lattice *= mapping;

    for (Vector3& coord : pos_)
        coord = coord * mapping;
}

void Configuration::MoveAtomsIntoCell()
{
    const Matrix3 map_to_lattice_basis = lattice.inverse();

    for (int i=0; i<(int)size(); i++)
    {
        Vector3 pos_in_lattice_basis = pos_[i] * map_to_lattice_basis;
        
        for (int a=0; a<3; a++)
            pos_in_lattice_basis[a] -= floor(pos_in_lattice_basis[a]);

        pos_[i] = pos_in_lattice_basis * lattice;
    }
}

void Configuration::CorrectSupercell()
{
    // 1. Make the lattice as orthogonal as possible
    bool proceed=true;
    while (proceed)
    {
        proceed = false;
        for (int i=0; i<3; i++)
        {
            Vector3 l_i = Vector3(lattice[i][0], lattice[i][1], lattice[i][2]);
            for (int j=0; j<3; j++)
                if (i != j)
                {
                    Vector3 l_j = Vector3(lattice[j][0], lattice[j][1], lattice[j][2]);
                    double orig_l_j_norm2 = l_j.NormSq();
                    double new_l_j_norm2 = (l_j + l_i).NormSq();
                    for (int itr=1; new_l_j_norm2 < orig_l_j_norm2; itr++)
                    {
                        l_j += l_i;
                        new_l_j_norm2 = (l_j + l_i).NormSq();
                        proceed = true;
                    }
                    new_l_j_norm2 = (l_j - l_i).NormSq();
                    for (int itr=1; new_l_j_norm2 < orig_l_j_norm2; itr++)
                    {
                        l_j -= l_i;
                        new_l_j_norm2 = (l_j - l_i).NormSq();
                        proceed = true;
                    }

                    lattice[j][0] = l_j[0];
                    lattice[j][1] = l_j[1];
                    lattice[j][2] = l_j[2];
                }
        }
    }

    // 2. Update atom positions
    MoveAtomsIntoCell();

    // 3. Make lattice matrix lower triangle
    Matrix3 Q;
    Matrix3 L;
    lattice.transpose().QRdecomp(Q, L);
    Deform(Q);

    // 4. Set zero triangle instead of small numbers
    lattice[0][1] = 0.0;
    lattice[0][2] = 0.0;
    lattice[1][2] = 0.0;

    // 5. Rotate forces if present
    if (has_forces())
        for (Vector3& force : forces_)
            force = force * Q;

    // 6. Rotate stresses if present
    if (has_stresses())
        stresses = Q.transpose() * stresses * Q;

    // 7. Clear nbh data
    ClearNbhs();
}

void Configuration::ReplicateUnitCell(int nx, int ny, int nz)
{
    int size_rep = size()*nx*ny*nz;
    Matrix3 lattice_rep;
    std::vector<int> types_rep(size_rep);
    std::vector<Vector3> pos_rep(size_rep);
    std::vector<Vector3> force_rep(size_rep);
    std::vector<Vector3> magmom_rep(size_rep);
    std::vector<Vector3> en_der_wrt_magmom_rep(size_rep);

    has_energy(true);
    has_site_energies(false);
    has_forces(true);
    has_magmom(true);
    has_en_der_wrt_magmom(true);
    has_stresses(true);
    has_charges(false);

    energy *= nx*ny*nz;
    stresses *= nx*ny*nz;

    for (int a = 0; a < 3; a++) {
        lattice_rep[0][a] = lattice[0][a] * nx; 
        lattice_rep[1][a] = lattice[1][a] * ny; 
        lattice_rep[2][a] = lattice[2][a] * nz; 
    }

    int idx = 0;

    for (int i = 0; i < size(); i++) {
        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < ny; jj++) {
                for (int kk = 0; kk < nz; kk++) {
                    for (int l = 0; l < 3; l++) {
                        pos_rep[idx][l] = pos_[i][l] + ii*lattice[0][l] + jj*lattice[1][l] + kk*lattice[2][l]; 
                        force_rep[idx][l] = forces_[i][l];
                        magmom_rep[idx][l] = magmom_[i][l];
                        en_der_wrt_magmom_rep[idx][l] = en_der_wrt_magmom_[i][l];
                    }
                    types_rep[idx] = types_[i];
                    idx++;
                }
            }
        }
    }

    resize(idx);

    lattice = lattice_rep;
    for (int i = 0; i < idx; i++) {
        types_[i] = types_rep[i];
        for (int l = 0; l < 3; l++) {
            pos_[i][l] = pos_rep[i][l]; 
            magmom_[i][l] = magmom_rep[i][l];
            en_der_wrt_magmom_[i][l] = en_der_wrt_magmom_rep[i][l];
        }
    }

}

//! Replicates the configuration by making the unit cell L -> AL
void Configuration::ReplicateUnitCell(templateMatrix3<int> A)
{
    has_energy(false);
    has_site_energies(false);
    has_forces(false);
    has_magmom(false);
    has_en_der_wrt_magmom(false);
    has_stresses(false);
    has_charges(false);

    // calculating detA and making sure detA>0
    int detA = A.det();
    if (detA < 0) {
        detA *= -1; A[0][0] *= -1; A[1][0] *= -1; A[2][0] *= -1;
        lattice[0][0] *= -1; lattice[0][1] *= -1; lattice[0][2] *= -1;
    }

    vector<templateVector3<int>> replicate_vectors;
    replicate_vectors.reserve(detA);

    // We need to find all (i,j,k) such that 0 <= (i,j,k)*A^{-1} < 1
    // and add those (i,j,k) into replicate_vectors.
    // Doing a silly loop

    templateVector3<int> lmn;
    for (lmn[0] = 0; lmn[0] < detA; lmn[0]++)
        for (lmn[1] = 0; lmn[1] < detA; lmn[1]++)
            for (lmn[2] = 0; lmn[2] < detA; lmn[2]++) {
                templateVector3<int> ijk = lmn*A;
                if (ijk[0] % detA == 0 && ijk[1] % detA == 0 && ijk[2] % detA == 0)
                    replicate_vectors.push_back(ijk / detA);
            }
    if (replicate_vectors.size() != detA)
        ERROR("replicate_vectors.size() != detA; something went wrong");

        // we use that replicate_vectors[0] == (0,0,0)

    int old_size;
    old_size = (int)pos_.size();
    pos_.resize(old_size * detA);
    for (int j = 1; j < detA; j++)
        for (int i = 0; i < old_size; i++)
            for (int a = 0; a < 3; a++) {
                pos_[i + old_size*j][a] = pos_[i][a]
                    + replicate_vectors[j][0] * lattice[0][a]
                    + replicate_vectors[j][1] * lattice[1][a]
                    + replicate_vectors[j][2] * lattice[2][a];
            }

    types_.resize(old_size * detA);
    for (int j = 1; j < detA; j++)
        for (int i = 0; i < old_size; i++)
            types_[i + j*old_size] = types_[i];

    Matrix3 A_double(
        (double)A[0][0], (double)A[0][1], (double)A[0][2],
        (double)A[1][0], (double)A[1][1], (double)A[1][2],
        (double)A[2][0], (double)A[2][1], (double)A[2][2]);
    lattice = A_double*lattice;

    MoveAtomsIntoCell();
}

Configuration::Configuration() :
    has_energy_(false),
    has_stresses_(false),
    lattice(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    stresses(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
{
}

Configuration::Configuration(   CrystalSystemType crystal_system, 
                                double lattice_parameter, 
                                int replication_times)
                                // TODO: basic_lattice -> deformation mapping
{
    if (crystal_system == CrystalSystemType::SC)
    {
        Matrix3 basic_lattice = Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); 
        basic_lattice *= lattice_parameter;
        lattice = basic_lattice * replication_times;

        for (int i = 0; i < replication_times; i++)
            for (int j = 0; j < replication_times; j++)
                for (int k = 0; k < replication_times; k++)
                {
                    pos_.emplace_back();

                    pos_.back()[0] = i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] = i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] = i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];
                }
    }
    else if (crystal_system == CrystalSystemType::BCC)
    {
        Matrix3 basic_lattice = Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        basic_lattice *= lattice_parameter;
        lattice = basic_lattice * replication_times;

        for (int i = 0; i < replication_times; i++)
            for (int j = 0; j < replication_times; j++)
                for (int k = 0; k < replication_times; k++)
                {
                    pos_.emplace_back();

                    pos_.back()[0] = i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] = i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] = i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

                    pos_.emplace_back();

                    pos_.back()[0] = 0.5 * (basic_lattice[0][0] + basic_lattice[1][0] + basic_lattice[2][0]);
                    pos_.back()[1] = 0.5 * (basic_lattice[0][1] + basic_lattice[1][1] + basic_lattice[2][1]);
                    pos_.back()[2] = 0.5 * (basic_lattice[0][2] + basic_lattice[1][2] + basic_lattice[2][2]);

                    pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];
                }
    }
    else if (crystal_system == CrystalSystemType::FCC)
    {
        Matrix3 basic_lattice = Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        basic_lattice *= lattice_parameter;
        lattice = basic_lattice * replication_times;

        for (int i = 0; i < replication_times; i++)
            for (int j = 0; j < replication_times; j++)
                for (int k = 0; k < replication_times; k++)
                {
                    pos_.emplace_back();

                    pos_.back()[0] = i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] = i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] = i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

                    pos_.emplace_back();

                    pos_.back()[0] = 0.5 * (0.0                 + basic_lattice[1][0] + basic_lattice[2][0]);
                    pos_.back()[1] = 0.5 * (basic_lattice[0][1] + basic_lattice[1][1] + basic_lattice[2][1]);
                    pos_.back()[2] = 0.5 * (basic_lattice[0][2] + basic_lattice[1][2] + basic_lattice[2][2]);

                    pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

                    pos_.emplace_back();

                    pos_.back()[0] = 0.5 * (basic_lattice[0][0] + basic_lattice[1][0] + basic_lattice[2][0]);
                    pos_.back()[1] = 0.5 * (basic_lattice[0][1] + 0.0                 + basic_lattice[2][1]);
                    pos_.back()[2] = 0.5 * (basic_lattice[0][2] + basic_lattice[1][2] + basic_lattice[2][2]);

                    pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

                    pos_.emplace_back();

                    pos_.back()[0] = 0.5 * (basic_lattice[0][0] + basic_lattice[1][0] + basic_lattice[2][0]);
                    pos_.back()[1] = 0.5 * (basic_lattice[0][1] + basic_lattice[1][1] + basic_lattice[2][1]);
                    pos_.back()[2] = 0.5 * (basic_lattice[0][2] + basic_lattice[1][2] + 0.0);

                    pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];
                }
    }
    else if (crystal_system == CrystalSystemType::HCP)
    {
        Matrix3 basic_lattice = Matrix3(1.0, 0.0, 0.0, -0.5, sqrt(3)/2.0, 0.0, 0.0, 0.0, 2*sqrt(2.0/3.0));
        basic_lattice *= lattice_parameter;
        lattice = basic_lattice * replication_times;

        for (int i = 0; i < replication_times; i++)
            for (int j = 0; j < replication_times; j++)
                for (int k = 0; k < replication_times; k++)
                {
                    pos_.emplace_back();

                    pos_.back()[0] = i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] = i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] = i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

                    pos_.emplace_back();

                    pos_.back()[0] = 2.0/3.0 * basic_lattice[0][0] + 1.0/3.0 * basic_lattice[1][0] + 0.5 * basic_lattice[2][0];
                    pos_.back()[1] = 2.0/3.0 * basic_lattice[0][1] + 1.0/3.0 * basic_lattice[1][1] + 0.5 * basic_lattice[2][1];
                    pos_.back()[2] = 2.0/3.0 * basic_lattice[0][2] + 1.0/3.0 * basic_lattice[1][2] + 0.5 * basic_lattice[2][2];

                    pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
                    pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
                    pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];
                }
    }
    types_.resize(size());

    has_energy_ = has_stresses_ = false;
}

Configuration::Configuration(   CrystalSystemType crystal_system, 
                                double basic_lattice_parameter1, 
                                double basic_lattice_parameter2, 
                                double basic_lattice_parameter3, 
                                int replication_times)
    : Configuration(crystal_system, 1, replication_times)
{
    Matrix3 stretch(    basic_lattice_parameter1, 0, 0,
                        0, basic_lattice_parameter2, 0,
                        0, 0, basic_lattice_parameter3);
    Deform(stretch);
}

Configuration::~Configuration()
{
//  destroy();
}

Configuration Configuration::GetCfgFromLocalEnv(int atom_ind, Matrix3 new_lattice) 
{
    if (atom_ind > size())
        ERROR("Incorrect atom index");

    Configuration extracted_cfg;
    extracted_cfg.lattice = new_lattice;
    extracted_cfg.resize(1);
    extracted_cfg.pos(0) = Vector3(0.0, 0.0, 0.0);
    extracted_cfg.type(0) = type(atom_ind);

    new_lattice *= 0.5;

    Vector3 nl1(new_lattice[0][0], new_lattice[0][1], new_lattice[0][2]);
    Vector3 nl2(new_lattice[1][0], new_lattice[1][1], new_lattice[1][2]);
    Vector3 nl3(new_lattice[2][0], new_lattice[2][1], new_lattice[2][2]);

    int sz = 1;
    auto inverse_new_lattice = new_lattice.inverse();

    if (ghost_inds.size() > 0)    // this indicates non-periodic cfg with ghost atoms
    {
        for (int i=0; i<size(); i++)
        {
            Vector3 rel_pos = pos(i) - pos(atom_ind);
            Vector3 frac_rel_pos = rel_pos * inverse_new_lattice;
            if (i != atom_ind && fabs(frac_rel_pos[0])<1 && fabs(frac_rel_pos[1])<1 && fabs(frac_rel_pos[2])<1)
            {
                extracted_cfg.resize(++sz);
                extracted_cfg.pos(sz-1) = rel_pos;
                extracted_cfg.type(sz-1) = type(i);
            }
        }
    }
    else    // periodic cfg
    {
        double cutoff_sq = (nl1+nl2+nl3).NormSq();
        cutoff_sq = max(cutoff_sq, (nl1+nl2-nl3).NormSq());
        cutoff_sq = max(cutoff_sq, (nl1-nl2-nl3).NormSq());
        cutoff_sq = max(cutoff_sq, (nl1-nl2+nl3).NormSq());

        Vector3 l1(lattice[0][0], lattice[0][1], lattice[0][2]);
        Vector3 l2(lattice[1][0], lattice[1][1], lattice[1][2]);
        Vector3 l3(lattice[2][0], lattice[2][1], lattice[2][2]);

        int ll1 = (l1.NormSq()==0) ? 0 : (int)sqrt(cutoff_sq/l1.NormSq())+1;
        int ll2 = (l2.NormSq()==0) ? 0 : (int)sqrt(cutoff_sq/l2.NormSq())+1;
        int ll3 = (l3.NormSq()==0) ? 0 : (int)sqrt(cutoff_sq/l3.NormSq())+1;

        for (int i=0; i<size(); i++)
            if (i != atom_ind)
                for (int ix=-ll1; ix<=ll1; ix++)
                    for (int iy=-ll2; iy<=ll2; iy++)
                        for (int iz=-ll3; iz<=ll3; iz++)
                        {
                            Vector3 gpos = pos(i) + ix*l1 + iy*l2 + iz*l3;

                            Vector3 rel_pos = gpos - pos(atom_ind);
                            Vector3 frac_rel_pos = rel_pos * inverse_new_lattice;
                            if (fabs(frac_rel_pos[0])<1 && fabs(frac_rel_pos[1])<1 && fabs(frac_rel_pos[2])<1)
                            {
                                extracted_cfg.resize(++sz);
                                extracted_cfg.pos(sz-1) = gpos - pos(atom_ind);
                                extracted_cfg.type(sz-1) = type(i);
                            }
                        }
    }

    for (int i=0; i<extracted_cfg.size(); i++)
        extracted_cfg.pos(i) += nl1 + nl2 + nl3;

    return extracted_cfg;
}

bool Configuration::Load(std::ifstream& ifs)
{
    if (!ifs.is_open()) ERROR("Input stream is not open");
    if (ifs.eof()) return false;
    if (ifs.fail()) ERROR("Input stream is open but not good");

    char tag = 0;

    streampos cfg_beg_pos = ifs.tellg();
    if (ifs.fail()) ERROR("tellg failed");
    ifs >> tag;
    if (ifs.eof()) return false;
    if (ifs.fail()) ERROR("cannot read even one symbol!");
    ifs.seekg(cfg_beg_pos);
    if (ifs.fail()) ERROR("seekg failed");

    // checks if this is the binary file format
    if (tag == 'n')
        return LoadBin(ifs);

    string tmp_str;

    ifs >> tmp_str;

    // checks if this is the old file format
    if (tmp_str == "BEGIN") {
        ifs.seekg(cfg_beg_pos);
        if (ifs.fail()) ERROR("seekg failed");
        return LoadOld(ifs);
    }

    // checks if this is the new file format
    if (tmp_str != "BEGIN_CFG") ERROR("unknown token " + tmp_str);

    destroy();

    has_energy_ = has_stresses_ = false;

    // these are internal variables, if they remain false then this is an error
    bool has_atom_pos = false;
    int _size = -1;
    bool is_stress_virial = false;
    bool is_stress_negative = false;
    bool frac_pos = false;

    while (!ifs.eof()) {
        ifs >> tmp_str;

        if (!ifs.good() && tmp_str != "END_CFG")
            ERROR("Unable to read field name");

        if (tmp_str == "Size") {
            int new_size;
            ifs >> new_size;
            if (_size != -1 && _size != new_size)
                ERROR("Size given, but does not match earlier size");
            _size = new_size;
            if (!ifs.good()) ERROR("Unable to read configuration size");
            if (_size < 0)  ERROR("Incorrect configuration size");
            resize(_size);
        }
        else if (tmp_str == "Supercell" || tmp_str == "SuperCell") {
            ifs >> lattice[0][0] >> lattice[0][1] >> lattice[0][2]
                >> lattice[1][0] >> lattice[1][1] >> lattice[1][2]
                >> lattice[2][0] >> lattice[2][1] >> lattice[2][2];
            ifs.clear();
        } else if (tmp_str == "AtomData:" || tmp_str == "Atomic_data:") {
            if (_size == 0)
                Warning("Configuration with no atoms has been read");

            if (_size == -1)
                ERROR("Reading AtomData without Size given is not implemented.");

            enum class DataType { ID, TYPE, XC, YC, ZC, XD, YD, ZD, FX, FY, FZ, SITE_EN, CHARGE, 
                MAGMOMX, MAGMOMY, MAGMOMZ, ENDERMAGMOMX, ENDERMAGMOMY, ENDERMAGMOMZ, GHOST, GRADES };
            vector<DataType> datamap;

            int control_summ_pos[3] = { 0,0,0 };
            int control_summ_frc[3] = { 0,0,0 };
            int control_summ_magmom[3] = { 0,0,0 };
            int control_summ_en_der_wrt_magmom[3] = { 0,0,0 };

            // in ids, '.first' is the id that is read and '.second' is the line number
            vector<int> ids;

            getline(ifs, tmp_str);
            istringstream keywords(tmp_str);
            do {
                string keyword;
                keywords >> keyword;

                if (keyword == "id") {
                    ids.resize(_size);
                    datamap.push_back(DataType::ID);
                } else if (keyword == "type") {
                    types_.resize(_size);
                    datamap.push_back(DataType::TYPE);
                } else if (keyword == "cartes_x") {
                    pos_.resize(_size);
                    datamap.push_back(DataType::XC);
                    control_summ_pos[0]++;
                } else if (keyword == "cartes_y") {
                    pos_.resize(_size);
                    datamap.push_back(DataType::YC);
                    control_summ_pos[1]++;
                } else if (keyword == "cartes_z") {
                    pos_.resize(_size);
                    datamap.push_back(DataType::ZC);
                    control_summ_pos[2]++;
                } else if (keyword == "direct_x") {
                    pos_.resize(_size);
                    datamap.push_back(DataType::XD);
                    control_summ_pos[0]--;
                } else if (keyword == "direct_y") {
                    pos_.resize(_size);
                    datamap.push_back(DataType::YD);
                    control_summ_pos[1]--;
                } else if (keyword == "direct_z") {
                    pos_.resize(_size);
                    datamap.push_back(DataType::ZD);
                    control_summ_pos[2]--;
                } else if (keyword == "site_en") {
                    has_site_energies(true);
                    datamap.push_back(DataType::SITE_EN);
                } else if (keyword == "fx") {
                    has_forces(true);
                    datamap.push_back(DataType::FX);
                    control_summ_frc[0]++;
                } else if (keyword == "fy") {
                    has_forces(true);
                    datamap.push_back(DataType::FY);
                    control_summ_frc[1]++;
                } else if (keyword == "fz") {
                    has_forces(true);
                    datamap.push_back(DataType::FZ);
                    control_summ_frc[2]++;
                } else if (keyword == "magmom_x") {
                    has_magmom(true);
                    datamap.push_back(DataType::MAGMOMX);
                    control_summ_magmom[0]++;
                } else if (keyword == "magmom_y") {
                    has_magmom(true);
                    datamap.push_back(DataType::MAGMOMY);
                    control_summ_magmom[1]++;
                } else if (keyword == "magmom_z") {
                    has_magmom(true);
                    datamap.push_back(DataType::MAGMOMZ);
                    control_summ_magmom[2]++;
                } else if (keyword == "en_der_mx") {
                    has_en_der_wrt_magmom(true);
                    datamap.push_back(DataType::ENDERMAGMOMX);
                    control_summ_en_der_wrt_magmom[0]++;
                } else if (keyword == "en_der_my") {
                    has_en_der_wrt_magmom(true);
                    datamap.push_back(DataType::ENDERMAGMOMY);
                    control_summ_en_der_wrt_magmom[1]++;
                } else if (keyword == "en_der_mz") {
                    has_en_der_wrt_magmom(true);
                    datamap.push_back(DataType::ENDERMAGMOMZ);
                    control_summ_en_der_wrt_magmom[2]++;
                } else if (keyword == "charge") {
                    has_charges(true);
                    datamap.push_back(DataType::CHARGE);
                } else if (keyword == "ghost") {
                    datamap.push_back(DataType::GHOST);
                } else if (keyword == "nbh_grades") {
                    has_nbh_grades(true);
                    datamap.push_back(DataType::GRADES);
                } else if (keyword == "")
                    break;
                else
                    ERROR("Unknown AtomData field");
            } while (keywords);

            // control_summ_pos should either be all 1 or -1 or all 0
            if (control_summ_pos[0] != control_summ_pos[1] ||
                control_summ_pos[0] != control_summ_pos[2] ||
                abs(control_summ_pos[0]) > 1)
                ERROR("Configuration reading eror. Wrong atom position information");
            if (control_summ_frc[0] != control_summ_frc[1] ||
                control_summ_frc[0] != control_summ_frc[2] ||
                control_summ_frc[0] > 1)
                ERROR("Configuration reading eror. Wrong force information");
            if (control_summ_magmom[0] != control_summ_magmom[1] ||
                control_summ_magmom[0] != control_summ_magmom[2] ||
                control_summ_magmom[0] > 1)
                ERROR("Configuration reading eror. Wrong magnetic moments information");
            if (control_summ_en_der_wrt_magmom[0] != control_summ_en_der_wrt_magmom[1] ||
                control_summ_en_der_wrt_magmom[0] != control_summ_en_der_wrt_magmom[2] ||
                control_summ_en_der_wrt_magmom[0] > 1)
                ERROR("Configuration reading eror. Wrong energy derivatives w.r.t. magnetic moments information");

            has_atom_pos = (abs(control_summ_pos[0]) == 1);
            frac_pos = (control_summ_pos[0] == -1);

            if (_size > 0) {
                for (int i = 0; i < size(); i++)
                    for (int j = 0; j < (int)datamap.size(); j++)
                        if (datamap[j] == DataType::TYPE)
                            ifs >> types_[i];
                        else if (datamap[j] == DataType::ID)
                            ifs >> ids[i];
                        else if (datamap[j] == DataType::XC)
                            ifs >> pos_[i][0];
                        else if (datamap[j] == DataType::YC)
                            ifs >> pos_[i][1];
                        else if (datamap[j] == DataType::ZC)
                            ifs >> pos_[i][2];
                        else if (datamap[j] == DataType::XD)
                            ifs >> pos_[i][0];
                        else if (datamap[j] == DataType::YD)
                            ifs >> pos_[i][1];
                        else if (datamap[j] == DataType::ZD)
                            ifs >> pos_[i][2];
                        else if (datamap[j] == DataType::FX)
                            ifs >> forces_[i][0];
                        else if (datamap[j] == DataType::FY)
                            ifs >> forces_[i][1];
                        else if (datamap[j] == DataType::FZ)
                            ifs >> forces_[i][2];
                        else if (datamap[j] == DataType::MAGMOMX)
                            ifs >> magmom_[i][0];
                        else if (datamap[j] == DataType::MAGMOMY)
                            ifs >> magmom_[i][1];
                        else if (datamap[j] == DataType::MAGMOMZ)
                            ifs >> magmom_[i][2];
                        else if (datamap[j] == DataType::ENDERMAGMOMX)
                            ifs >> en_der_wrt_magmom_[i][0];
                        else if (datamap[j] == DataType::ENDERMAGMOMY)
                            ifs >> en_der_wrt_magmom_[i][1];
                        else if (datamap[j] == DataType::ENDERMAGMOMZ)
                            ifs >> en_der_wrt_magmom_[i][2];
                        else if (datamap[j] == DataType::SITE_EN)
                            ifs >> site_energies_[i];
                        else if (datamap[j] == DataType::CHARGE)
                            ifs >> charges_[i];
                        else if (datamap[j] == DataType::GRADES)
                            ifs >> nbh_grades_[i];
                        else if (datamap[j] == DataType::GHOST)
                        {
                            string tmp_ghost;
                            ifs >> tmp_ghost;
                            if (tmp_ghost=="true")
                                ghost_inds.insert(i);
                            else if (tmp_ghost!="false")
                                ERROR("Ghost flag must be either \"true\" or \"false\"");
                        }
                        if (!ifs.good())
                            ERROR("Unable to read AtomData. Wrong Size specified?");
            } else {
                // HERE GOES THE CODE OF READING ATOMDATA WITHOUT SIZE GIVEN
            }

            // if ids are given (i.e., their size != 0) then check that they are 1, 2, 3,...
            for (int i = 0; i < (int)ids.size(); i++)
                if (ids[i] != i + 1)
                    ERROR("Invalid file format: wrong order of id\'s");

            if (!has_atom_pos) ERROR("Invalid file format: no atom positions");

            // if types are not given, then set all to zero
            if (types_.size() == 0) {
                types_.resize(size());
                FillWithZero(types_);
            }
        } else if (tmp_str == "Energy") {
            ifs >> energy;
            if (!ifs.good())
                ERROR("Configuration reading eror. Reading energy failed");
            has_energy_ = true;
        }
        else if (tmp_str == "Stress:" || tmp_str == "Virial:" || tmp_str == "PlusStress:") {
            if (tmp_str == "Virial:") is_stress_virial = true;
            if (tmp_str == "Stress:") {
                is_stress_negative = true;
#ifndef MLIP_NO_WRONG_STRESS_ERROR
                ERROR( "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!!          ERROR         ERROR        ERROR         !!!\n"
                "!!!      Read a configuration with (negative) Stress. !!!\n"
                "!!!      This feature will be removed soon!           !!!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#else
                Warning( "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!!         WARNING       WARNING      WARNING        !!!\n"
                "!!!      Read a configuration with (negative) Stress. !!!\n"
                "!!!      This feature will be removed soon!           !!!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#endif
            }

            vector<double*> datamap;

            getline(ifs, tmp_str);
            istringstream keywords(tmp_str);

            do {
                string keyword;
                keywords >> keyword;
                if (keyword == "xx")
                    datamap.push_back(&stresses[0][0]);
                else if (keyword == "yy")
                    datamap.push_back(&stresses[1][1]);
                else if (keyword == "zz")
                    datamap.push_back(&stresses[2][2]);
                else if (keyword == "xy")
                    datamap.push_back(&stresses[0][1]);
                else if (keyword == "xz")
                    datamap.push_back(&stresses[0][2]);
                else if (keyword == "yz")
                    datamap.push_back(&stresses[1][2]);
            } while (keywords);

            for (int j = 0; j < (int)datamap.size(); j++)
                ifs >> *datamap[j];
            if (!ifs.good())
                ERROR("Configuration reading error. Failed to read stresses");

            stresses[1][0] = stresses[0][1];
            stresses[2][1] = stresses[1][2];
            stresses[2][0] = stresses[0][2];
            has_stresses_ = true;
        }
        else if (tmp_str == "Features:") {
            getline(ifs, tmp_str);
            istringstream keywords(tmp_str);

            do {
                string keyword;
                keywords >> keyword;
                if (keyword != "")
                    ifs >> features[keyword];
            } while (keywords);

            if (!ifs.good())
                ERROR("Configuration reading error (Broken features)");
        } else if (tmp_str == "Feature") {
            string key, value;
            ifs >> key;
            ifs.ignore(1);
            key = mlip_string_unescape(key);
            getline(ifs, value);
            if (!ifs.good())
                ERROR("Configuration reading error (Broken feature)");

            if (features.count(key) == 0)
                features[key] = value;
            else // multiline feature value
                features[key] += '\n' + value;

        } else if (tmp_str == "END_CFG")
            break;
        else
            ERROR("Configuration reading error. Unknown field \"" + tmp_str + "\"");
    }

    if (is_stress_virial)
        stresses *= 1.0 / fabs(lattice.det());
    if (is_stress_negative)
        stresses *= -1;
    if (frac_pos)
        if (lattice.det() == 0.0)
            ERROR("Non-periodic unitcell and fractional coordinates detected (incompatible case)");
        else
            for (int i=0; i<size(); i++)
                pos_[i] = pos_[i] * lattice;

    if (!has_atom_pos && size() == 0)
        return false;
    else
        return true;
}

bool Configuration::LoadOld(ifstream& ifs)
{
    destroy();

    if (!ifs.is_open())
        ERROR("Input stream not open");

    has_energy_ = has_stresses_ = false;

    bool _has_lattice = false;
    bool _has_atom_pos = false;
    bool _has_atom_types = false;
    int _size = 0;

    string tmp_str = "";

    ifs >> tmp_str;
    if (tmp_str != "BEGIN")
        return false;

    while (!ifs.eof())
    {
        ifs >> tmp_str;
        if (ifs.fail()) ERROR("Unable to read field name");

        if (tmp_str == "Size")
        {
            ifs >> _size;
        }
        else if (tmp_str == "MinusStress")
        {
            ifs >> stresses[0][0]
                >> stresses[1][1]
                >> stresses[2][2]
                >> stresses[0][1]
                >> stresses[1][2]
                >> stresses[0][2];

            stresses[1][0] = stresses[0][1];
            stresses[2][1] = stresses[1][2];
            stresses[2][0] = stresses[0][2];

            has_stresses_ = true;
        }
        else if (tmp_str == "MinusStress9")
        {
            ifs >> stresses[0][0]
                >> stresses[0][1]
                >> stresses[0][2]
                >> stresses[1][0]
                >> stresses[1][1]
                >> stresses[1][2]
                >> stresses[2][0]
                >> stresses[2][1]
                >> stresses[2][2];

            has_stresses_ = true;
        }
        else if ((tmp_str == "VirialStress") || (tmp_str == "Stress"))
        {
            ifs >> stresses[0][0]
                >> stresses[1][1]
                >> stresses[2][2]
                >> stresses[0][1]
                >> stresses[1][2]
                >> stresses[0][2];

            stresses[1][0] = stresses[0][1];
            stresses[2][1] = stresses[1][2];
            stresses[2][0] = stresses[0][2];

            stresses *= -1;

            has_stresses_ = true;
        }
        else if (tmp_str == "Lattice")
        {
            ifs >> lattice[0][0] >> lattice[0][1] >> lattice[0][2]
                >> lattice[1][0] >> lattice[1][1] >> lattice[1][2]
                >> lattice[2][0] >> lattice[2][1] >> lattice[2][2];
            _has_lattice = true;
        }
        else if ((tmp_str == "AtmData") || (tmp_str == "AtomData"))
        {
            double x, y, z;
            // loop while we keep reading numbers
            for (ifs >> x; ifs.good(); ifs >> x) {
                ifs >> y >> z;
                if (ifs.good()) {
                    pos_.emplace_back();
                    pos_.back()[0] = x;
                    pos_.back()[1] = y;
                    pos_.back()[2] = z;
                }
                else ERROR("Error reading coordinates");

                ifs >> x >> y >> z;
                if (ifs.good()) {
                    forces_.emplace_back();
                    forces_.back()[0] = x;
                    forces_.back()[1] = y;
                    forces_.back()[2] = z;
                }
                else ERROR("Error reading forces");
            }
            ifs.clear();
            _has_atom_pos = true;
        }
        else if (tmp_str == "AtomPos")
        {
            double x, y, z;
            // loop while we keep reading numbers
            for (ifs >> x; ifs.good(); ifs >> x) {
                ifs >> y >> z;
                if (ifs.good()) {
                    pos_.emplace_back();
                    pos_.back()[0] = x;
                    pos_.back()[1] = y;
                    pos_.back()[2] = z;
                }
                else ERROR("Error reading coordinates");
            }
            ifs.clear();
            _has_atom_pos = true;
        }
        else if (tmp_str == "AtomTypes") 
        {
            int type;
            // loop while we keep reading numbers
            for (ifs >> type; ifs.good(); ifs >> type)
                types_.push_back(type);
            ifs.clear();
            _has_atom_types = true;
        } 
        else if (tmp_str == "SiteEnergies")
        {
            double foo;
            // loop while we keep reading numbers
            ifs >> foo;
            while (ifs.good())
            {
                site_energies_.push_back(foo);
                ifs >> foo;
            }
            ifs.clear();
        }
        else if (tmp_str == "AtomForces")
        {
            double x, y, z;
            // loop while we keep reading numbers
            for (ifs >> x; ifs.good(); ifs >> x) {
                ifs >> y >> z;
                if (ifs.good()) {
                    forces_.emplace_back();
                    forces_.back()[0] = x;
                    forces_.back()[1] = y;
                    forces_.back()[2] = z;
                }
                else ERROR("Error reading forces");
            }
            ifs.clear();
        }
        else if (tmp_str == "Charges")
        {
            double foo;
            // loop while we keep reading numbers
            ifs >> foo;
            while (!ifs.fail())
            {
                charges_.push_back(foo);
                ifs >> foo;
            }
            ifs.clear();
        }
        else if (tmp_str == "Energy")
        {
            ifs >> energy;
            if (!ifs.good()) ERROR("Error reading energy");
            has_energy_ = true;
        }
        else if (tmp_str == "Feature")
        {
            std::string name, value;
            ifs >> name >> value;
            features[name] = value;
        }
        else if (tmp_str == "END")
            break;
        else 
            ERROR("Unknown configuration field \"" + tmp_str + "\"");

        if (!ifs.good()) ERROR("error reading");
    }
    
    if (!(_has_atom_pos && _has_lattice)) ERROR("No lattice or atom positions");
    if (_size == 0)
        _size = (int)pos_.size();
    if (_has_atom_types && types_.size() != _size) ERROR("Problem with configuration size");
    if (_size != (int)pos_.size()) ERROR("Problem with configuration size");
    if (has_forces() && (_size != (int)forces_.size())) ERROR("Problem with configuration size");
    if (has_site_energies() && (_size != (int)site_energies_.size())) ERROR("Problem with configuration size");
    if (has_charges() && (_size != (int)charges_.size())) ERROR("Problem with configuration size");
    if (has_magmom() && (_size != (int)magmom_.size())) ERROR("Problem with configuration size");
    if (has_en_der_wrt_magmom() && (_size != (int)en_der_wrt_magmom_.size())) ERROR("Problem with configuration size");

    if (!_has_atom_types) types_.resize(_size, 0);
    _has_atom_types = true;

    return true;
}

bool Configuration::LoadBin(std::ifstream & ifs)
{
    destroy();

    if (!ifs.is_open())
        ERROR("Input stream not open");

    char tag = 0;
    has_energy_ = has_stresses_ = false;
    bool has_pos = false;
    bool has_lattice = false;
    bool has_types = false;
    int _size;

    ifs >> tag;
    if (tag != 'n') {
        if (ifs.eof()) return false; // empty configuration
        ERROR("Corrupted binary configuration file");
    }
    else
        ifs.read((char*) &_size, sizeof(_size));

    if (_size < 0)
        ERROR("Corrupted binary configuration file");

    while (!ifs.eof())
    {
        ifs >> tag;

        if (tag == 'l') {
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    ifs.read((char*)&lattice[a][b], sizeof(lattice[0][0]));
            has_lattice = true;
        }
        else if (tag == 'p') {
            pos_.resize(_size);
            ifs.read((char*)&pos_[0], _size*sizeof(pos_[0]));
            has_pos = true;
        }
        else if (tag == 't') {
            types_.resize(_size);
            ifs.read((char*)&types_[0], _size*sizeof(types_[0]));
            has_types = true;
        } 
        else if (tag == 'f') {
            forces_.resize(_size);
            ifs.read((char*)&forces_[0], _size*sizeof(forces_[0]));
        } 
        else if (tag == 'm') {
            magmom_.resize(_size);
            ifs.read((char*)&magmom_[0], _size*sizeof(magmom_[0]));
        }
        else if (tag == 'd') {
            en_der_wrt_magmom_.resize(_size);
            ifs.read((char*)&en_der_wrt_magmom_[0], _size*sizeof(en_der_wrt_magmom_[0]));
        }
        else if (tag == 's') {
            ifs.read((char*)&stresses[0][0], sizeof(stresses[0][0]));
            ifs.read((char*)&stresses[1][1], sizeof(stresses[0][0]));
            ifs.read((char*)&stresses[2][2], sizeof(stresses[0][0]));
            ifs.read((char*)&stresses[0][1], sizeof(stresses[0][0]));
            ifs.read((char*)&stresses[1][2], sizeof(stresses[0][0]));
            ifs.read((char*)&stresses[0][2], sizeof(stresses[0][0]));
            stresses[1][0] = stresses[0][1];
            stresses[2][1] = stresses[1][2];
            stresses[2][0] = stresses[0][2];
            has_stresses_ = true;
        }
        else if (tag == 'v') {
            site_energies_.resize(_size);
            ifs.read((char*)&site_energies_[0], _size*sizeof(site_energies_[0]));
        }
        else if (tag == 'c') {
            charges_.resize(_size);
            ifs.read((char*)&charges_[0], _size*sizeof(charges_[0]));
        }
        else if (tag == 'g') {
            vector<char> ghost_flag(_size);
            ifs.read((char*)&ghost_flag[0], _size*sizeof(ghost_flag[0]));
            for (int i=0; i<_size; i++)
                if (ghost_flag[i]=='1') ghost_inds.insert(i);
        }
        else if (tag == 'e') {
            ifs.read((char*)&energy, sizeof(energy));
            has_energy_ = true;
        } 
        else if (tag == 'F') {
            std::string name, value;
            int l;
            ifs.read((char*)&l, sizeof(l));
            name.resize(l);
            ifs.read(&name[0], l * sizeof(char));
            ifs.read((char*)&l, sizeof(l));
            value.resize(l);
            ifs.read(&value[0], l * sizeof(char));
            features[name] = value;
        } 
        else if (tag == '#')
            break;
        else
            ERROR("Unknown tag \'" + tag + "\'");
    }

    if (!has_types) types_.resize(_size, 0);
    has_types = true;

    if (!(has_lattice && has_pos)) ERROR("Corrupted binary configuration file");

    return true;
}

void Configuration::Save(ofstream& ofs, unsigned int flags) const
{
    int LAT_WIDTH; int LAT_PRECISION;
    int IND_WIDTH; int IND_PRECISION;
    int TYP_WIDTH; int TYP_PRECISION;
    int POS_WIDTH; int POS_PRECISION;
    int SEN_WIDTH; int SEN_PRECISION;
    int FRC_WIDTH; int FRC_PRECISION;
    int CHG_WIDTH; int CHG_PRECISION;
    int MAG_WIDTH; int MAG_PRECISION;
    int EDM_WIDTH; int EDM_PRECISION;
    int GRD_WIDTH; int GRD_PRECISION;
    int ENE_WIDTH; int ENE_PRECISION;
    int STR_WIDTH; int STR_PRECISION;

    unsigned int save_binary        = flags & SAVE_BINARY;
    unsigned int without_loss       = flags & SAVE_NO_LOSS;
    unsigned int direct_atom_coord  = flags & SAVE_DIRECT_COORDS;
    unsigned int save_ghost_atoms   = flags & SAVE_GHOST_ATOMS;

    if (!ofs.is_open())
        ERROR("Output stream not open");

    if (save_binary) {
        SaveBin(ofs);
        return;
    }

    ofs.precision(16);
    if (without_loss) {
        ofs << scientific;
        LAT_WIDTH = 1; LAT_PRECISION = 16;
        IND_WIDTH = 1; IND_PRECISION = 0;
        TYP_WIDTH = 1; TYP_PRECISION = 0;
        POS_WIDTH = 1; POS_PRECISION = 16;
        SEN_WIDTH = 1; SEN_PRECISION = 16;
        FRC_WIDTH = 1; FRC_PRECISION = 16;
        CHG_WIDTH = 1; CHG_PRECISION = 16;
        GRD_WIDTH = 1; GRD_PRECISION = 16;
        MAG_WIDTH = 1; MAG_PRECISION = 16;
        EDM_WIDTH = 1; EDM_PRECISION = 16;
        ENE_WIDTH = 1; ENE_PRECISION = 16;
        STR_WIDTH = 1; STR_PRECISION = 16;
    }
    else {
        ofs << fixed;
        LAT_WIDTH = 13; LAT_PRECISION =  6;
        IND_WIDTH = 10; IND_PRECISION =  0;
        TYP_WIDTH =  4; TYP_PRECISION =  0;
        POS_WIDTH = 13; POS_PRECISION =  6;
        SEN_WIDTH = 17; SEN_PRECISION = 12;
        FRC_WIDTH = 11; FRC_PRECISION =  6;
        CHG_WIDTH =  8; CHG_PRECISION =  4;
        GRD_WIDTH = 15; GRD_PRECISION =  5;
        MAG_WIDTH = 12; MAG_PRECISION =  6;
        EDM_WIDTH = 13; EDM_PRECISION =  5;
        ENE_WIDTH = 20; ENE_PRECISION = 12;
        STR_WIDTH = 11; STR_PRECISION =  5;
    }

    int _size;
    if (save_ghost_atoms)
        _size = size();
    else
        _size = size() - (int)ghost_inds.size();

    ofs << "BEGIN_CFG\n";

    ofs << " Size\n    " << _size << '\n';
    if (!is_mpi_splited)
    {
        int count = 0;
        for (int a = 0; a < 3; a++)
            if (lattice[a][0] != 0 || lattice[a][1] != 0 || lattice[a][2] != 0) count = a+1;
        
        if (count != 0) {
            ofs << " Supercell\n";
            for (int a = 0; a < count; a++)
                ofs << "    " << LAT_FORMATTING << lattice[a][0]
                    << ' ' << LAT_FORMATTING << lattice[a][1]
                    << ' ' << LAT_FORMATTING << lattice[a][2] << '\n';
        }
    }

    if (size() > 0)
    {
        ofs << " AtomData:  id type";
        
            if (direct_atom_coord)
                ofs << "       direct_x      direct_y      direct_z";
            else    
                ofs << "       cartes_x      cartes_y      cartes_z";
        if (has_site_energies())    ofs << "            site_en";
        if (has_forces())           ofs << "           fx          fy          fz";
        if (has_charges())          ofs << "      charge";
        if (has_magmom())           ofs << "      magmom_x     magmom_y     magmom_z";
        if (has_en_der_wrt_magmom())ofs << "      en_der_mx     en_der_my     en_der_mz";
        if (save_ghost_atoms && !ghost_inds.empty())          
            ofs << "     ghost";
        if (has_nbh_grades())          ofs << "       nbh_grades";
        ofs << '\n';

        for (int i=0; i<size(); i++)
        {
            if (!save_ghost_atoms && !ghost_inds.empty() && ghost_inds.count(i)!=0)
                continue;
            else
            {
                ofs << "    " << IND_FORMATTING << i+1;
                ofs << ' '    << TYP_FORMATTING << types_[i];
                    if (!direct_atom_coord)
                        ofs << "  "<< POS_FORMATTING << pos_[i][0]
                            << ' ' << POS_FORMATTING << pos_[i][1]
                            << ' ' << POS_FORMATTING << pos_[i][2];
                    else
                    {
                        Vector3 dpos = direct_pos(i);
                        ofs << "  "<< POS_FORMATTING << dpos[0]
                            << ' ' << POS_FORMATTING << dpos[1]
                            << ' ' << POS_FORMATTING << dpos[2];
                    }
                if (has_site_energies())    
                    ofs << "  " << SEN_FORMATTING << site_energy(i);
                if (has_forces())   
                    ofs << "  "<< FRC_FORMATTING << forces_[i][0]
                        << ' ' << FRC_FORMATTING << forces_[i][1]
                        << ' ' << FRC_FORMATTING << forces_[i][2];
                if (has_charges())
                    ofs << "    " << CHG_FORMATTING << charges(i);
                if (has_magmom())
                    ofs << "  "<< MAG_FORMATTING << magmom_[i][0]
                        << ' ' << MAG_FORMATTING << magmom_[i][1]
                        << ' ' << MAG_FORMATTING << magmom_[i][2];
                if (has_en_der_wrt_magmom())  
                    ofs << "  "<< EDM_FORMATTING << en_der_wrt_magmom_[i][0]
                        << ' ' << EDM_FORMATTING << en_der_wrt_magmom_[i][1]
                        << ' ' << EDM_FORMATTING << en_der_wrt_magmom_[i][2];
                if (save_ghost_atoms && !ghost_inds.empty())
                    if (ghost_inds.find(i) != ghost_inds.end())
                        ofs << "     true ";
                    else
                        ofs << "     false";
                if (has_nbh_grades())
                    ofs << "  "<< GRD_FORMATTING << nbh_grades_[i];
                ofs << '\n';
            }
        }
    }

    if (has_energy())
    {
        //if (log10(energy) > ENE_WIDTH-2)
        //    ERROR("Energy is too large ("+to_string(energy)+") and connot be written properly");
        ofs << " Energy\n    " << ENE_FORMATTING << energy << '\n';
    }

    if (has_stresses())
    {
        //double maxstress = max(fabs(stresses[0][0]), fabs(stresses[1][1]));
        //maxstress = max(maxstress, fabs(stresses[2][2]));
        //maxstress = max(maxstress, fabs(stresses[0][1]));
        //maxstress = max(maxstress, fabs(stresses[0][2]));
        //maxstress = max(maxstress, fabs(stresses[1][2]));
        //if (log10(maxstress) > STR_WIDTH-2)
        //    ERROR(("Stresses are too large ("+to_string(maxstress)+") and connot be written properly").c_str());

        ofs << " PlusStress:  xx          yy          zz"
            << "          yz          xz          xy\n";
                
            ofs << "     " << STR_FORMATTING << stresses[0][0]
                << ' ' << STR_FORMATTING << stresses[1][1]
                << ' ' << STR_FORMATTING << stresses[2][2]
                << ' ' << STR_FORMATTING << stresses[1][2]
                << ' ' << STR_FORMATTING << stresses[0][2]
                << ' ' << STR_FORMATTING << stresses[0][1];
        ofs << '\n';
    }

    if (features.size()>0)
    {
        for (auto& feature : features) 
        {
            std::string key = std::string(feature.first);
            std::string val = std::string(feature.second);
            while (!val.empty() && !key.empty())
            {
                size_t cut = val.find_first_of('\n');
                if (cut > val.size())
                {
                    ofs << " Feature   " << mlip_string_escape(key) << '\t' << val << '\n';
                    break;
                }
                else
                {
                    ofs << " Feature   " << mlip_string_escape(key)
                        << '\t' << val.substr(0, cut) << '\n';
                    val = val.substr(cut+1);
                    if (val.empty())
                        ofs << " Feature   " << mlip_string_escape(key)
                            << "\t\n";
                }
            }
        }
    }

    ofs << "END_CFG\n" << endl;
}

void Configuration::AppendToFile(const std::string & filename, unsigned int flags) const
{
    if (is_mpi_splited)
    {
#ifdef MLIP_DEBUG
        int splited_everywhere;
        int splited_check = is_mpi_splited ? 1 : 0;
        MPI_Allreduce(&splited_check, &splited_everywhere, 1, MPI_INT, MPI_SUM, mpi.comm);
        if (splited_everywhere != 0 && splited_everywhere != mpi.size)
            ERROR("Contradictory configuration status");
#endif

        int local_size = size() - (int)ghost_inds.size();
        double local_stress[6];
        local_stress[0] = stresses[0][0];
        local_stress[1] = stresses[1][1];
        local_stress[2] = stresses[2][2];
        local_stress[3] = stresses[1][2];
        local_stress[4] = stresses[0][2];
        local_stress[5] = stresses[0][1];

        int global_size;
        double global_energy;
        double local_energy = energy; 
        double global_stress[6];
        MPI_Reduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(local_stress, global_stress, 6, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);

        if (mpi.rank == 0)
        {
            ofstream ofs(filename, ios::app | ios::binary);
            ofs.precision(16);
            ofs << fixed;
            int LAT_WIDTH = 13; int LAT_PRECISION =  6;
            
            ofs << "BEGIN_CFG\n";

            {// writing size
                ofs << " Size\n    " << global_size << '\n';
            }

            {// writing lattice
                int count = 0;
                for (int a=0; a<3; a++)
                    if (lattice[a][0] != 0 || lattice[a][1] != 0 || lattice[a][2] != 0) count = a+1;

                if (count != 0) {
                    ofs << " Supercell\n";
                    for (int a = 0; a < count; a++)
                        ofs << "    " << LAT_FORMATTING << lattice[a][0]
                        << ' ' << LAT_FORMATTING << lattice[a][1]
                        << ' ' << LAT_FORMATTING << lattice[a][2] << '\n';
                }
            }

            {// writing AtomData heading
                ofs << " AtomData:  id type";

                ofs << "       cartes_x      cartes_y      cartes_z";
                if (has_site_energies())    ofs << "    site_en";
                if (has_forces())           ofs << "           fx          fy          fz";
                if (has_magmom())          ofs << "     magmom_x     magmom_y     magmom_z";
                if (has_en_der_wrt_magmom())          ofs << "      en_der_mx     en_der_my     en_der_mz";
                if (has_charges())          ofs << "      charge";
                if (has_nbh_grades())          ofs << "       nbh_grades";
                ofs << '\n';
            }
        }

        // writing AtomData
        int atoms_written_count = 0;
        for (int proc=0; proc<mpi.size; proc++)
        {
            if (proc == mpi.rank)
            {
                ofstream ofs(filename, ios::app | ios::binary);
                ofs.precision(16);
                ofs << fixed;
                int IND_WIDTH = 10; int IND_PRECISION =  0;
                int TYP_WIDTH =  4; int TYP_PRECISION =  0;
                int POS_WIDTH = 13; int POS_PRECISION =  6;
                int SEN_WIDTH = 17; int SEN_PRECISION = 12;
                int FRC_WIDTH = 11; int FRC_PRECISION =  6;
                int CHG_WIDTH =  8; int CHG_PRECISION =  4;
                int MAG_WIDTH =  8; int MAG_PRECISION =  4;
                int EDM_WIDTH =  9; int EDM_PRECISION =  4;
                int GRD_WIDTH = 15; int GRD_PRECISION =  5;

                for (int i=0; i<size(); i++)
                    if (ghost_inds.empty() || ghost_inds.count(i)==0)
                    {
                        atoms_written_count++;
                        ofs << "    " << IND_FORMATTING << atoms_written_count;

                        ofs << ' ' << TYP_FORMATTING << types_[i];

                        ofs << "  " << POS_FORMATTING << pos_[i][0]
                            << ' ' << POS_FORMATTING << pos_[i][1]
                            << ' ' << POS_FORMATTING << pos_[i][2];

                        if (has_site_energies())
                            ofs << "  " << SEN_FORMATTING << site_energy(i);

                        if (has_forces())
                            ofs << "  " << FRC_FORMATTING << forces_[i][0]
                            << ' ' << FRC_FORMATTING << forces_[i][1]
                            << ' ' << FRC_FORMATTING << forces_[i][2];

                        if (has_magmom())
                            ofs << "  " << MAG_FORMATTING << magmom_[i][0]
                            << ' ' << MAG_FORMATTING << magmom_[i][1]
                            << ' ' << MAG_FORMATTING << magmom_[i][2];

                        if (has_en_der_wrt_magmom())
                            ofs << "  " << EDM_FORMATTING << en_der_wrt_magmom_[i][0]
                            << ' ' << EDM_FORMATTING << en_der_wrt_magmom_[i][1]
                            << ' ' << EDM_FORMATTING << en_der_wrt_magmom_[i][2];

                        if (has_charges())
                            ofs << "    " << CHG_FORMATTING << charges(i);

                        if (has_nbh_grades())
                            ofs << "    " << GRD_FORMATTING << nbh_grades_[i];

                        ofs << '\n';
                    }
            }

            MPI_Bcast(&atoms_written_count, 1, MPI_INT, proc, mpi.comm);
        }

        if (mpi.rank == 0)
        {
            ofstream ofs(filename, ios::app | ios::binary);
            ofs.precision(16);
            ofs << fixed;
            int ENE_WIDTH = 20; int ENE_PRECISION = 12;
            int STR_WIDTH = 11; int STR_PRECISION =  5;

            if (has_energy())
                ofs << " Energy\n    " << ENE_FORMATTING << global_energy << '\n';

            if (has_stresses())
            {
                ofs << " PlusStress:  xx          yy          zz"
                    << "          yz          xz          xy\n";

                ofs << "     " << STR_FORMATTING << global_stress[0]
                    << ' ' << STR_FORMATTING << global_stress[1]
                    << ' ' << STR_FORMATTING << global_stress[2]
                    << ' ' << STR_FORMATTING << global_stress[3]
                    << ' ' << STR_FORMATTING << global_stress[4]
                    << ' ' << STR_FORMATTING << global_stress[5];
                ofs << '\n';
            }

            if (features.size()>0)
            {
                for (auto& feature : features) 
                {
                    std::string key = std::string(feature.first);
                    std::string val = std::string(feature.second);
                    while (!val.empty() && !key.empty())
                    {
                        size_t cut = val.find_first_of('\n');
                        if (cut > val.size())
                        {
                            ofs << " Feature   " << mlip_string_escape(key) << '\t' << val << '\n';
                            break;
                        }
                        else
                        {
                            ofs << " Feature   " << mlip_string_escape(key)
                                << '\t' << val.substr(0, cut) << '\n';
                            val = val.substr(cut+1);
                            if (val.empty())
                                ofs << " Feature   " << mlip_string_escape(key)
                                << "\t\n";
                        }
                    }
                }
            }

            ofs << "END_CFG\n" << endl;
        }
    }
    else
    {
        ofstream ofs(filename, ios::app | ios::binary);
        Save(ofs, flags);
    }
}

void Configuration::SaveBin(std::ofstream & ofs) const
{
    if (!ofs.is_open())
        ERROR("Output stream not open");

    int _size = size();

    ofs << 'n';
    ofs.write((char*) &_size, sizeof(_size));

    ofs << 'l';
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            ofs.write((char*) &lattice[a][b], sizeof(double));

    ofs << 'p';
        if ((_size > 0) && (!pos_.empty()))
            ofs.write((char*)&pos_[0], _size*sizeof(pos_[0]));

    ofs << 't';
    if ((_size > 0) && (!types_.empty()))
        ofs.write((char*)&types_[0], _size*sizeof(types_[0]));

    if (has_forces())
    {
        ofs << 'f';
        ofs.write((char*)&forces_[0], _size*sizeof(forces_[0]));
    }

    if (has_magmom())
    {
        ofs << 'm';
        ofs.write((char*)&magmom_[0], _size*sizeof(magmom_[0]));
    }

    if (has_en_der_wrt_magmom())
    {
        ofs << 'd';
        ofs.write((char*)&en_der_wrt_magmom_[0], _size*sizeof(en_der_wrt_magmom_[0]));
    }

    if (has_site_energies())
    {
        ofs << 'v';
        ofs.write((char*)&site_energies_[0], _size*sizeof(site_energies_[0]));
    }

    if (has_charges())
    {
        ofs << 'c';
        ofs.write((char*)&charges_[0], _size*sizeof(charges_[0]));
    }

    if (!ghost_inds.empty())
    {
        ofs << 'g';
        std::vector<char> ghost_flag(_size, '0'); // do not write vector<bool> it is not 
        for (int i : ghost_inds)
            ghost_flag[i] = '1';
        ofs.write((char*)ghost_flag.data(), _size*sizeof(ghost_flag[0]));
    }

    if (has_stresses())
    {
        ofs << 's';
        ofs.write((char*)&stresses[0][0], sizeof(stresses[0][0]));
        ofs.write((char*)&stresses[1][1], sizeof(stresses[0][0]));
        ofs.write((char*)&stresses[2][2], sizeof(stresses[0][0]));
        ofs.write((char*)&stresses[0][1], sizeof(stresses[0][0]));
        ofs.write((char*)&stresses[1][2], sizeof(stresses[0][0]));
        ofs.write((char*)&stresses[0][2], sizeof(stresses[0][0]));
    }

    if (has_energy())
    {
        ofs << 'e';
        ofs.write((char*)&energy, sizeof(energy));
    }
    
    for (auto& feature : features) 
    {
        ofs << 'F';
        int l;
        l = (int)feature.first.length();
        ofs.write((char*)&l, sizeof(l));
        ofs.write((char*)&feature.first[0], l*sizeof(char));
        l = (int)feature.second.length();
        ofs.write((char*)&l, sizeof(l));
        ofs.write((char*)&feature.second[0], l*sizeof(char));
    }

    ofs << '#';
    ofs.flush();
}

bool Configuration::LoadNextFromOUTCAR(std::ifstream& ifs, int maxiter)
{
    std::string line;
    
    int iter = -1;
    int lines_read = 0;

    // Skipping to the end of the SCF loop
    while (
        (line.substr(0, 104) !=
        "------------------------ aborting loop because EDIFF is reached ----------------------------------------")
        &&
        (line.substr(0, 43) != "                  Total CPU time used (sec)")
        ) 
    {
        std::getline(ifs, line);

        if (ifs.eof() || !ifs.good())
            ERROR((string)"Outcar-file parsing error: Can't find string \"aborting loop because EDIFF is reached\" "
            + to_string(++lines_read) + " lines have been read"); // remove: 3
        //Reading number of ionic iterations
      //else if (line.substr(0, 51) == "----------------------------------------- Iteration") //- not general, number of "-" symbols can vary!!!
		else if (line.find("--- Iteration") != std::string::npos)
        {
            //line.erase(0, 51);    //what if the number starts earlier? function works without this line
            std::stringstream stream(line);

            string s = "";
            char t=(char)0;

            while (t != '(')
                t = (char)stream.get();

            stream >> iter;
        }
    }

    if (maxiter != 0 && iter >= maxiter) {
        features["EFS_by"] = "VASP_not_converged";
        Warning("Loading configuration from OUTCAR: non-converged calculation detected. " + 
                 to_string(iter) + " iterations done");
    }
    else
        features["EFS_by"] = "VASP";

    // No "Total CPU time used (sec)" = likely broken outcar
    if (line.substr(0, 43) == "                  Total CPU time used (sec)")
        return false;

    // We now expect a configuration, so we can erase the current one
    has_energy(false);
    has_forces(false);
    has_stresses(false);
    pos_.clear();
    forces_.clear();

    // stresses reading block
    {
        // read until we find stresses (sometimes absent in the file) or lattice (always present)
        while (
            (line.substr(0, 24) != "  FORCE on cell =-STRESS") &&
            (line.substr(0, 71) != "      direct lattice vectors                 reciprocal lattice vectors")
            ) {
            std::getline(ifs, line);
            if (ifs.eof())
                ERROR((string)"Outcar-file parsing error: Can't find stresses");
        }

        // check if found the stresses
        if (line.substr(0, 24) == "  FORCE on cell =-STRESS") {

            while (line.substr(0, 7) != "  Total") {
                std::getline(ifs, line);
                if (ifs.eof())
                    ERROR((string)"Outcar-file parsing error: Can't find the stresses\"");
            }

            // now parsing the line
            std::stringstream stream(line);

            has_stresses(true);
            std::string tempstring;
            stream >> tempstring;

            stream >> stresses[0][0]
                >> stresses[1][1]
                >> stresses[2][2]
                >> stresses[0][1]
                >> stresses[1][2]
                >> stresses[0][2];

            stresses[1][0] = stresses[0][1];
            stresses[2][1] = stresses[1][2];
            stresses[2][0] = stresses[0][2];
        }
    }

    {// lattice reading block
        while (line.substr(0, 71) != "      direct lattice vectors                 reciprocal lattice vectors") {
            std::getline(ifs, line);
            if (ifs.eof())
                ERROR((string)"Outcar-file parsing error: Can't find lattice data");
        }
        double foo;
        ifs >> lattice[0][0] >> lattice[0][1] >> lattice[0][2] >> foo >> foo >> foo;
        ifs >> lattice[1][0] >> lattice[1][1] >> lattice[1][2] >> foo >> foo >> foo;
        ifs >> lattice[2][0] >> lattice[2][1] >> lattice[2][2] >> foo >> foo >> foo;
    }

    {// positions and forces reading block
        while (line.substr(0, 70) != " POSITION                                       TOTAL-FORCE (eV/Angst)") {
            std::getline(ifs, line);

            if (ifs.eof())
                ERROR((string)"Outcar-file parsing error: Can't find forces");
        }

        std::getline(ifs, line);
        while ((!ifs.fail()) && (!ifs.eof())) {
            pos_.emplace_back();
            forces_.emplace_back();
            ifs >> pos_.back()[0] >> pos_.back()[1] >> pos_.back()[2]
                >> forces_.back()[0] >> forces_.back()[1] >> forces_.back()[2];
        }

        if (ifs.eof())
            ERROR("error reading");
        ifs.clear();

        pos_.pop_back();
        forces_.pop_back();
    }

    {// energy reading block
        while (line.substr(0, 46) != "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)") {
            std::getline(ifs, line);
            if (ifs.eof())
                ERROR((string)"Outcar-file parsing error: Can't find energy");
        }
        std::getline(ifs, line);
        string str1 = "";
        string str2 = "";
        string str3 = "";
        string str4 = "";
        ifs >> str1 >> str2 >> str3 >> str4;
        if (str1 != "free" && str2 != "energy" && str3 != "TOTEN" && str4 != "=")
            ERROR((string)"Outcar-file parsing error: Can't read energy");
        ifs >> energy;
    }

    if (ifs.fail() || ifs.eof()) {
        ifs.close();
        ERROR("error reading");
    } else {
        has_energy_ = true;
        site_energies_.clear();
    }

    return true;
}

bool Configuration::LoadNextFromLAMMPSDump(std::ifstream& ifs)
{
    std::string line;    

    ifs >> line;

    if (ifs.eof()) return false;

    while (line.substr(0,21) != "ITEM: NUMBER OF ATOMS") {
        std::getline(ifs, line);
    }

    int size;
    if (line.substr(0,21) == "ITEM: NUMBER OF ATOMS") {
        ifs >> size;
        //cfg.resize(size);
    }

    while (line.substr(0,35) != "ITEM: BOX BOUNDS xy xz yz pp pp pp") {
        std::getline(ifs, line);
    }

    double lo_bound[3];
    double hi_bound[3];
    double lo[3];
    double hi[3];
    double xy, xz, yz;

    if (line.substr(0,35) == "ITEM: BOX BOUNDS xy xz yz pp pp pp") {
        ifs >> lo_bound[0] >> hi_bound[0] >> xy;
        std::getline(ifs, line);
        ifs >> lo_bound[1] >> hi_bound[1] >> xz;
        std::getline(ifs, line);
        ifs >> lo_bound[2] >> hi_bound[2] >> yz;

        lo[0] = lo_bound[0] - __min(__min(0.0,xy), __min(xz,xy+xz));
        hi[0] = hi_bound[0] - __max(__max(0.0,xy), __max(xz,xy+xz));
        lo[1] = lo_bound[1] - __min(0.0,yz);
        hi[1] = hi_bound[1] - __max(0.0,yz);
        lo[2] = lo_bound[2];
        hi[2] = hi_bound[2];

        for (int a = 0; a < 3; a++)
            lattice[a][a] = hi[a] - lo[a];

        lattice[1][0] = xy;
        lattice[2][0] = xz;
        lattice[2][1] = yz;
        lattice[0][1] = 0;
        lattice[0][2] = 0;
        lattice[1][2] = 0;  

        std::getline(ifs, line); 
    }

    while (line.substr(0,11) != "ITEM: ATOMS") {
        std::getline(ifs, line);
    }

    if (line.substr(0,11) == "ITEM: ATOMS") {
	pos_.resize(size);
	types_.resize(size);
        for (int i = 0; i < size; i++) {
            int id, type;
            double pos[3];
            ifs >> id >> type >> pos[0] >> pos[1] >> pos[2];
            types_[id-1] = type-1;
            for (int a = 0; a < 3; a++) {
                pos_[id-1][a] = pos[a];
            }
            std::getline(ifs,line);
        }
    }   

    return true;
} 

int LoadPreambleFromOUTCAR(std::ifstream& ifs, std::vector<int>& ions_per_type)
{
    std::string line;

    // Finding ions per type data by keywords
    while (line.substr(0, 18) !=
        "   ions per type =") {
        std::getline(ifs, line);
        if (ifs.eof())
            ERROR((string)"\"ions per type\" not given");
    }

    // now parsing the line, extracing ions per type
    {
        line.erase(0, 18);
        std::stringstream stream(line);
        int num;
        stream >> num;
        while (stream) {
            ions_per_type.emplace_back(num);
            stream >> num;
        }
    }


    //Reading max. number of ionic iterations
    while (line.substr(0, 11) !=
        "   NELM   =") {
        std::getline(ifs, line);
        if (ifs.eof())
            ERROR((string)"\"Number of electronic iterations\" not given");
    }

    {
        line.erase(0, 11);
        std::stringstream stream(line);
        int maxiter = -1;
        stream >> maxiter;
        return maxiter;

    }
}

int LoadPreambleFromOUTCAR(std::ifstream& ifs, 
        std::vector<int>& types,
        std::vector<int>& ions_per_type)
{
    std::string line;

    int maxiter = -1;

    while (!ifs.eof()) 
    {
        std::getline(ifs, line);

        /*if (line.substr(0,11) == "   VRHFIN =") 
        {
            string tmp(line.substr(11, 2));
            if (tmp[1] == ' ' || tmp[1] == ':') tmp = tmp.substr(0, 1);
			auto it = MENDELEEV_STRING_TO_NUM.find(tmp);
			if(it == MENDELEEV_STRING_TO_NUM.end())
				ERROR("Unrecognized chemical element " + tmp + "; while parsing '"+line+"'");

            bool is_non_unique = true;
            for (int i = 0; i < types.size(); i++) {
                if (it->second == types[i]) {
                    is_non_unique = false;
                    break;
                }
            }
            if (is_non_unique) types.push_back(it->second);
        }*/
        if (line.substr(0, 18) == "   ions per type =") 
        {
            line.erase(0, 18);
            std::stringstream stream(line);
            int num;
            stream >> num;
            while (stream) 
            {
                ions_per_type.emplace_back(num);
                stream >> num;
            }
        } 
        else if (line.substr(0, 11) == "   NELM   =") 
        {
            line.erase(0, 11);
            std::stringstream stream(line);
            stream >> maxiter;
            break;
        }
        else if (line.substr(0,8) == " POTCAR:") {
            line.erase(0, 8);
            std::stringstream stream(line);
            string word;
            stream >> word;
            while (stream) 
            {
                stream >> word;
                string word_new = "\0";
                for (int i = 0; i < word.size(); i++) {
                    if (word[i] != '_') word_new += word[i];
                    else break;
                }
                auto it = MENDELEEV_STRING_TO_NUM.find(word_new);
                if (it != MENDELEEV_STRING_TO_NUM.end()) {
                    bool is_non_unique = true;
                    for (int i = 0; i < types.size(); i++) {
                        if (it->second == types[i]) {
                            is_non_unique = false;
                            break;
                        }
                    }
                    if (is_non_unique) types.push_back(it->second);
                }
            }
        }
    }

    return maxiter;
}

void Configuration::LoadFromOUTCAR(const std::string& filename, bool relative_types)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        ERROR((string)"Can't open OUTCAR file \"" + filename + "\" for reading");

    std::vector<int> types;
    std::vector<int> ions_per_type;
    int maxiter = LoadPreambleFromOUTCAR(ifs, types, ions_per_type);

    if ( -1 == maxiter ) {
        ERROR((string)"OUTCAR reading failed.");
    }

    if (!LoadNextFromOUTCAR(ifs, maxiter))
        ERROR("No configuration read from OUTCAR");

    std::string line;
    while (line.substr(0, 36) != " total amount of memory used by VASP") {
        std::getline(ifs, line);
        if (ifs.eof())
            ERROR((string)"OUTCAR does not end with \"total amount of memory used by VASP\"");
    }

    // filling types from ions_per_type
    types_.resize(size(), 0);
    int i = 0;
    for (int k=0; k<(int)types.size(); k++)
        for (int j=0; j<ions_per_type[k]; j++)
			if (relative_types)
                	types_[i++] = k;
				else
					types_[i++] = types[k];

    if (i != size())
        ERROR((string)"Outcar-file parsing error: ions_per_type do not match the atoms count");
}

//! Assumes atom types 0, 1, ... correspond to those in POTCAR.
void Configuration::WriteVaspPOSCAR(const std::string& filename, vector<int> types_mapping) const
{
	const int LAT_WIDTH = 13; const int LAT_PRECISION = 6;
	const int POS_WIDTH = 13; const int POS_PRECISION = 6;
    const double DEFAULT_LATTICE_PERIOD = 15.0; // angstroms

	std::ofstream ofs(filename);
	if (!ofs.is_open())
		ERROR((string)"Can't open file \"" + filename + "\" for output");

	ofs.precision(16);

	ofs << "MLIP output to VASP\n";

	int _size = size();

	// calculating of number of atoms per type
	std::vector<int> type_count(types_mapping.size());
	for (int i = 0; i < _size; i++)
		for (int j = 0; j < types_mapping.size(); j++)
			if (type(i)==types_mapping[j])
				type_count[j]++;

	double scale_factor = 1.0;	

	ofs << scale_factor << '\n';
	for	(int i=0; i<3; i++)
        if (Vector3(lattice[i][0], lattice[i][1], lattice[i][2]) != Vector3(0, 0, 0))   // periodic lattice direction
		    ofs << "    " << LAT_FORMATTING << lattice[i][0] / scale_factor
			    << ' '    << LAT_FORMATTING << lattice[i][1] / scale_factor
			    << ' '    << LAT_FORMATTING << lattice[i][2] / scale_factor << '\n';
        else                                                            // non periodic direction
        {
            // 1. get periodic information data (direction of lattice vector)
            set<int> periodic_dims;
            for (int j=0; j<3; j++)
                if (lattice[j][0]!=0 && lattice[j][1]!=0 && lattice[j][2]!=0)   // periodic lattice direction
                    periodic_dims.insert(j);

            Vector3 new_dir = {0.0, 0.0, 0.0};

            if (periodic_dims.size() == 0)
            {
                if (i==0)
                    new_dir ={ 1.0, 0.0, 0.0 };
                else if (i==1)
                    new_dir ={ 0.0, 1.0, 0.0 };
                else if (i==2)
                    new_dir ={ 0.0, 0.0, 1.0 };
                else
                    ERROR("A bug detected!");
            }
            else if (periodic_dims.size() == 1)
            {
                Vector3 d1 = { 1.0, 0.0, 0.0 };
                Vector3 d2 = { 0.0, 1.0, 0.0 };
                Vector3 d3 = { 0.0, 0.0, 1.0 };
                Vector3 given_dir = lattice[*periodic_dims.begin()];// given lattice vector
                Vector3 v1, v2;                                     // directions of other lattice vectors
                
                // for v1 we choose maximal orthogonality to 'given_dir' among 'd1','d2','d3'.
                if (fabs(d1*given_dir) < fabs(d2*given_dir) &&      
                    fabs(d1*given_dir) < fabs(d3*given_dir))
                    v1 = d1;
                else if (fabs(d2*given_dir) < fabs(d1*given_dir) &&
                    fabs(d2*given_dir) < fabs(d3*given_dir))
                    v1 = d2;
                else
                    v1 = d3;
                // v2 is orthogonal to v1 and given dir
                v2 = Vector3::VectProd(given_dir, v1);              
                v2 /= v2.Norm();
                
                // new_dir is either v1 or v2
                if (i==1 || (i==2 && *periodic_dims.begin()==2)) 
                    new_dir = v1;
                else
                    new_dir = v2;
            }
            else if (periodic_dims.size() == 2)
            {
                Vector3 v1(lattice[*periodic_dims.begin()]);
                Vector3 v2(lattice[*(++periodic_dims.begin())]);
                new_dir = Vector3::VectProd(v1, v2);
                new_dir /= new_dir.Norm();
            }
            else
                ERROR("Impossible internal state detected!");

            // 2. find the minimal and maximal coordinate
            double x_min = HUGE_DOUBLE;
            double x_max = -HUGE_DOUBLE;
            for (Vector3 x : pos_)
            {
                x_min = __min(x*new_dir, x_min);
                x_max = __max(x*new_dir, x_max);
            }
            // 3. compose and write the fixed lattice vector
            Vector3 new_lat_vec = new_dir * (DEFAULT_LATTICE_PERIOD + x_max-x_min);
            ofs << "    " << LAT_FORMATTING << new_lat_vec[0] / scale_factor
                << ' '    << LAT_FORMATTING << new_lat_vec[1] / scale_factor
                << ' '    << LAT_FORMATTING << new_lat_vec[2] / scale_factor << '\n';
        }

	int meet_atoms = 0;
	for (int t = 0; t < types_mapping.size(); t++)
	{
		ofs << ' ' << type_count[t];
		meet_atoms+=type_count[t];
		if (meet_atoms==_size)
			break;
		else if (meet_atoms > _size)
			ERROR("Something is wrong with the types");
	}
	ofs << '\n';

    set<int> fixed_atoms;
    if (features.count("relax:fixed_atoms") > 0)
    {
        ofs << "Selective dynamics\n";
        std::string strind = features.at("relax:fixed_atoms");
        while (!strind.empty())
        {
            size_t commapos = strind.find(',');
            if (commapos == -1)
            {
                fixed_atoms.insert(stoi(strind));
                break;
            }
            int ind = stoi(strind.substr(0, commapos));
            fixed_atoms.insert(ind);
            strind = strind.substr(commapos+1);
        }
    }

    ofs << "cart\n";
	for (int t = 0; t < types_mapping.size(); t++)
		for (int i = 0; i < _size; i++)
            if (type(i) == types_mapping[t])
            {
                ofs << "    " << POS_FORMATTING << pos_[i][0] / scale_factor
                    << ' '    << POS_FORMATTING << pos_[i][1] / scale_factor
                    << ' '    << POS_FORMATTING << pos_[i][2] / scale_factor;
                if (!fixed_atoms.empty())
                    if (fixed_atoms.count(i) > 0)
                        ofs << "\t F F F";
                    else
                        ofs << "\t T T T";

                ofs << '\n';
            }

	ofs.close();
}

bool Configuration::LoadDynamicsFromOUTCAR(std::vector<Configuration> &db, const std::string& filename, bool save_nonconverged, bool relative_types)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        ERROR((string)"Can't open OUTCAR file \"" + filename + "\" for reading");

    std::string line;
    std::vector<int> ions_per_type;
    std::vector<int> type_list;
	int maxiter = LoadPreambleFromOUTCAR(ifs, type_list, ions_per_type);

    if ( -1 == maxiter) {
        ERROR((string)"OUTCAR reading failed.");
    }

    MlipException err("");
    bool caught = false;
    bool status = false;
    for (Configuration cfg; ;) {
        try { status = cfg.LoadNextFromOUTCAR(ifs,maxiter); }
        catch (const MlipException &e) {
            err = e; caught = true;
        }
        if (caught || !status) break;
        // filling types from ions_per_type
        cfg.types_.resize(cfg.size(), 0);
        int i = 0;
//         for (int k = 0; k < (int)ions_per_type.size(); k++)

        //cout << (int)type_list.size() << endl;
        for (int k = 0; k < (int)type_list.size(); k++)
            for (int j = 0; j < ions_per_type[k]; j++)
				if (relative_types) 
                	cfg.types_[i++] = k;
				else
					cfg.types_[i++] = type_list[k];

        if (i != cfg.size())
            ERROR((string)"OUTCAR file parsing error: ions_per_type (" + to_string(i) + ") do not match the atoms count (" + to_string(cfg.size()) + ")");

		if ((save_nonconverged)||((!save_nonconverged)&&(cfg.features["EFS_by"] != "VASP_not_converged")))
			db.push_back(cfg);
    }

    return !caught;
}

//! loads the last configuration from OUTCAR dynamics
//! returns false if the OUTCAR is broken, but at least one configuration was read successfully
bool Configuration::LoadLastFromOUTCAR(const std::string& filename, bool relative_types)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        ERROR((string)"Can't open OUTCAR file \"" + filename + "\" for reading");

    std::string line;
    std::vector<int> ions_per_type;
    std::vector<int> type_list;
	int maxiter = LoadPreambleFromOUTCAR(ifs, type_list, ions_per_type);

    if ( -1 == maxiter) {
        ERROR((string)"OUTCAR reading failed.");
    }

    MlipException err("");
    int i = 0;
    bool caught = false;
    try 
    {
        while (LoadNextFromOUTCAR(ifs,maxiter))
            ;
        // filling types from ions_per_type
        types_.resize(size(), 0);
//         for (int k = 0; k < (int)ions_per_type.size(); k++)
        for (int k = 0; k < (int)type_list.size(); k++)
            for (int j = 0; j < ions_per_type[k]; j++)
                if (relative_types)
                	types_[i++] = k;
				else
					types_[i++] = type_list[k];
    }
    catch (const MlipException &e) {
        err = e; caught = true;
    }

    if (i != size()) {
        if(caught)
            ERROR((std::string)err.What() + "\nAlso: tried to return the latest good config,\n  but ions_per_type (" + to_string(i) + ") do not match the atoms count (" + to_string(size()) + ")\\\n");
        ERROR((string)"OUTCAR file parsing error: ions_per_type (" + to_string(i) + ") do not match the atoms count (" + to_string(size()) + ")");
    }
    return !caught;
}

//! stupid O(N^2) algorithm
double Configuration::MinDist()
{
    // is the configuration periodic in any dimension?
    bool is_periodic[3];
    for (int a = 0; a < 3; a++)
        is_periodic[a] = lattice[a][0] != 0.0
        || lattice[a][1] != 0.0
        || lattice[a][2] != 0.0;

    // length of extension in 3 directions
    int ll[3];
    for (int a = 0; a < 3; a++)
        ll[a] = is_periodic[a] ? 1 : 0;

    double mindist = 9.9e99;
    for (int ind2 = 0; ind2 < size(); ind2++)
        for (int ind = ind2; ind < size(); ind++)
            for (int i = -ll[0]; i <= ll[0]; i++)
                for (int j = -ll[1]; j <= ll[1]; j++)
                    for (int k = -ll[2]; k <= ll[2]; k++)
                        if ((ind2 != ind) || (i != 0) || (j != 0) || (k != 0)) {
                            Vector3 candidate = pos(ind);
                            for (int a = 0; a < 3; a++)
                                candidate[a] += i*lattice[0][a] + j*lattice[1][a] + k*lattice[2][a];
                            mindist = __min(mindist, distance(pos(ind2), candidate));
                        }
    return mindist;
}

map<pair<int, int>, double> Configuration::GetTypeMindists()
{
    auto typeset = GetTypes();
    int dim = (int)typeset.size();

    map<pair<int, int>, double> mindistmap;
    for (int type1 : typeset)
        for (int type2 : typeset)
            mindistmap[make_pair(type1, type2)] = HUGE_DOUBLE;

    // is the configuration periodic in any dimension?
    bool is_periodic[3];
    for (int a = 0; a < 3; a++)
        is_periodic[a] = lattice[a][0] != 0.0
        || lattice[a][1] != 0.0
        || lattice[a][2] != 0.0;

    // length of extension in 3 directions
    int ll[3];
    for (int a = 0; a < 3; a++)
        ll[a] = is_periodic[a] ? 1 : 0;

    double mindist = 9.9e99;
    for (int ind2 = 0; ind2 < size(); ind2++)
        for (int ind = ind2; ind < size(); ind++)
            for (int i = -ll[0]; i <= ll[0]; i++)
                for (int j = -ll[1]; j <= ll[1]; j++)
                    for (int k = -ll[2]; k <= ll[2]; k++)
                        if ((ind2 != ind) || (i != 0) || (j != 0) || (k != 0)) 
                        {
                            Vector3 candidate = pos(ind);
                            for (int a = 0; a < 3; a++)
                                candidate[a] += i*lattice[0][a] + j*lattice[1][a] + k*lattice[2][a];
                            double& dist = mindistmap[make_pair(types_[ind], types_[ind2])];
                            dist = __min(dist, distance(pos(ind2), candidate));
                        }

    for (int i : typeset)
        for (int j : typeset)
        {
            double& dist = mindistmap[make_pair(i, j)];
            dist = __min(dist, mindistmap[make_pair(j, i)]);
        }

    string mindistr = "{";
    for (int i : typeset)
        for (int j : typeset)
            if (i <= j)
                mindistr += "("+to_string(i)+","+to_string(j)+"):" + to_string(mindistmap[make_pair(i, j)]) + ",";

    mindistr.resize(mindistr.size()-1);
    features["mindists"] = mindistr+'}';

    return mindistmap;
}

double Configuration::MinMagMom()
{
    double min_mag_mom = 9e+99;

    for (int i = 0; i < size(); i++) {
        if (fabs(magmom_[i][1]) > 1e-6 || fabs(magmom_[i][2]) > 1e-6) 
            ERROR("Magnetic moment should be a scalar in the current MLIP version!");
        double curr_mag_mom = 0;
        for (int a = 0; a < 3; a++)
            curr_mag_mom = magmom_[i][0];

        if (curr_mag_mom < min_mag_mom) min_mag_mom = curr_mag_mom;
    }

    return min_mag_mom;
}

double Configuration::MaxMagMom()
{
    double max_mag_mom = 0;

    for (int i = 0; i < size(); i++) {
        if (fabs(magmom_[i][1]) > 1e-6 || fabs(magmom_[i][2]) > 1e-6) 
            ERROR("Magnetic moment should be a scalar in the current MLIP version!");
        double curr_mag_mom = 0;
        for (int a = 0; a < 3; a++)
            curr_mag_mom = magmom_[i][0];

        if (curr_mag_mom > max_mag_mom) max_mag_mom = curr_mag_mom;
    }

    return max_mag_mom;
}

vector<Configuration> LoadCfgs(const string& filename, const int max_count)
{
    vector<Configuration> cfgs;
    ifstream ifs(filename, ios::binary);
    if (!ifs.is_open())
        ERROR("Can't open file \"" + filename + "\" for reading configurations");
    int cntr = 0;
    for (Configuration cfg; cntr < max_count && cfg.Load(ifs); cntr++)
        cfgs.emplace_back(cfg);
    return cfgs;
}

vector<Configuration> MPI_LoadCfgs(const string& filename, const int max_count)
{
    vector<Configuration> cfgs;
    ifstream ifs(filename, ios::binary);
    if (!ifs.is_open())
        ERROR("Can't open file \"" + filename + "\" for reading configurations");
    int cntr = 0;
    for (Configuration cfg; cntr < max_count && cfg.Load(ifs); cntr++)
        if (cntr % mpi.size == mpi.rank)
            cfgs.emplace_back(cfg);
    if ((cntr % mpi.size != 0) && mpi.rank >= (cntr % mpi.size))
            cfgs.emplace_back(Configuration());
    return cfgs;
}

void Configuration::ToAtomicUnits()
{
    const double length_unit = 1.889725989; // (becomes Bohr)
    const double energy_unit = 1.0/27.211385;   // (becomes Hartree)
    lattice *= length_unit;
    int size = (int)pos_.size();
    for (int i = 0; i < size; i++)
        pos(i) *= length_unit;
    energy *= energy_unit;
    if (has_forces()) {
        for (int i = 0; i < size; i++)
            force(i) *= energy_unit/length_unit;
    }
    if (has_stresses())
        stresses *= energy_unit;
}

void Configuration::FromAtomicUnits()
{
    const double length_unit = 1.0/1.889725989;
    const double energy_unit = 27.211385;
    lattice *= length_unit;
    int size = (int)pos_.size();
    for (int i = 0; i < size; i++)
        pos(i) *= length_unit;
    energy *= energy_unit;
    if (has_forces()) {
        for (int i = 0; i < size; i++)
            force(i) *= energy_unit / length_unit;
    }
    if (has_stresses())
        stresses *= energy_unit;
}

void Configuration::WriteLammpsDatafile(const std::string& filename) const
{
    Configuration cfg(*this);

   bool is_periodic[3];
    for (int a =0; a < 3; a++)
        is_periodic[a] = cfg.lattice[a][0] != 0.0
        || cfg.lattice[a][1] != 0.0
        || cfg.lattice[a][2] != 0.0;

    if (is_periodic[0] && is_periodic[1] && is_periodic[2]) {

    Matrix3 Q;
    Matrix3 L;
    cfg.lattice.transpose().QRdecomp(Q, L);
    cfg.Deform(Q);

    cfg.lattice[0][1] = 0.0;
    cfg.lattice[0][2] = 0.0;
    cfg.lattice[1][2] = 0.0;

    cfg.MoveAtomsIntoCell();

    ofstream ofs(filename);

    int n_at_types = 0; // number of atom types
    for (int i = 0; i < cfg.size(); i++)
        if (cfg.type(i) > n_at_types) n_at_types = cfg.type(i);
    n_at_types++;

    ofs << "\n";
    ofs << std::fixed << std::setprecision(9) << 0.0;
    ofs << " " << cfg.lattice[0][0] << " xlo xhi\n";

    ofs << std::fixed << std::setprecision(9) << 0.0;
    ofs << " " << cfg.lattice[1][1] << " ylo yhi\n";

    ofs << std::fixed << std::setprecision(9) << 0.0;
    ofs << " " << cfg.lattice[2][2] << " zlo zhi\n";

    ofs << std::fixed << std::setprecision(9);
    ofs << cfg.lattice[1][0] << " ";
    ofs << std::fixed << std::setprecision(9);
    ofs << cfg.lattice[2][0] << " ";
    ofs << std::fixed << std::setprecision(9);
    ofs << cfg.lattice[2][1] << " xy xz yz\n\n";

    ofs << cfg.size() << " atoms\n\n";
    ofs << n_at_types << " atom types\n\n";

    ofs << "Atoms\n\n";

    for (int i = 0; i < cfg.size(); i++) {
        ofs << std::right << std::setw(6) << i + 1;
        ofs << std::right << std::setw(6) << cfg.type(i) + 1;
        ofs << std::right << std::setw(20) << std::fixed << std::setprecision(8) << cfg.pos(i, 0);
        ofs << std::right << std::setw(20) << cfg.pos(i, 1);
        ofs << std::right << std::setw(20) << cfg.pos(i, 2) << "\n";
    }

    ofs.close();

    }
    else {
        ofstream ofs(filename);

        int n_at_types = 0; // number of atom types
        for (int i = 0; i < cfg.size(); i++)
            if (cfg.type(i) > n_at_types) n_at_types = cfg.type(i);
        n_at_types++;

        ofs << "\n";
        ofs << std::fixed << std::setprecision(9) << -100.0;
        ofs << " " << 100.0 << " xlo xhi\n";

        ofs << std::fixed << std::setprecision(9) << -100.0;
        ofs << " " << 100.0 << " ylo yhi\n";

        ofs << std::fixed << std::setprecision(9) << -100.0;
        ofs << " " << 100.0 << " zlo zhi\n";

        ofs << std::fixed << std::setprecision(9);
        ofs << 0 << " ";
        ofs << std::fixed << std::setprecision(9);
        ofs << 0 << " ";
        ofs << std::fixed << std::setprecision(9);
        ofs << 0 << " xy xz yz\n\n";

        ofs << cfg.size() << " atoms\n\n";
        ofs << n_at_types << " atom types\n\n";

        ofs << "Atoms\n\n";

        for (int i = 0; i < cfg.size(); i++) {
            ofs << std::right << std::setw(6) << i + 1;
            ofs << std::right << std::setw(6) << cfg.type(i) + 1;
            ofs << std::right << std::setw(20) << std::fixed << std::setprecision(8) << cfg.pos(i, 0);
            ofs << std::right << std::setw(20) << cfg.pos(i, 1);
            ofs << std::right << std::setw(20) << cfg.pos(i, 2) << "\n";
        }

        ofs.close();
    }

}
