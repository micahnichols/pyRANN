/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Ivan Novikov
 */

#include "magnetic_moments.h"
#include <algorithm>

using namespace std;

void MagneticMoments::SkipAugmentationOccupancies(ifstream& ifs, int n_atoms)
{
    string foo_str;
    double foo_double;
    int foo_int, n_occupancies;

    for (int i = 0; i < n_atoms; i++) {
        ifs >> foo_str >> foo_str >> foo_int >> n_occupancies;
        for (int j = 0; j < n_occupancies; j++) 
            ifs >> foo_double;
        //cout << foo_double << endl;
    }

    if (!is_collinear) {
        for (int i = 0; i < n_atoms; i++) {
            ifs >> foo_str >> foo_str >> foo_str >> foo_str >> foo_int >> n_occupancies;
            for (int j = 0; j < n_occupancies; j++) 
                ifs >> foo_double;
            //cout << foo_double << endl;
        }
    }

    //skip initial magnetic moments
    double init_mag_mom;
    for (int i = 0; i < n_atoms; i++) 
        ifs >> init_mag_mom;

    //we do not need to read number of nodes in mesh along each direction one more time
    ifs >> foo_int >> foo_int >> foo_int;
}

void MagneticMoments::LoadDensityFromCHGCAR(string& chgcar_fnm, int n_at_types)
{
    ifstream ifs(chgcar_fnm);

    //skip configuration
    string foo_str;
    double foo_double;
        
    getline(ifs, foo_str);
    ifs >> foo_str;
    for (int i = 0; i < 3; i++)
        ifs >> foo_double >> foo_double >> foo_double;

    vector<int> n_at(n_at_types);
    int n_atoms = 0;
    for (int i = 0; i < n_at_types; i++)
        ifs >> foo_str;
    for (int i = 0; i < n_at_types; i++) {
        ifs >> n_at[i];
        n_atoms += n_at[i];
    }
    ifs >> foo_str;

    for (int i = 0; i < n_atoms; i++)
        ifs >> foo_double >> foo_double >> foo_double;

    //read number of nodes in mesh along each direction
    ifs >> mesh_size[0] >> mesh_size[1] >> mesh_size[2];
    int nx = mesh_size[0];
    int ny = mesh_size[1];
    int nz = mesh_size[2];

    //skip charge density
    for (int k = 0; k < nz; k++) 
        for (int j = 0; j < ny; j++) 
            for (int i = 0; i < nx; i++) 
                ifs >> foo_double;

    if (!is_collinear) n_directions = 3;

    for (int a = 0; a < n_directions; a++) {

        //skip augmentation occupancies and magnetic moments
        SkipAugmentationOccupancies(ifs, n_atoms);

        //read magnetic density
        density.resize(n_directions*nx*ny*nz);

        for (int k = 0; k < nz; k++) 
            for (int j = 0; j < ny; j++) 
                for (int i = 0; i < nx; i++) 
                    ifs >> density[a*nz*ny*nx+k*ny*nx+j*nx+i];

    }

    //cout << density[n_directions*nx*ny*nz-1] << endl;
}

bool MagneticMoments::IsPointInsideExtendedRectangularParallelepiped(Vector3& point, vector<Vector3>& min_max_vertex) {

    bool is_between_x = point[0] >= min_max_vertex[0][0] && point[0] <= min_max_vertex[1][0];
    bool is_between_y = point[1] >= min_max_vertex[0][1] && point[1] <= min_max_vertex[1][1];
    bool is_between_z = point[2] >= min_max_vertex[0][2] && point[2] <= min_max_vertex[1][2];

    if (is_between_x && is_between_y && is_between_z) return true;
    else return false;

}

void MagneticMoments::GenerateRectangularParallelepiped(Configuration& cfg, std::vector<Vector3>& min_max_vertex) {

    vector<Vector3> vertex(8);
    FillWithZero(vertex);

    for (int a = 0; a < 3; a++) {
        vertex[1][a] = cfg.lattice[0][a];
        vertex[2][a] = cfg.lattice[1][a];
        vertex[3][a] = cfg.lattice[0][a]+cfg.lattice[1][a];
        vertex[4][a] = cfg.lattice[2][a];
        vertex[5][a] = cfg.lattice[2][a]+cfg.lattice[0][a];
        vertex[6][a] = cfg.lattice[2][a]+cfg.lattice[1][a];
        vertex[7][a] = cfg.lattice[2][a]+cfg.lattice[0][a]+cfg.lattice[1][a];
    }

    //for (int i = 0; i < 8; i++) 
    //    cout << vertex[i][0] << " " << vertex[i][1] << " " << vertex[i][2] << endl;

    min_max_vertex[0] = vertex[0];
    min_max_vertex[1] = vertex[0];

    for (int i = 1; i < 8; i++) {
        for (int a = 0; a < 3; a++) {
            if (min_max_vertex[0][a] > vertex[i][a]) min_max_vertex[0][a] = vertex[i][a];
            if (min_max_vertex[1][a] < vertex[i][a]) min_max_vertex[1][a] = vertex[i][a];
        }
    }

    //for (int i = 0; i < 2; i++) 
    //    cout << min_max_vertex[i][0] << " " << min_max_vertex[i][1] << " " << min_max_vertex[i][2] << endl;

    //cout << (min_max_vertex[1][0] - min_max_vertex[0][0])*(min_max_vertex[1][1] - min_max_vertex[0][1])*(min_max_vertex[1][2] - min_max_vertex[0][2]) << endl;

}

void MagneticMoments::ExtendRectangularParallelepiped(Configuration& cfg, std::vector<Vector3>& min_max_vertex) {

    GenerateRectangularParallelepiped(cfg, min_max_vertex);

    //we extend each side of rectangular parallelepiped on the same quantity 4*max_nearest_neigh_dist
    double max_nearest_neigh_dist = 0;
    for (int i = 0; i < nearest_neigh_dist.size(); i++)
        if (nearest_neigh_dist[i] > max_nearest_neigh_dist) max_nearest_neigh_dist = nearest_neigh_dist[i];

    double extension = 4*max_nearest_neigh_dist;
	
    for (int a = 0; a < 3; a++) {
        min_max_vertex[0][a] = -extension;
        min_max_vertex[1][a] += extension;
    }

    //for (int i = 0; i < 8; i++) 
    //    cout << vertex[i][0] << " " << vertex[i][1] << " " << vertex[i][2] << endl;

}

void MagneticMoments::CreateGhostAtoms(Configuration& cfg, Neighborhood& ghost_atom) {

    vector<Vector3> min_max_vertex(2);
    ExtendRectangularParallelepiped(cfg, min_max_vertex);

    bool is_inside = true;
    ghost_atom.count = 0;
    for (int l = 1; is_inside; l++) {
        is_inside = false;

    bool is_periodic[3];
    for (int a = 0; a < 3; a++)
        is_periodic[a] = cfg.lattice[a][0] != 0.0 || cfg.lattice[a][1] != 0.0 || cfg.lattice[a][2] != 0.0;

    int ll[3];
    for (int a = 0; a < 3; a++)
        ll[a] = is_periodic[a] ? l : 0;

    for (int i = -ll[0]; i <= ll[0]; i++) {
        for (int j = -ll[1]; j <= ll[1]; j++) {
            for (int k = -ll[2]; k <= ll[2]; k++) {
                if (std::max(std::max(abs(i), abs(j)), abs(k)) == l) {
                    for (int ind = 0; ind < cfg.size(); ind++) {
                        Vector3 candidate = cfg.pos(ind);
                        for (int a = 0; a < 3; a++)
                            candidate[a] += i*cfg.lattice[0][a] + j*cfg.lattice[1][a]+ 
                            k*cfg.lattice[2][a];
                            is_inside = IsPointInsideExtendedRectangularParallelepiped(candidate, min_max_vertex);
                            if (is_inside) {
                                ghost_atom.count++;
                                ghost_atom.inds.push_back(ind);
                                ghost_atom.vecs.emplace_back();
                                ghost_atom.vecs.back()[0] = candidate[0];
                                ghost_atom.vecs.back()[1] = candidate[1];
                                ghost_atom.vecs.back()[2] = candidate[2];
                                ghost_atom.types.push_back(cfg.type(ind));
                            }
                        }
                    }
                }
            }
        }
    }

}

void MagneticMoments::CalculateWeights(Configuration& cfg) {

    Neighborhood ghost_atom;
    weight.resize(mesh_size[0]*mesh_size[1]*mesh_size[2], cfg.size());
    CreateGhostAtoms(cfg, ghost_atom);

    //cout << ghost_atom.count << endl;

    Matrix3 h;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            h[a][b] = cfg.lattice[a][b] / mesh_size[a];

    for (int k = 0; k < mesh_size[2]; k++) {
        for (int j = 0; j < mesh_size[1]; j++) {
            for (int i = 0; i < mesh_size[0]; i++) { 
                Vector3 x;
                for (int b = 0; b < 3; b++) 
                    x[b] = i*h[0][b] + j*h[1][b] + k*h[2][b];
                for (int ind = 0; ind < ghost_atom.count; ind++) {
                    double dist = distance(x, ghost_atom.vecs[ind]);
                    double power = dist * dist / (2 * nearest_neigh_dist[ghost_atom.types[ind]] * 
                                   nearest_neigh_dist[ghost_atom.types[ind]]);
                    weight(k*mesh_size[1]*mesh_size[0]+j*mesh_size[0]+i,ghost_atom.inds[ind]) += exp(-power);
                }
                for (int ind = 0; ind < cfg.size(); ind++) {
                    double dist = distance(x, cfg.pos(ind));
                    double power = dist * dist / (2 * nearest_neigh_dist[cfg.type(ind)] * 
                                   nearest_neigh_dist[cfg.type(ind)]);
                    weight(k*mesh_size[1]*mesh_size[0]+j*mesh_size[0]+i,ind) += exp(-power);
                }
            }
        }
    }

    Array1D rho_sum(mesh_size[2]*mesh_size[1]*mesh_size[0]);
    FillWithZero(rho_sum);

    for (int k = 0; k < mesh_size[2]; k++) {
        for (int j = 0; j < mesh_size[1]; j++) {
            for (int i = 0; i < mesh_size[0]; i++) { 
                int ijk = k*mesh_size[1]*mesh_size[0]+j*mesh_size[0]+i;
                for (int ind = 0; ind < cfg.size(); ind++) 
                    rho_sum[ijk] += weight(ijk,ind);
            }
        }
    }

    for (int k = 0; k < mesh_size[2]; k++) {
        for (int j = 0; j < mesh_size[1]; j++) {
            for (int i = 0; i < mesh_size[0]; i++) {
                int ijk = k*mesh_size[1]*mesh_size[0]+j*mesh_size[0]+i;
                for (int ind = 0; ind < cfg.size(); ind++) 
                    weight(ijk,ind) /= rho_sum[ijk];
            }
        }
    }

}

void MagneticMoments::Calculate(Configuration& cfg) {

    Matrix3 h;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            h[a][b] = cfg.lattice[a][b] / mesh_size[a];

    cfg.has_magmom(true);

    if (is_collinear) {

        for (int ind = 0; ind < cfg.size(); ind++) { 
            for (int k = 0; k < mesh_size[2]; k++) {
                for (int j = 0; j < mesh_size[1]; j++) {
                    for (int i = 0; i < mesh_size[0]; i++) {
                        int ijk = k*mesh_size[1]*mesh_size[0]+j*mesh_size[0]+i;
                        cfg.magmom(ind)[0] += weight(ijk,ind) * density[ijk] * h.det();
                    }
                }
            }
        }

        for (int ind = 0; ind < cfg.size(); ind++) 
            cfg.magmom(ind)[0] /= cfg.lattice.det();

        cfg.features["COLLINEAR"] = "YES";

    }
    else {

        for (int a = 0; a < 3; a++) {
            for (int ind = 0; ind < cfg.size(); ind++) { 
                for (int k = 0; k < mesh_size[2]; k++) {
                    for (int j = 0; j < mesh_size[1]; j++) {
                        for (int i = 0; i < mesh_size[0]; i++) {
                            int ijk_weight = k*mesh_size[1]*mesh_size[0]+j*mesh_size[0]+i;
                            int ijk_density = a*mesh_size[2]*mesh_size[1]*mesh_size[0]+ijk_weight;
                            cfg.magmom(ind)[a] += weight(ijk_weight,ind) * density[ijk_density] * h.det();
                        }
                    }
                }
            }
        }

        for (int ind = 0; ind < cfg.size(); ind++) { 
            cfg.magmom(ind) /= cfg.lattice.det();
            cfg.magmom(ind)[1] = 0;
            cfg.magmom(ind)[2] = 0;
        }

        cfg.features["COLLINEAR"] = "NO";

    }

}
