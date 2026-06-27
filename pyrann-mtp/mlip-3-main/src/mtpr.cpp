/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include <algorithm>
#include <random>
#include "mtpr.h"


using namespace std;


void MLMTPR::Load(const string& filename)
{
    alpha_count = 0;

    ifstream ifs(filename);
    if (!ifs.is_open())
        ERROR((string)"Cannot open " + filename);

    char tmpline[1000];
    string tmpstr;

    ifs.getline(tmpline, 1000);
    int len = (int)((string)tmpline).length();
    if (tmpline[len - 1] == '\r')   // Ensures compatibility between Linux and Windows line endings
        tmpline[len - 1] = '\0';

    if ((string)tmpline != "MTP")
        ERROR("Can read only MTP format potentials");

        // version reading block
        ifs.getline(tmpline, 1000);
        len = (int)((string)tmpline).length();
        if (tmpline[len - 1] == '\r')   // Ensures compatibility between Linux and Windows line endings
            tmpline[len - 1] = '\0';
        if ((string)tmpline != "version = 1.1.0")
            ERROR("MTP file must have version \"1.1.0\"");

        // name/description reading block
        ifs >> tmpstr;
        if (tmpstr == "potential_name") // optional 
        {
            ifs.ignore(2);
            ifs >> pot_desc;
            ifs >> tmpstr;
        }

        if (tmpstr == "scaling") // optional 
        {
            ifs.ignore(2);
            ifs >> scaling;
            ifs >> tmpstr;
        }

        if (tmpstr == "species_count")
        {
            ifs.ignore(2);
            ifs >> species_count;
            ifs >> tmpstr;

        }
        else
            species_count=0;

    if (tmpstr == "potential_tag")
    {
        getline(ifs, tmpstr);
        ifs >> tmpstr;
    }

    if (tmpstr != "radial_basis_type")
        ERROR("Error reading .mtp file \""+filename+"\"");
    ifs.ignore(2);
    ifs >> rbasis_type;
    
    if (rbasis_type == "BChebyshev") {
        p_RadialBasis = new Basis_Chebyshev(ifs);
    }
    else if (rbasis_type == "BChebyshev_repuls") {
        p_RadialBasis = new Basis_Chebyshev_repuls(ifs);
    }
    else if (rbasis_type == "RBChebyshev") {
        p_RadialBasis = new RB_Chebyshev(ifs);
    }
    else if (rbasis_type == "RBChebyshev_repuls") {
        p_RadialBasis = new RB_Chebyshev_repuls(ifs);
    }
    else if (rbasis_type == "RBShapeev") {
        p_RadialBasis = new Basis_Shapeev(ifs);
        Warning("Non-linear MTP is not fully compatible with RBShapeev");
    }
    else if (rbasis_type == "RBTaylor") {
        p_RadialBasis = new Basis_Taylor(ifs);
    }
    else
        ERROR("Wrong radial basis type");

    // We do not need double scaling
    if (p_RadialBasis->scaling != 1.0) {
        scaling *= p_RadialBasis->scaling;
        p_RadialBasis->scaling = 1.0;
    }

    ifs >> tmpstr;
    if (tmpstr == "magnetic_basis_type") {
        ifs.ignore(2);
        ifs >> mbasis_type;
        if (mbasis_type == "BChebyshev") {
            p_MagneticBasis = new Basis_Chebyshev(ifs);
        }
        else if (mbasis_type == "BChebyshev_repuls") {
            p_MagneticBasis = new Basis_Chebyshev_repuls(ifs);
        }
        else if (mbasis_type == "RBChebyshev") {
            p_MagneticBasis = new RB_Chebyshev(ifs);
        }
        else if (mbasis_type == "RBChebyshev_repuls") {
            p_MagneticBasis = new RB_Chebyshev_repuls(ifs);
        }
        else if (mbasis_type == "RBShapeev") {
            p_MagneticBasis = new Basis_Shapeev(ifs);
        }
        else if (mbasis_type == "RBTaylor") {
            p_MagneticBasis = new Basis_Taylor(ifs);
        }
        else if (mbasis_type == "") {
            ;
        }
        else
            ERROR("Wrong magnetic basis type");

        mbasis_size = p_MagneticBasis->size;

        // We do not need double scaling
        /*if (p_MagneticBasis->scaling != 1.0) {
            scaling *= p_MagneticBasis->scaling;
            p_MagneticBasis->scaling = 1.0;
        }*/
        ifs >> tmpstr;
    }

    if (tmpstr != "radial_funcs_count")
        ERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> radial_func_count;

    //Radial coeffs initialization
    int pairs_count = species_count*species_count;           //number of species pairs
    int mbasis_size2 = mbasis_size*mbasis_size;              //squared magnetic basis size
    //cout << mbasis_size2 << endl;

    char foo = ' ';

    regression_coeffs.resize(pairs_count*radial_func_count*(p_RadialBasis->size*mbasis_size2));

    //cout << "MTPR pairs_count = " << pairs_count << endl;
    //if(inited) cout << "MTPR inited" << endl; else cout << "MTPR NOT inited" << endl;
    //if (pairs_count>0)
    //    inited = true;
    
    int type1 = -1, type2 = -1;
    string tmp = "";

    ifs >> tmpstr;
    if (tmpstr == "radial_coeffs")
    {
        for (int s1 = 0; s1 < species_count; s1++)					//loads the radial coefficients for each pair of atomic species
            for (int s2 = 0; s2 < species_count; s2++)
            {
                //ifs >>foo >> foo >> foo;
                ifs >> tmpstr;
				
                if (tmpstr[1] == '-') {                        //first atomic type
                    type1 = stoi(tmpstr.substr(0, 1));
                    type2 = stoi(tmpstr.substr(2, tmpstr.length()-2));
                }
                else if (tmpstr[2] == '-') {					//second atomic type
                    type1 = stoi(tmpstr.substr(0, 2));
                    type2 = stoi(tmpstr.substr(3, tmpstr.length() - 3));
                }

                if (std::find(atomic_numbers.begin(), atomic_numbers.end(), type1) == atomic_numbers.end())  //add atomic type to the list if not done yet
                    atomic_numbers.push_back(type1);

                if (std::find(atomic_numbers.begin(), atomic_numbers.end(), type2) == atomic_numbers.end())  //add atomic type to the list if not done yet
                    atomic_numbers.push_back(type2);

                double t;

                for (int i = 0; i < radial_func_count; i++)
                {
                    ifs >> foo;

                    for (int j = 0; j < p_RadialBasis->size; j++) { //reading the coefficients from file
                        for (int ii = 0; ii < mbasis_size; ii++) {
                            for (int jj = 0; jj < mbasis_size; jj++) {
                                ifs >> t >> foo;
                                regression_coeffs[(s1*species_count+s2)*radial_func_count*(p_RadialBasis->size*mbasis_size2) +
                                    mbasis_size2*(i*(p_RadialBasis->size) + j) + ii*mbasis_size + jj] = t;
                            }
                        }
                    }
                }
            }
        ifs >> tmpstr;
		inited = true;
    }
    else
    {
		inited = false;

        for(int i=0;i<species_count;i++)
		atomic_numbers.push_back(i);

        //cout << "Radial coeffs not found, initializing defaults" << endl;
        regression_coeffs.resize(species_count*species_count*radial_func_count*(p_RadialBasis->size*mbasis_size2));

        for (pairs_count = 0; pairs_count < species_count*species_count; pairs_count++)
            for (int i = 0; i < radial_func_count; i++)
            {
                for (int j = 0; j < p_RadialBasis->size; j++) {
                    for (int ii = 0; ii < mbasis_size; ii++) {
                        for (int jj = 0; jj < mbasis_size; jj++) {
                    regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->size*mbasis_size2) +
                    mbasis_size2*(i*(p_RadialBasis->size) + j) + ii*mbasis_size + jj] = 1e-6 + i*1e-7 + j*1e-7;
                        }
                    }
               }

                //TODO: implement shear on main diagonal
                    for (int ii = 0; ii < mbasis_size; ii++) {
                        for (int jj = 0; jj < mbasis_size; jj++) {
                regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->size*mbasis_size2) +
                    mbasis_size2*(i*(p_RadialBasis->size) + min(i, p_RadialBasis->size)) + ii*mbasis_size + jj] = 1e-3*(pairs_count+1)+ i*1e-4;
                        }
                    }
            }
    }

    if (tmpstr != "alpha_moments_count")
        ERROR("Error reading .mtp file \""+filename+"\"");
    ifs.ignore(2);
    ifs >> alpha_moments_count;
    if (ifs.fail())
        ERROR("Error reading .mtp file \""+filename+"\"");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_basic_count")
        ERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> alpha_index_basic_count;
    if (ifs.fail())
        ERROR("Error reading .mtp file \""+filename+"\"");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_basic")
        ERROR("Error reading .mtp file \""+filename+"\"");
    ifs.ignore(4);

    alpha_index_basic = new int[alpha_index_basic_count][4];    
    if (alpha_index_basic == nullptr)
        ERROR("Memory allocation error");

    int radial_func_max = -1;
    for (int i = 0; i < alpha_index_basic_count; i++)
    {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> alpha_index_basic[i][0] >> tmpch >> alpha_index_basic[i][1] >> tmpch >> alpha_index_basic[i][2] >> tmpch >> alpha_index_basic[i][3];
        if (ifs.fail())
            ERROR("Error reading .mtp file \""+filename+"\"");

        if (alpha_index_basic[i][0]>radial_func_max)
            radial_func_max = alpha_index_basic[i][0];
    }

    if (radial_func_max!=radial_func_count-1)
        ERROR("Wrong number of radial functions specified in \""+filename+"\"");

    ifs.ignore(1000, '\n');

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_times_count")
        ERROR("Error reading .mtp file \""+filename+"\"");
    ifs.ignore(2);
    ifs >> alpha_index_times_count;
    if (ifs.fail())
        ERROR("Error reading .mtp file \""+filename+"\"");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_times")
        ERROR("Error reading .mtp file \""+filename+"\"");
    ifs.ignore(4);

    alpha_index_times = new int[alpha_index_times_count][4];    
    if (alpha_index_times == nullptr)
        ERROR("Memory allocation error \""+filename+"\"");

    for (int i = 0; i < alpha_index_times_count; i++)
    {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> alpha_index_times[i][0] >> tmpch >> alpha_index_times[i][1] >> tmpch >> alpha_index_times[i][2] >> tmpch >> alpha_index_times[i][3];
        if (ifs.fail())
            ERROR("Error reading .mtp file \""+filename+"\"");
    }

    ifs.ignore(1000, '\n');

    ifs >> tmpstr;
    if (tmpstr != "alpha_scalar_moments")
        ERROR("Error reading .mtp file \""+filename+"\"");
    ifs.ignore(2);
    ifs >> alpha_scalar_moments;
    if (alpha_index_times_count < 0)
        ERROR("Error reading .mtp file \""+filename+"\"");

    alpha_moment_mapping = new int[alpha_scalar_moments];
    if (alpha_moment_mapping == nullptr)
        ERROR("Memory allocation error");

    ifs >> tmpstr;
    if (tmpstr != "alpha_moment_mapping")
        ERROR("Error reading .mtp file \""+filename+"\"");
    ifs.ignore(4);
    for (int i = 0; i < alpha_scalar_moments; i++)
    {
        char tmpch = ' ';
        ifs >> alpha_moment_mapping[i] >> tmpch;
        if (ifs.fail())
            ERROR("Error reading .mtp file \""+filename+"\"");
    }
    ifs.ignore(1000, '\n');

    alpha_count = alpha_scalar_moments + 1;

    //Reading linear coeffs
    ifs >> tmpstr;
    if (tmpstr != "species_coeffs")
    {
        inited = false;
        //cout << "Linear coeffs not found, initializing defaults, species_count = " << species_count << endl;
		regression_coeffs.resize(Rsize() + alpha_count + species_count - 1);
        for (int i = 0; i < alpha_count + species_count - 1; i++)
            linear_coeffs(i) = 1e-3;
    }
    else
    {
        ifs.ignore(4);
        regression_coeffs.resize(Rsize() +species_count);
        for (int i = 0; i < species_count; i++)
            ifs >> linear_coeffs(i) >> foo;

        ifs >> tmpstr;

        if (tmpstr != "moment_coeffs")
            ERROR("Cannot read linear coeffs from \""+filename+"\"");

        ifs.ignore(2);
        regression_coeffs.resize(Rsize() + alpha_count + species_count - 1);
        ifs.ignore(10, '{');

        for (int i = 0; i < alpha_count - 1; i++)
            ifs >> linear_coeffs(i + species_count) >> foo;
    }
    
    MemAlloc();
}

void MLMTPR::Save(const string& filename)
{
    ofstream ofs(filename);
    ofs.setf(ios::scientific);
    ofs.precision(15);

    ofs << "MTP\n";
    ofs << "version = 1.1.0\n";
    ofs << "potential_name = " << pot_desc << endl;
    if(scaling != 1.0)
        ofs << "scaling = " << scaling << endl;
    if (inited)
        ofs << "species_count = " << species_count << endl;
    ofs << "potential_tag = " << "" << endl;
    p_RadialBasis->WriteBasis(ofs);
    if (p_MagneticBasis) {
        ofs << "magnetic_basis_type = " << p_MagneticBasis->GetRBTypeString() << '\n';
        //if (scaling != 1.0)
        //    ofs << "\tscaling = " << p_MagneticBasis->scaling << '\n';
        ofs << "\tmin_val = " << p_MagneticBasis->min_val << '\n'
            << "\tmax_val = " << p_MagneticBasis->max_val << '\n'
            << "\tbasis_size = " << p_MagneticBasis->size << '\n';
    }
    ofs << "\tradial_funcs_count = " << radial_func_count << '\n';
    ofs << "\tradial_coeffs" << '\n';

    int q = 0;
    if (inited)
    {
    for (int i = 0; i < species_count; i++)
        for (int j = 0; j < species_count; j++)
        {
            ofs <<"\t\t"<< atomic_numbers[i] << "-" << atomic_numbers[j] << "\n";
            for (int k = 0; k < radial_func_count; k++)
            {
                ofs << "\t\t\t{";
                for (int l = 0; l < p_RadialBasis->size; l++)
                {
                    for (int ii = 0; ii < mbasis_size; ii++)
                    {
                        for (int jj = 0; jj < mbasis_size; jj++)
                        {
                            ofs << regression_coeffs[q++];
                            if (l*mbasis_size*mbasis_size+ii*mbasis_size+jj != p_RadialBasis->size*mbasis_size*mbasis_size - 1)
                                ofs << ", ";
                            else
                                ofs << "}" << endl;
                        }
                    }
                }
            }
        }
    }

    ofs << "alpha_moments_count = " << alpha_moments_count << '\n';
    ofs << "alpha_index_basic_count = " << alpha_index_basic_count << '\n';
    ofs << "alpha_index_basic = {";
    for (int i = 0; i < alpha_index_basic_count; i++)
    {
        ofs << '{'
            << alpha_index_basic[i][0] << ", "
            << alpha_index_basic[i][1] << ", "
            << alpha_index_basic[i][2] << ", "
            << alpha_index_basic[i][3] << "}";
        if (i < alpha_index_basic_count - 1)
            ofs << ", ";
    }
    ofs << "}\n";
    ofs << "alpha_index_times_count = " << alpha_index_times_count << '\n';
    ofs << "alpha_index_times = {";
    for (int i = 0; i < alpha_index_times_count; i++)
    {
        ofs << '{'
            << alpha_index_times[i][0] << ", "
            << alpha_index_times[i][1] << ", "
            << alpha_index_times[i][2] << ", "
            << alpha_index_times[i][3] << "}";
        if (i < alpha_index_times_count - 1)
            ofs << ", ";
    }
    ofs << "}\n";
    ofs << "alpha_scalar_moments = " << alpha_scalar_moments << '\n';
    ofs << "alpha_moment_mapping = {";
    for (int i = 0; i < alpha_scalar_moments - 1; i++)
        ofs << alpha_moment_mapping[i] << ", ";
    ofs << alpha_moment_mapping[alpha_scalar_moments - 1] << "}\n";

    if (inited)
    {
        ofs << "species_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
            if (i != 0)
                ofs << ", ";
            ofs << linear_coeffs(i);
        }
        ofs << '}' << endl;

        ofs << "moment_coeffs = {";
        for (int i = 0; i < alpha_count - 1; i++)
        {
            if (i != 0)
                ofs << ", ";
            ofs << linear_coeffs(i + species_count);
        }
        ofs << '}';
    }
    ofs.close();
}

void MLMTPR::CalcEFSComponents(Configuration& cfg)
{
    int n = alpha_count + species_count - 1;

    if (cfg.nbh_cutoff != p_RadialBasis->max_val)
        cfg.InitNbhs(p_RadialBasis->max_val);

    forces_cmpnts.resize(cfg.size(), n, 3);

    FillWithZero(energy_cmpnts);
    forces_cmpnts.set(0);
    stress_cmpnts.set(0);

    InitMagMomBasis(cfg);

    for (Neighborhood& nbh : cfg.nbhs) 
    {
        CalcBasisFuncsDers(nbh);

        int type_central = type_from_atomic_number(nbh.my_type); //type_from_atomic_number() function is used here and everywhere in the mtpr.cpp to provide compatibility with absolute atomic numeration

        if (type_central >= species_count)
            throw MlipException("Too few species count in the MTP potential!");

        energy_cmpnts[type_central] += basis_vals[0];

        for (int k = species_count; k < n; k++) {
            int i = k - species_count + 1;

            energy_cmpnts[k] += basis_vals[i];

            for (int j = 0; j < nbh.count; j++)
                for (int a = 0; a < 3; a++) {
                    forces_cmpnts(nbh.my_ind, k, a) += basis_ders(i, j, a);
                    forces_cmpnts(nbh.inds[j], k, a) -= basis_ders(i, j, a);
                }

            for (int j = 0; j < nbh.count; j++)
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        stress_cmpnts(k,a,b) -= basis_ders(i, j, a) * nbh.vecs[j][b];
        }
    }
}

void MLMTPR::CalcEComponents(Configuration& cfg)
{
    int n = alpha_count + species_count - 1;

    if (cfg.nbh_cutoff != p_RadialBasis->max_val)
        cfg.InitNbhs(p_RadialBasis->max_val); 

    FillWithZero(energy_cmpnts);

    InitMagMomBasis(cfg);

    for (Neighborhood& nbh : cfg.nbhs) 
    {
        CalcBasisFuncs(nbh, basis_vals);

        int type_central = type_from_atomic_number(nbh.my_type);

        if (type_central >=species_count)
            throw MlipException("Too few species count in the MTP potential!");

        energy_cmpnts[type_central] += basis_vals[0];

        for (int k = species_count; k < n; k++)
        {
            int i = k - species_count + 1;
            energy_cmpnts[k] += basis_vals[i];
        }
    }
}

void MLMTPR::CalcBasisFuncs(Neighborhood& nbh, std::vector<double>& bf_vals)
{
    bool magmom_treating = (cfg_magnetic_basis_val.size2 > 1); // switches off magnetic staff if false

    int C = species_count;                          //number of different species in current potential
    int K = radial_func_count;                      //number of radial functions in current potential
    int R = p_RadialBasis->size;                    //number of Chebyshev polynomials constituting one radial function
    int M = mbasis_size;                            //size of magnetic basis size

    for (int i = 0; i < alpha_moments_count; i++)
        moment_vals[i] = 0;

    // max_alpha_index_basic calculation
    int max_alpha_index_basic = 0;
    for (int i = 0; i < alpha_index_basic_count; i++)
        max_alpha_index_basic = max(max_alpha_index_basic,
            alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
    max_alpha_index_basic++;
    dist_powers_.resize(max_alpha_index_basic);
    coords_powers_.resize(max_alpha_index_basic);

    int type_central = type_from_atomic_number(nbh.my_type);

    if (type_central>=species_count)
            throw MlipException("Too few species count in the MTP potential!");

    for (int j = 0; j < nbh.count; j++) {
        const Vector3& NeighbVect_j = nbh.vecs[j];

        // calculates vals and ders for j-th atom in the neighborhood
        p_RadialBasis->CalcDers(nbh.dists[j]);
        for (int xi = 0; xi < p_RadialBasis->size; xi++)
            p_RadialBasis->vals[xi] *= scaling;
        for (int xi = 0; xi < p_RadialBasis->size; xi++)
            p_RadialBasis->ders[xi] *= scaling;

        int type_outer = type_from_atomic_number(nbh.types[j]);
        
        if (type_outer>=species_count)
            throw MlipException("Too few species count in the MTP potential!");

        dist_powers_[0] = 1;
        coords_powers_[0] = Vector3(1, 1, 1);
        for (int k = 1; k < max_alpha_index_basic; k++) {
            dist_powers_[k] = dist_powers_[k - 1] * nbh.dists[j];
            for (int a = 0; a < 3; a++)
                coords_powers_[k][a] = coords_powers_[k-1][a] * NeighbVect_j[a];
        }

        for (int i = 0; i < alpha_index_basic_count; i++) {
            double val = 0;
            int mu = alpha_index_basic[i][0];

            for (int ii = 0; ii < M; ii++) 
                for (int jj = 0; jj < M; jj++)
                {
                    double mag_val_mag_val;
                    if (magmom_treating)
                        mag_val_mag_val = cfg_magnetic_basis_val(nbh.my_ind, ii) *
                        cfg_magnetic_basis_val(nbh.inds[j], jj);
                    else
                        mag_val_mag_val = 1.0;

                    for (int xi = 0; xi < p_RadialBasis->size; xi++)
                        val += regression_coeffs[(type_central*C + type_outer)*K*R*M*M + (mu * R + xi)*M*M + ii*M + jj] *
                               p_RadialBasis->vals[xi] * mag_val_mag_val;
                }

            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            double powk = 1.0 / dist_powers_[k];
            val *= powk;        

            double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
            double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
            double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

            double mult0 = pow0*pow1*pow2;
            moment_vals[i] += val * mult0;
        }
    }
    // Next: calculating non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; i++) {
        double val0 = moment_vals[alpha_index_times[i][0]];
        double val1 = moment_vals[alpha_index_times[i][1]];
        int val2 = alpha_index_times[i][2];
        moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
    }
    // Next: copying all b_i corresponding to scalars into separate arrays,
    // basis_vals and basis_ders
    bf_vals[0] = 1.0;  // setting the constant basis function

    for (int i = 0; i < alpha_scalar_moments; i++) 
        bf_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
    
}

void MLMTPR::CalcBasisFuncsDers(const Neighborhood& nbh)
{
    bool magmom_treating = (cfg_magnetic_basis_val.size2 > 1); // switches off magnetic staff if false

    int C = species_count;                   //number of different species in current potential
    int K = radial_func_count;               //number of radial functions in current potential
    int R = p_RadialBasis->size;             //number of Chebyshev polynomials constituting one radial function
    int M = mbasis_size;                     //size of magnetic basis size

    if (nbh.count != moment_ders.size2)
        moment_ders.resize(alpha_moments_count, nbh.count, 3);

    for (int i = 0; i < alpha_moments_count; i++)
        moment_vals[i] = 0;

    moment_ders.set(0);

    // max_alpha_index_basic calculation
    int max_alpha_index_basic = 0;
    for (int i = 0; i < alpha_index_basic_count; i++)
        max_alpha_index_basic = max(max_alpha_index_basic,
            alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
    max_alpha_index_basic++;
    dist_powers_.resize(max_alpha_index_basic);
    coords_powers_.resize(max_alpha_index_basic);

    int type_central = type_from_atomic_number(nbh.my_type);

    if (type_central>=species_count)
            throw MlipException("Too few species count in the MTP potential!");

    for (int j = 0; j < nbh.count; j++) {
        const Vector3& NeighbVect_j = nbh.vecs[j];

        // calculates vals and ders for j-th atom in the neighborhood
        p_RadialBasis->CalcDers(nbh.dists[j]);
        for (int xi = 0; xi < p_RadialBasis->size; xi++)
            p_RadialBasis->vals[xi] *= scaling;
        for (int xi = 0; xi < p_RadialBasis->size; xi++)
            p_RadialBasis->ders[xi] *= scaling;

        int type_outer = type_from_atomic_number(nbh.types[j]);

        if (type_outer>=species_count)
            throw MlipException("Too few species count in the MTP potential!");

        dist_powers_[0] = 1;
        coords_powers_[0] = Vector3(1, 1, 1);
        for (int k = 1; k < max_alpha_index_basic; k++) {
            dist_powers_[k] = dist_powers_[k - 1] * nbh.dists[j];
            for (int a = 0; a < 3; a++)
                coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
        }

        for (int i = 0; i < alpha_index_basic_count; i++) {
            double val = 0;
            double der = 0;
            int mu = alpha_index_basic[i][0];

            // here \phi_xi(r) is RadialBasis::vals[xi]
            for (int ii = 0; ii < M; ii++) {
                for (int jj = 0; jj < M; jj++)
                { 
                    double mag_val_mag_val;
                    if (magmom_treating)
                        mag_val_mag_val = cfg_magnetic_basis_val(nbh.my_ind, ii) *
                        cfg_magnetic_basis_val(nbh.inds[j], jj);
                    else
                        mag_val_mag_val = 1.0;

                    for (int xi = 0; xi < p_RadialBasis->size; xi++)
                    {
                        int idx = (type_central*C + type_outer)*K*R*M*M + (mu * R + xi)*M*M + ii*M + jj;
                        val += regression_coeffs[idx] * p_RadialBasis->vals[xi] * mag_val_mag_val;
                        der += regression_coeffs[idx] * p_RadialBasis->ders[xi] * mag_val_mag_val;
                    }
                }
            }

            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            double powk = 1.0 / dist_powers_[k];
            val *= powk;
            der = der * powk - k * val / nbh.dists[j];

            double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
            double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
            double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

            double mult0 = pow0*pow1*pow2;

            moment_vals[i] += val * mult0;

            mult0 *= der / nbh.dists[j];
            moment_ders(i, j, 0) += mult0 * NeighbVect_j[0];
            moment_ders(i, j, 1) += mult0 * NeighbVect_j[1];
            moment_ders(i, j, 2) += mult0 * NeighbVect_j[2];

            if (alpha_index_basic[i][1] != 0) {
                moment_ders(i, j, 0) += val * alpha_index_basic[i][1]
                    * coords_powers_[alpha_index_basic[i][1] - 1][0]
                    * pow1
                    * pow2;
            }
            if (alpha_index_basic[i][2] != 0) {
                moment_ders(i, j, 1) += val * alpha_index_basic[i][2]
                    * pow0
                    * coords_powers_[alpha_index_basic[i][2] - 1][1]
                    * pow2;
            }
            if (alpha_index_basic[i][3] != 0) {
                moment_ders(i, j, 2) += val * alpha_index_basic[i][3]
                    * pow0
                    * pow1
                    * coords_powers_[alpha_index_basic[i][3] - 1][2];
            }
        }
    }
    // Next: calculating non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; i++) {
        double val0 = moment_vals[alpha_index_times[i][0]];
        double val1 = moment_vals[alpha_index_times[i][1]];
        int val2 = alpha_index_times[i][2];
        moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;

        for (int j = 0; j < nbh.count; j++) {
            for (int a = 0; a < 3; a++) {
                moment_ders(alpha_index_times[i][3], j, a) += val2 * (moment_ders(alpha_index_times[i][0], j, a) * val1 + val0 * moment_ders(alpha_index_times[i][1], j, a));
            }
        }
    }
    // Next: copying all b_i corresponding to scalars into separate arrays,
    // basis_vals and basis_ders
    basis_vals[0] = 1.0;  // setting the constant basis function

    if (basis_ders.size2 != nbh.count) // TODO: remove this check?
        basis_ders.resize(alpha_count, nbh.count, 3);

    if (nbh.count > 0)
        memset(&basis_ders(0, 0, 0), 0, 3 * nbh.count*sizeof(double));

    if (nbh.count > 0)
        for (int i = 0; i < alpha_scalar_moments; i++) {
            basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
            memcpy(&basis_ders(1 + i, 0, 0), &moment_ders(alpha_moment_mapping[i], 0, 0), 3 * nbh.count*sizeof(double));
        }
    else
        for (int i = 0; i < alpha_scalar_moments; i++) 
            basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::MemAlloc()
{
    int n = alpha_count - 1 + species_count;

    energy_cmpnts.resize(n);
    forces_cmpnts.reserve(n * 3);
    stress_cmpnts.resize(n,3,3);

    moment_vals.resize(alpha_moments_count);
    basis_vals.resize(alpha_count);
    site_energy_ders_wrt_moments_.resize(alpha_moments_count);
	reg_vector.resize(alpha_scalar_moments+species_count); 			//resize the regularization vector
}

MLMTPR::~MLMTPR()
{
    if (alpha_moment_mapping != NULL) delete[] alpha_moment_mapping;
    if (alpha_index_times != NULL) delete[] alpha_index_times;
    if (alpha_index_basic != NULL) delete[] alpha_index_basic;

    alpha_moment_mapping = NULL;
    alpha_index_times = NULL;
    alpha_index_basic = NULL;

    if (p_RadialBasis!=NULL)
    delete p_RadialBasis;
}

void MLMTPR::CalcSiteEnergyDers(const Neighborhood& nbh)
{
    bool magmom_treating = (cfg_magnetic_basis_val.size2 > 1); // switches off magnetic staff if false

    buff_site_energy_ = 0.0;
    buff_site_energy_ders_.resize(nbh.count);
    FillWithZero(buff_site_energy_ders_);
    buff_se_ders_wrt_magmom_neigh_.resize(nbh.count);
    FillWithZero(buff_se_ders_wrt_magmom_neigh_);
    buff_se_ders_wrt_magmom_central_ = 0;

    int C = species_count;                      //number of different species in current potential
    int K = radial_func_count;                  //number of radial functions in current potential
    int R = p_RadialBasis->size;                //number of Chebyshev polynomials constituting one radial function
    int M = mbasis_size;                        //size of magnetic basis size

    if (nbh.count != moment_jacobian_.size2)
        moment_jacobian_.resize(alpha_index_basic_count, nbh.count, 3);

    std::fill(moment_vals.begin(),moment_vals.end(),0);
    moment_jacobian_.set(0);

    moment_jacobian_wrt_magmom_neigh.resize(alpha_index_basic_count, nbh.count);
    moment_jacobian_wrt_magmom_neigh.set(0);
    moment_jacobian_wrt_magmom_central.resize(alpha_index_basic_count);
    FillWithZero(moment_jacobian_wrt_magmom_central);

    // max_alpha_index_basic calculation
    int max_alpha_index_basic = 0;
    for (int i = 0; i < alpha_index_basic_count; i++)
        max_alpha_index_basic = max(max_alpha_index_basic,
            alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
    max_alpha_index_basic++;
    dist_powers_.resize(max_alpha_index_basic);
    coords_powers_.resize(max_alpha_index_basic);

    int type_central = type_from_atomic_number(nbh.my_type);   //type_from_atomic_number() function is used here and everywhere in the mtpr.cpp to provide compatibility with absolute atomic numeration

    if (type_central>=species_count)
            throw MlipException("Too few species count in the MTP potential!");

    for (int j = 0; j < nbh.count; j++) {
        const Vector3& NeighbVect_j = nbh.vecs[j];

        // calculates vals and ders for j-th atom in the neighborhood
        p_RadialBasis->CalcDers(nbh.dists[j]);
        for (int xi = 0; xi < p_RadialBasis->size; xi++)
            p_RadialBasis->vals[xi] *= scaling;
        for (int xi = 0; xi < p_RadialBasis->size; xi++)
            p_RadialBasis->ders[xi] *= scaling;

        dist_powers_[0] = 1;
        coords_powers_[0] = Vector3(1, 1, 1);
        for (int k = 1; k < max_alpha_index_basic; k++) {
            dist_powers_[k] = dist_powers_[k - 1] * nbh.dists[j];
            for (int a = 0; a < 3; a++)
                coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
        }

        int type_outer = type_from_atomic_number(nbh.types[j]);

        if (type_outer>=species_count)
            throw MlipException("Too few species count in the MTP potential!");

        for (int i = 0; i < alpha_index_basic_count; i++) {

            double val = 0, der = 0;
            double der_magmom_neigh = 0;
            double der_magmom_central = 0;
            
            int mu = alpha_index_basic[i][0];

            for (int ii = 0; ii < M; ii++) {
                for (int jj = 0; jj < M; jj++) 
                { 
                    double mag_val_mag_val, mag_val_mag_der_neigh, mag_val_mag_der_centr;
                    if (magmom_treating)
                    {
                        mag_val_mag_val = cfg_magnetic_basis_val(nbh.my_ind, ii) *
                                          cfg_magnetic_basis_val(nbh.inds[j], jj);
                        mag_val_mag_der_neigh = cfg_magnetic_basis_val(nbh.my_ind, ii) * 
                                                cfg_magnetic_basis_der(nbh.inds[j], jj);
                        mag_val_mag_der_centr = cfg_magnetic_basis_der(nbh.my_ind, ii) * 
                                                cfg_magnetic_basis_val(nbh.inds[j], jj);
                    }
                    else
                    {
                        mag_val_mag_val = 1.0;
                        mag_val_mag_der_neigh = 0.0;
                        mag_val_mag_der_centr = 0.0;
                    }

                    for (int xi = 0; xi < R; xi++)
                    {
                        int idx = (type_central*C + type_outer)*K*R*M*M + (mu * R + xi)*M*M + ii*M + jj;
                        val += regression_coeffs[idx] * p_RadialBasis->vals[xi] * mag_val_mag_val;
                        der += regression_coeffs[idx] * p_RadialBasis->ders[xi] * mag_val_mag_val;
                        der_magmom_neigh += regression_coeffs[idx] * p_RadialBasis->vals[xi] * mag_val_mag_der_neigh;
                        der_magmom_central += regression_coeffs[idx] * p_RadialBasis->vals[xi] * mag_val_mag_der_centr;
                    }
                }
            }

            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            double powk = 1.0 / dist_powers_[k];
            val *= powk;
            der = der * powk - k * val / nbh.dists[j];

            double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
            double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
            double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

            double mult0 = pow0*pow1*pow2;

            moment_vals[i] += val * mult0;
            mult0 *= der / nbh.dists[j];
            moment_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
            moment_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
            moment_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];

            moment_jacobian_wrt_magmom_neigh(i, j) += pow0 * pow1 * pow2 * powk * der_magmom_neigh;
            moment_jacobian_wrt_magmom_central[i] += pow0 * pow1 * pow2 * powk * der_magmom_central;

            if (alpha_index_basic[i][1] != 0) {
                moment_jacobian_(i, j, 0) += val * alpha_index_basic[i][1]
                    * coords_powers_[alpha_index_basic[i][1] - 1][0]
                    * pow1
                    * pow2;
            }
            if (alpha_index_basic[i][2] != 0) {
                moment_jacobian_(i, j, 1) += val * alpha_index_basic[i][2]
                    * pow0
                    * coords_powers_[alpha_index_basic[i][2] - 1][1]
                    * pow2;
            }
            if (alpha_index_basic[i][3] != 0) {
                moment_jacobian_(i, j, 2) += val * alpha_index_basic[i][3]
                    * pow0
                    * pow1
                    * coords_powers_[alpha_index_basic[i][3] - 1][2];
            }
        }

        //Repulsive term. Creates an artificial force (and energy) contribution moving the atoms towards mindist, if they stay closer. 
        if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls")
        if (nbh.dists[j] < p_RadialBasis->min_val)
        {
            double multiplier = 10000;
            buff_site_energy_ += multiplier*(exp(-10*(nbh.dists[j]-1)) - exp(-10*(p_RadialBasis->min_val-1)));
            for (int a = 0; a < 3; a++)
                buff_site_energy_ders_[j][a] += -10 * multiplier*(exp(-10 * (nbh.dists[j] - 1))/ nbh.dists[j])*nbh.vecs[j][a];
        }
    }

    // Next: calculating non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; i++) {
        double val0 = moment_vals[alpha_index_times[i][0]];
        double val1 = moment_vals[alpha_index_times[i][1]];
        int val2 = alpha_index_times[i][2];
        moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
    }

    // convolving with coefficients
    buff_site_energy_ += linear_coeffs(type_central);

    for (int i = 0; i < alpha_scalar_moments; i++)
        buff_site_energy_ += linear_coeffs(species_count + i) * moment_vals[alpha_moment_mapping[i]];

    //if (wgt_eqtn_forces != 0) //!!! CHECK IT!!!
    {
        // Backpropagation starts

        // Backpropagation step 1: site energy derivative is the corresponding linear combination
        memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(site_energy_ders_wrt_moments_[0]));

        for (int i = 0; i < alpha_scalar_moments; i++)
            site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_coeffs(species_count + i);

        // Backpropagation step 2: expressing through basic moments:
        for (int i = alpha_index_times_count - 1; i >= 0; i--) {
            double val0 = moment_vals[alpha_index_times[i][0]];
            double val1 = moment_vals[alpha_index_times[i][1]];
            int val2 = alpha_index_times[i][2];

            site_energy_ders_wrt_moments_[alpha_index_times[i][1]] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
                * val2 * val0;
            site_energy_ders_wrt_moments_[alpha_index_times[i][0]] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
                * val2 * val1;
        }

        // Backpropagation step 3: multiply by the Jacobian:
        for (int i = 0; i < alpha_index_basic_count; i++) {
            for (int j = 0; j < nbh.count; j++) {
                for (int a = 0; a < 3; a++)
                    buff_site_energy_ders_[j][a] += site_energy_ders_wrt_moments_[i] * moment_jacobian_(i, j, a);
                buff_se_ders_wrt_magmom_neigh_[j] += site_energy_ders_wrt_moments_[i] * moment_jacobian_wrt_magmom_neigh(i, j);
           }
           buff_se_ders_wrt_magmom_central_ += site_energy_ders_wrt_moments_[i] * moment_jacobian_wrt_magmom_central[i];
        }
    }
}

void MLMTPR::AccumulateCombinationGrad( const Neighborhood& nbh,
                                        std::vector<double>& out_grad_accumulator,
                                        const double se_weight,
                                        const Vector3* se_ders_weights) 
{
    int C = species_count;                      //number of different species in current potential
    int K = radial_func_count;                      //number of radial functions in current potential
    int R = p_RadialBasis->size;  //number of Chebyshev polynomials constituting one radial function
    int M = mbasis_size;                     //size of magnetic basis size

    int coeff_count = C*C*R*K*M*M;

    buff_site_energy_ders_.resize(nbh.count);
    out_grad_accumulator.resize(CoeffCount());

    buff_site_energy_ = 0.0;
    FillWithZero(buff_site_energy_ders_);

    bool magmom_treating = (cfg_magnetic_basis_val.size2 > 1); // switches off magnetic staff if false

    {
        site_energy_ders_wrt_moments_.resize(alpha_moments_count);
        std::vector<double> mom_val(alpha_moments_count);

        for (int i = 0; i < alpha_moments_count; i++)
        {
            mom_val[i] = 0;
            site_energy_ders_wrt_moments_[i] = 0;
        }

        moment_ders.set(0);

        //derivatives of the basic moments (mom_val) wrt interatomic distances (NeighbVect_j[a])
        Array3D mom_jacobian_(alpha_index_basic_count, nbh.count, 3);
        mom_jacobian_.set(0);

        //derivatives of the basic moments (mom_val) wrt radial coefficients (regression_coeffs[....])
        Array3D mom_rad_jacobian_(alpha_index_basic_count, C, R*K*M*M);
        mom_rad_jacobian_.set(0);

        //derivatives of the loss function wrt (mixed derivatives of basic moments wrt radial coefficients wrt coordinates)
        Array3D dloss_dmom_rad_coord_jacobian_(alpha_index_basic_count, C, R*K*M*M);
        dloss_dmom_rad_coord_jacobian_.set(0);

        //derivatives of the loss functions wrt site energies
        vector<double> dloss_dsenders(alpha_moments_count);
        memset(&dloss_dsenders[0], 0, alpha_moments_count*sizeof(double));

        //derivatives of the loss functions wrt the basic moments (mom_val)
        vector<double>dloss_dmom(alpha_moments_count);
        memset(&dloss_dmom[0], 0, alpha_moments_count*sizeof(double));

        // max_alpha_index_basic calculation
        int max_alpha_index_basic = 0;
        for (int i = 0; i < alpha_index_basic_count; i++)
            max_alpha_index_basic = max(max_alpha_index_basic,
                alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);

        max_alpha_index_basic++;
        std::vector<double> auto_dist_powers_(max_alpha_index_basic);
        std::vector<Vector3> auto_coords_powers_(max_alpha_index_basic);

        int type_central = type_from_atomic_number(nbh.my_type);

        if (type_central>=species_count)
            throw MlipException("Too few species count in the MTP potential!");

        for (int j = 0; j < nbh.count; j++) {
            const Vector3& NeighbVect_j = nbh.vecs[j];

            // calculates vals and ders for j-th atom in the neighborhood
            p_RadialBasis->CalcDers(nbh.dists[j]);
            for (int xi = 0; xi < p_RadialBasis->size; xi++)
                p_RadialBasis->vals[xi] *= scaling;
            for (int xi = 0; xi < p_RadialBasis->size; xi++)
                p_RadialBasis->ders[xi] *= scaling;

            auto_dist_powers_[0] = 1;
            auto_coords_powers_[0] = Vector3(1, 1, 1);
            for (int k = 1; k < max_alpha_index_basic; k++) {
                auto_dist_powers_[k] = auto_dist_powers_[k - 1] * nbh.dists[j];
                for (int a = 0; a < 3; a++)
                    auto_coords_powers_[k][a] = auto_coords_powers_[k - 1][a] * NeighbVect_j[a];
            }

            int type_outer = type_from_atomic_number(nbh.types[j]); 

            for (int i = 0; i < alpha_index_basic_count; i++) {

                double val = 0;
                double der = 0;

                int mu = alpha_index_basic[i][0];
                int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];

                double powk = 1.0 / auto_dist_powers_[k];

                double pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
                double pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
                double pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];

                double mult0 = pow0*pow1*pow2;

                for (int beta = 0; beta < M; beta++) { 
                    for (int gamma = 0; gamma < M; gamma++) 
                    {
                        double mag_val_mag_val;
                        if (magmom_treating)
                            mag_val_mag_val = cfg_magnetic_basis_val(nbh.my_ind, beta) *
                                              cfg_magnetic_basis_val(nbh.inds[j], gamma);
                        else 
                            mag_val_mag_val = 1.0;

                        for (int xi = 0; xi < R; xi++) {
                            int idx = (type_central*C + type_outer)*K*R*M*M + M*M*(mu * R + xi) + beta*M + gamma;
                            double rad_val_mag_val_mag_val = p_RadialBasis->vals[xi] * mag_val_mag_val;
                            double rad_der_mag_val_mag_val = p_RadialBasis->ders[xi] * mag_val_mag_val;
                            val += regression_coeffs[idx] * rad_val_mag_val_mag_val * powk;
                            der += regression_coeffs[idx] * rad_der_mag_val_mag_val;

                            mom_val[i] += regression_coeffs[idx] * rad_val_mag_val_mag_val * powk*mult0;

                            mom_rad_jacobian_(i, type_outer, M*M*(mu * R + xi) + beta*M + gamma) += rad_val_mag_val_mag_val * 
                                                                                            powk * mult0;

                            if (se_ders_weights != nullptr)
                            {
                                double derx = (NeighbVect_j[0] / nbh.dists[j])*(rad_der_mag_val_mag_val * powk*mult0 - rad_val_mag_val_mag_val * k*powk*mult0 / nbh.dists[j]);
                                double dery = (NeighbVect_j[1] / nbh.dists[j])*(rad_der_mag_val_mag_val * powk*mult0 - rad_val_mag_val_mag_val * k*powk*mult0 / nbh.dists[j]);
                                double derz = (NeighbVect_j[2] / nbh.dists[j])*(rad_der_mag_val_mag_val * powk*mult0 - rad_val_mag_val_mag_val * k*powk*mult0 / nbh.dists[j]);

                                if (alpha_index_basic[i][1] != 0) {
                                    derx += rad_val_mag_val_mag_val * powk * alpha_index_basic[i][1]
                                        * auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
                                        * pow1
                                        * pow2;
                                }

                                if (alpha_index_basic[i][2] != 0) {
                                    dery += rad_val_mag_val_mag_val * powk* alpha_index_basic[i][2]
                                        * pow0
                                        * auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
                                        * pow2;
                                }

                                if (alpha_index_basic[i][3] != 0) {
                                    derz += rad_val_mag_val_mag_val * powk* alpha_index_basic[i][3]
                                        * pow0
                                        * pow1
                                        * auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
                                }

                                dloss_dmom_rad_coord_jacobian_(i, type_outer, M*M*(mu * R + xi) + beta*M + gamma) += se_ders_weights[j][0] * derx;
                                dloss_dmom_rad_coord_jacobian_(i, type_outer, M*M*(mu * R + xi) + beta*M + gamma) += se_ders_weights[j][1] * dery;
                                dloss_dmom_rad_coord_jacobian_(i, type_outer, M*M*(mu * R + xi) + beta*M + gamma) += se_ders_weights[j][2] * derz;
                            }
                        }
                    }
                }

                der = der * powk - k * val / nbh.dists[j];
                mult0 *= der / nbh.dists[j];

                mom_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
                mom_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
                mom_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];
                
                if (alpha_index_basic[i][1] != 0) {
                    mom_jacobian_(i, j, 0) += val * alpha_index_basic[i][1]
                        * auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
                        * pow1
                        * pow2;
                }
                if (alpha_index_basic[i][2] != 0) {
                    mom_jacobian_(i, j, 1) += val * alpha_index_basic[i][2]
                        * pow0
                        * auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
                        * pow2;
                }
                if (alpha_index_basic[i][3] != 0) {
                    mom_jacobian_(i, j, 2) += val * alpha_index_basic[i][3]
                        * pow0
                        * pow1
                        * auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
                }
            }
            //Repulsive term 
            if (p_RadialBasis->GetRBTypeString()=="RBChebyshev_repuls")
            if (nbh.dists[j] < p_RadialBasis->min_val)
            {
                double multiplier = 10000;
                buff_site_energy_ += multiplier*(exp(-10 * (nbh.dists[j] - 1)) - exp(-10 * (p_RadialBasis->min_val - 1)));
                for (int a = 0; a < 3; a++)
                    buff_site_energy_ders_[j][a] += -10 * multiplier*(exp(-10 * (nbh.dists[j] - 1)) / nbh.dists[j])*nbh.vecs[j][a];
            }
        }
        // Next: calculating non-elementary b_i
        for (int i = 0; i < alpha_index_times_count; i++) {
            double val0 = mom_val[alpha_index_times[i][0]];
            double val1 = mom_val[alpha_index_times[i][1]];
            double val2 = alpha_index_times[i][2];
            mom_val[alpha_index_times[i][3]] += val2 * val0 * val1;
        }
        // convolving with coefficients
        buff_site_energy_ += linear_coeffs(type_central);
        for (int i = 0; i < alpha_scalar_moments; i++)
            buff_site_energy_ += linear_coeffs(species_count + i) * mom_val[alpha_moment_mapping[i]];

        // Backpropagation starts

        // Backpropagation step 1: site energy derivative is the corresponding linear combination
        for (int i = 0; i < alpha_scalar_moments; i++)
            site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_coeffs(species_count + i);

        // Backpropagation step 2: expressing through basic moments:
        for (int i = alpha_index_times_count - 1; i >= 0; i--) {
            double val0 = mom_val[alpha_index_times[i][0]];
            double val1 = mom_val[alpha_index_times[i][1]];
            double val2 = alpha_index_times[i][2];

            site_energy_ders_wrt_moments_[alpha_index_times[i][1]] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
                * val2 * val0;
            site_energy_ders_wrt_moments_[alpha_index_times[i][0]] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
                * val2 * val1;
        }
        // Backpropagation step 3: multiply by the Jacobian:
        //if (se_ders_weights != nullptr)
            for (int i = 0; i < alpha_index_basic_count; i++)
                for (int j = 0; j < nbh.count; j++)
                    for (int a = 0; a < 3; a++) {
                        buff_site_energy_ders_[j][a] += site_energy_ders_wrt_moments_[i] * mom_jacobian_(i, j, a);

                        if (se_ders_weights != nullptr)
                        dloss_dsenders[i] += se_ders_weights[j][a] * mom_jacobian_(i, j, a);
                    }

        for (int i = 0; i < alpha_index_basic_count; i++)
            dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i];

        if (se_ders_weights)
        {
            for (int i = 0; i < alpha_index_times_count; i++) {
                double val0 = mom_val[alpha_index_times[i][0]];
                double val1 = mom_val[alpha_index_times[i][1]];
                double val2 = alpha_index_times[i][2];

                dloss_dsenders[alpha_index_times[i][3]] += dloss_dsenders[alpha_index_times[i][1]] * val2*val0;
                dloss_dsenders[alpha_index_times[i][3]] += dloss_dsenders[alpha_index_times[i][0]] * val2*val1;
            }

            for (int i = 0; i < alpha_index_times_count; i++) {
                //              double val0 = mom_val[alpha_index_times[i][0]];
                //              double val1 = mom_val[alpha_index_times[i][1]];
                double val2 = alpha_index_times[i][2];

                dloss_dmom[alpha_index_times[i][1]] += dloss_dsenders[alpha_index_times[i][0]] *
                    site_energy_ders_wrt_moments_[alpha_index_times[i][3]] * val2;

                dloss_dmom[alpha_index_times[i][0]] += dloss_dsenders[alpha_index_times[i][1]] *
                    site_energy_ders_wrt_moments_[alpha_index_times[i][3]] * val2;
            }

            for (int i = alpha_index_times_count - 1; i >= 0; i--) {
                double val0 = mom_val[alpha_index_times[i][0]];
                double val1 = mom_val[alpha_index_times[i][1]];
                double val2 = alpha_index_times[i][2];

                dloss_dmom[alpha_index_times[i][0]] += dloss_dmom[alpha_index_times[i][3]] * val2*val1;
                dloss_dmom[alpha_index_times[i][1]] += dloss_dmom[alpha_index_times[i][3]] * val2*val0;
            }
        }
        for (int i = 0; i < alpha_index_basic_count; i++)
            for (int j = 0; j < C; j++)
                for (int k = 0; k < R*K*M*M; k++)
                {
                    out_grad_accumulator[(C*type_central + j)*R*K*M*M + k] += dloss_dmom[i] * mom_rad_jacobian_(i, j, k); //from basic moments

                    if (se_ders_weights != nullptr)  //if (wgt_eqtn_forces != 0)       /// CHECK IT CAREFULLY!!!!
                    out_grad_accumulator[(C*type_central + j)*R*K*M*M + k] += site_energy_ders_wrt_moments_[i] * dloss_dmom_rad_coord_jacobian_(i, j, k); //from basic moment ders 
                }

        out_grad_accumulator[coeff_count + type_central] += se_weight;
        for (int i = 0; i < alpha_scalar_moments; i++)
        {
            out_grad_accumulator[i + coeff_count + species_count] += se_weight * mom_val[alpha_moment_mapping[i]];      //from site energy

            if (se_ders_weights != nullptr)
                out_grad_accumulator[i + coeff_count + species_count] += dloss_dsenders[alpha_moment_mapping[i]];           //from site forces
        }
    }
}

void MLMTPR::AddPenaltyGrad(const double coeff,                             // Must calculate add penalty and optionally (if out_penalty_grad_accumulator != nullptr) its gradient w.r.t. coefficients multiplied by coeff to out_penalty
                            double& out_penalty_accumulator,
                            Array1D* out_penalty_grad_accumulator)
{
    const int C = species_count;
    const int K = radial_func_count;
    const int R = p_RadialBasis->size;
    const int M = mbasis_size;

    if (out_penalty_grad_accumulator != nullptr)
        out_penalty_grad_accumulator->resize(CoeffCount());

	//this normalizes the sum of squares of coefficients in C*C*R dimensions to 1 for each K
    for (int k = 0; k < K; k++)
    {
        double norm = 0;

        for (int i = 0; i < C; i++)
            for (int j = 0; j < C; j++)
                for (int l = 0; l < R; l++)
                    for (int beta = 0; beta < M; beta++)
                        for (int gamma = 0; gamma < M; gamma++) {
                            int idx = (i*C + j)*K*R*M*M + M*M*(k*R + l) + beta*M + gamma;
                            norm += regression_coeffs[idx] * regression_coeffs[idx];
                        }

        out_penalty_accumulator += coeff * (norm - 1) * (norm - 1);

        if (out_penalty_grad_accumulator != nullptr)
            for (int i = 0; i < C; i++)
                for (int j = 0; j < C; j++)
                    for (int l = 0; l < R; l++)
                        for (int beta = 0; beta < M; beta++)
                            for (int gamma = 0; gamma < M; gamma++) {
                                int idx = (i*C + j)*K*R*M*M + M*M*(k*R + l) + beta*M + gamma;
                                (*out_penalty_grad_accumulator)[idx] += coeff * 4 * (norm - 1) * regression_coeffs[idx];
                            }
    }

	//this is targeted on making the radial functions for different K orthogonal
    for (int k1 = 0; k1 < K; k1++)
        for (int k2 = k1 + 1; k2 < K; k2++)
        {
            double scal = 0;

            for (int i = 0; i < C; i++)
                for (int j = 0; j < C; j++)
                    for (int l = 0; l < R; l++)
                        for (int beta = 0; beta < M; beta++)
                            for (int gamma = 0; gamma < M; gamma++) {
                                int idx1 = (i*C + j)*K*R*M*M + M*M*(k1*R + l) + beta*M + gamma;
                                int idx2 = (i*C + j)*K*R*M*M + M*M*(k2*R + l) + beta*M + gamma;
                                scal += regression_coeffs[idx1] * regression_coeffs[idx2];
                            }

            out_penalty_accumulator += coeff * scal*scal;

            if (out_penalty_grad_accumulator != nullptr)
                for (int i = 0; i < C; i++)
                    for (int j = 0; j < C; j++)
                        for (int l = 0; l < R; l++)
                            for (int beta = 0; beta < M; beta++)
                                for (int gamma = 0; gamma < M; gamma++) { 
                                    int idx1 = (i*C + j)*K*R*M*M + M*M*(k1*R + l) + beta*M + gamma;
                                    int idx2 = (i*C + j)*K*R*M*M + M*M*(k2*R + l) + beta*M + gamma;
                                    (*out_penalty_grad_accumulator)[idx1] += coeff * 2 * scal*regression_coeffs[idx2];
                                    (*out_penalty_grad_accumulator)[idx2] += coeff * 2 * scal*regression_coeffs[idx1];
                                }
        }

	//contribution from regularization of SLAE matrix
	for (int i=0; i<LinSize();i++)
		out_penalty_accumulator+=linear_coeffs(i)*linear_coeffs(i)*reg_vector[i];

	//contribution from regularization of SLAE matrix
	if (out_penalty_grad_accumulator != nullptr)
    	for (int i=0; i<LinSize();i++)
    		(*out_penalty_grad_accumulator)[Rsize()+i]+=2*linear_coeffs(i)*reg_vector[i];
		
}

void MLMTPR::Orthogonalize()
{
    const int C = species_count;
    const int K = radial_func_count;
    const int R = p_RadialBasis->size;
    const int M = mbasis_size;

	//making the radial functions for different K orthogonal
    for (int k = 0; k < K; k++)
    {
        for (int k2 = 0; k2 < k; k2++) {
            double norm = 0;

            for (int i = 0; i < C; i++)
                for (int j = 0; j < C; j++)
                    for (int l = 0; l < R; l++)
                        for (int beta = 0; beta < M; beta++)
                            for (int gamma = 0; gamma < M; gamma++) {
                                int idx = (i*C + j)*K*R*M*M + M*M*(k*R + l) + beta*M + gamma;
                                norm += regression_coeffs[idx] * regression_coeffs[idx];
                            }

            double scal = 0;
            for (int i = 0; i < C; i++)
                for (int j = 0; j < C; j++)
                    for (int l = 0; l < R; l++)
                        for (int beta = 0; beta < M; beta++)
                            for (int gamma = 0; gamma < M; gamma++) {
                                int idx = (i*C + j)*K*R*M*M + M*M*(k*R + l) + beta*M + gamma;
                                int idx2 = (i*C + j)*K*R*M*M + M*M*(k2*R + l) + beta*M + gamma;
                                scal += regression_coeffs[idx] * regression_coeffs[idx2];
                            }

            for (int i = 0; i < C; i++)
                for (int j = 0; j < C; j++)
                    for (int l = 0; l < R; l++)
                        for (int beta = 0; beta < M; beta++)
                            for (int gamma = 0; gamma < M; gamma++) {
                                int idx = (i*C + j)*K*R*M*M + M*M*(k*R + l) + beta*M + gamma;
                                int idx2 = (i*C + j)*K*R*M*M + M*M*(k2*R + l) + beta*M + gamma;
                                regression_coeffs[idx] -= regression_coeffs[idx2] * scal / norm;
                            }
        }

        {
            double norm = 0;

			//this normalizes the sum of squares of coefficients in C*C*R dimensions to 1 for each K
            for (int i = 0; i < C; i++)
                for (int j = 0; j < C; j++)
                    for (int l = 0; l < R; l++)
                        for (int beta = 0; beta < M; beta++)
                            for (int gamma = 0; gamma < M; gamma++) {
                                int idx = (i*C + j)*K*R*M*M + M*M*(k*R + l) + beta*M + gamma;
                                norm += regression_coeffs[idx] * regression_coeffs[idx]+1e-10;
                            }

            norm = sqrt(norm);
            for (int i = 0; i < C; i++)
                for (int j = 0; j < C; j++)
                    for (int l = 0; l < R; l++)
                        for (int beta = 0; beta < M; beta++)
                            for (int gamma = 0; gamma < M; gamma++) {
                                int idx = (i*C + j)*K*R*M*M + M*M*(k*R + l) + beta*M + gamma;
                                regression_coeffs[idx] /= norm;
                            }
        }
    }
}

void MLMTPR::AddSpecies(std::set<int> species, bool init_random)
{
    MPI_Barrier(mpi.comm);

    //This means no coefficients were present in MTPR potential, and the atomic_numbers should be cleared
    if (!inited)
    {
        species_count=0;
        atomic_numbers.clear();
        regression_coeffs.resize(alpha_count - 1);
    }

    //species detected in the training set
    vector<int> db_species(atomic_numbers);

    for (int sp : species)
    {
        if (std::find(db_species.begin(), db_species.end(), sp) == db_species.end())
            db_species.push_back(sp);
    }

    MPI_Barrier(mpi.comm);
    int maxlen = 0;
    int this_size = (int)db_species.size();
    MPI_Allreduce(&this_size, &maxlen, 1, MPI_INT, MPI_MAX, mpi.comm);                     //finding maximum lengths of db_species among the processes

    vector<int> sendbuf(maxlen);		//constructing a sending buffers of equal sizes for all processes
    std::fill(sendbuf.begin(), sendbuf.end(), -1);   //fill sendbuf with -1
    memcpy(&sendbuf[0], &db_species[0], db_species.size() * sizeof(int));  //fill some parts of the sendbuf with actual species detected for each process (for a part of the training set)  

    vector<int> total_species(mpi.size*maxlen);   //constructing the receiving buffer
    std::fill(total_species.begin(), total_species.end(), -1);
    MPI_Barrier(mpi.comm);

    MPI_Allgather(&sendbuf[0], maxlen, MPI_INT, &total_species[0], maxlen, MPI_INT, mpi.comm);    //fill the array with species from all processes

    sort(total_species.begin(), total_species.end());

    for (int i = 0; i < mpi.size*maxlen; i++)                                        //adding new species to the MTPR potential
        if (total_species[i] >= 0)
            if (std::find(atomic_numbers.begin(), atomic_numbers.end(), total_species[i]) == atomic_numbers.end())
                atomic_numbers.push_back(total_species[i]);

    int new_spec = (int)atomic_numbers.size();  //new number of species (after adding new atomic types)
    int old_spec = species_count;               //old number of species

    for (int i=0; i< new_spec; i++)
        if (std::find(total_species.begin(), total_species.end(), atomic_numbers[i]) == total_species.end())
        {
            string errmsg = "Atomic number " + std::to_string(atomic_numbers[i]) + " from the MTP is not present in the training set. You are doing something wrong!";
            ERROR(errmsg);
        }

    if (old_spec < new_spec) {
        int K = radial_func_count;                        //number of radial functions in current potential
        int R = p_RadialBasis->size;               //number of Chebyshev polynomials constituting one radial function  
        int M = mbasis_size;

        //the code below does the rearrangement of the present coefficients in the memory and adding of the new ones
        vector<double> old_radial(old_spec*old_spec*K*R);
        if (old_radial.size() > 0)
            memcpy(&old_radial[0], &regression_coeffs[0], old_spec*old_spec*K*R * sizeof(double));

        // rearrange and extend the linear coefficients
        int nlin_old = LinSize();
        regression_coeffs.resize(Rsize() + nlin_old + new_spec - old_spec);

        for (int i = nlin_old + new_spec - old_spec - 1; i >= new_spec; i--)
            linear_coeffs(i) = linear_coeffs(i - new_spec + old_spec);

        for (int i = old_spec - 1; i >= 0; i--)
            linear_coeffs(i) = linear_coeffs(i);

        for (int i = 0; i < new_spec - old_spec; i++)
            linear_coeffs(i + old_spec) = 0;

        // rearrange and extend the radial coefficients
        species_count = new_spec;
        regression_coeffs.resize(new_spec*new_spec*K*R*M*M + nlin_old + new_spec - old_spec);

        std::random_device rand_device;
        std::default_random_engine generator(rand_device());
        std::uniform_real_distribution<> uniform(-1.0, 1.0);

        int pair = 0;
        int old_pair = 0;
        for (int p1 = 0; p1 < new_spec; p1++) {
            for (int p2 = 0; p2 < new_spec; p2++) {
                if ((p1 < old_spec) && (p2 < old_spec)) {
                    for (int i = 0; i < K*R*M*M; i++)
                        regression_coeffs[pair*K*R*M*M + i] = old_radial[old_pair*K*R*M*M + i];
                    old_pair++;
                }
                else 
                {
                    if (init_random)
                        for (int i = 0; i < K; i++)
                        {
                            for (int j = 0; j < R; j++)
                                for (int alpha = 0; alpha < M; alpha++)
                                    for (int beta = 0; beta < M; beta++)
                                        regression_coeffs[pair*K*R*M*M + (i * R + j)*M*M + alpha * M + beta] = 5e-7*uniform(generator);

                            for (int alpha = 0; alpha < M; alpha++)
                                for (int beta = 0; beta < M; beta++)
                                    regression_coeffs[pair*K*R*M*M + (i * R + min(i, R))*M*M + alpha * M + beta] = 1e-7*(1 + uniform(generator));
                        }
                    else
                    //setting the new coefficients to default values (in case of random initial guess they will be overwritten)
                        for (int i = 0; i < K; i++) 
                        {
                            for (int j = 0; j < R; j++)
                                for (int alpha = 0; alpha < M; alpha++)
                                    for (int beta = 0; beta < M; beta++)
                                        regression_coeffs[pair*K*R*M*M + (i * R + j)*M*M + alpha * M + beta] = 1e-6;

                            for (int alpha = 0; alpha < M; alpha++)
                                for (int beta = 0; beta < M; beta++)
                                    regression_coeffs[pair*K*R*M*M + (i * R + min(i, R))*M*M + alpha * M + beta] = 1e-3*(pair + 1);
                        }
                }
                pair++;
            }
        }
        MemAlloc();     //resize EFS components and basis functions
    }

    reg_vector.resize(alpha_scalar_moments + species_count); //resize the regularization vector

    MPI_Bcast(&Coeff()[0], CoeffCount(), MPI_DOUBLE, 0, mpi.comm);

    inited = true;
}
