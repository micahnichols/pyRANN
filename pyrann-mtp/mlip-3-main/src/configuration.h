/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifndef MLIP_CONFIGURATION_H
#define MLIP_CONFIGURATION_H


#include "common/matrix3.h"
#include "common/utils.h"
#include "common/comm.h"
#include "common/multidimensional_arrays.h"
#include <set>


enum class CrystalSystemType{SC, BCC, FCC, HCP};


//! Structure describing a neighborhood of an atom
struct Neighborhood
{
    int count;                      //  number of neighbors
    std::vector<int> inds;          //  array of indices of neighboring atoms
    std::vector<Vector3> vecs;      //  array of relative positions of the neighboring atoms
    std::vector<double> dists;      //  array of distances to the neighbor atoms
    std::vector<int> types;
    int my_type;
    int my_ind;
};


//! Atomic periodic configuration.
//! Configuration can be in either of the two states: uninitialized, when its size=0,
//! and initialized when its size()>0. The size()==pos_.size()==types_.size() always.
//! If the configuration is initialized, it may or may not have energy, forces, site_energies, etc.
//! To check whether it has energy, do `if(has_energy()) ...` .
//! To declare it does have energy, do `has_energy(true)`.
//! `has_energy(false)` does the opposite.
//! If forces (same as site_energies) are initialized, then forces_.size() == size().
//! Otherwise, forces_.size() == 0.
class Configuration
{
private:
    std::vector<int> orig_atom_inds;    //!< Auxiliary variable for constructing neighborhoods

    void InitNbhs_AddGhostAtoms(int cfg_size, const double cutoff); //!< Add new atoms to the configuration corresponding to periodical extention of the configuration. It doesn't change count-property. The number of atoms in extended configuration is available via pos.size().
    void InitNbhs_ConstructNbhs(int cfg_size, double cutoff);       //!< Calculate indicies of neighboring atoms. For atoms from periodical extensions indices of their atom-origins are used
    void InitNbhs_RemoveGhostAtoms(int cfg_size);
    
protected:
    std::vector<int> types_;            //!< Array of atom types 
    std::vector<Vector3> pos_;          //!< Array of atom coordinates
    std::vector<Vector3> forces_;       //!< Array of forces on atoms
    std::vector<double> site_energies_; //!< Array of site energies
    std::vector<double> charges_;
    std::vector<Vector3> magmom_;
    std::vector<Vector3> en_der_wrt_magmom_;
    std::vector<double> nbh_grades_;

    bool has_energy_;
    bool has_stresses_;

    bool LoadOld(std::ifstream& ifs);   //!< Loading old-format configuration from the filestream. Returns true if success, false if end-of-file.
    bool LoadBin(std::ifstream& ifs);   //!< Loading one configuration from binary filestream. Returns true if read, false if end-of-file.          

    //! reading one ionic iteration from VASP OUTCAR. Ions_per_type should be read before calling LoadNext.
    //! IMPORTANT: if there is no next configuration in the file, it should not overwrite the existing configuration!
    //! Returns true if success, false if end-of-file.
    bool LoadNextFromOUTCAR(std::ifstream&,int maxiter=0);
    int friend LoadPreambleFromOUTCAR(std::ifstream&, std::vector<int>& ions_per_type); //!< Fills ions_per_type & max. number of iterations

public:
    Matrix3 lattice;    //!< Latice vectors in rows (i.e., lattice[0], lattice[1], lattice[2] are the 3 vectors). Required for periodical extension of the configuration. If lattice[3]==(0,0,0) then it is not periodic w.r.t lattice[3]. If all lattice==0 then it is an open-shell molecule, no periodicity whatsoever

    bool is_mpi_splited = false;                            // indicates if configuration on the current processor is a part of a large configuration which atoms are distributed among processors

    double energy;      //!< Energy of the configuration
    Matrix3 stresses;   //!< Stress tensor = derivative of the energy wrt the lattice (almost)

    std::map<std::string, std::string> features;            //!< An std::map that contains all nonstandard features

    std::vector<Neighborhood> nbhs;
    double nbh_cutoff = 0.0;

    std::set<int> ghost_inds;                               // indices of ghost atoms 

    std::set<int> fitting_items;                            //!< additional data required by some Trainers for fitting a MLIP

    inline int size() const;                                //!< returns the number of atoms
    inline void resize(int new_size);                       //!< resizes the configuration (pos always, forces if present, site_energies if present)

    inline bool has_energy() const;                         //!< checks whether there is energy
    inline bool has_stresses() const;                       //!< checks whether there are stresses
    inline bool has_forces() const;                         //!< checks whether there are forces
    inline bool has_magmom() const;                         //!< checks whether there are magnetic moments
    inline bool has_en_der_wrt_magmom() const;              //!< checks whether there are energy derivatives w.r.t. magnetic moments
    inline bool has_site_energies() const;                  //!< checks whether there are site energies
    inline bool has_charges() const;                        //!< checks whether there are charges
    inline bool has_nbh_grades() const;                     //!< checks whether there are has_nbh_grades

    inline void has_energy(bool has_energy);                //!< declares whether or not there is energy
    inline void has_stresses(bool has_stresses);            //!< declares whether or not there are stresses
    inline void has_forces(bool has_forces);                //!< declares whether or not there are forces
    inline void has_magmom(bool has_magmom);                //!< declares whether or not there are magnetic moments
    inline void has_en_der_wrt_magmom(bool has_en_der_wrt_magmom); //!< declares whether or not there are energy derivatives w.r.t. magnetic moments
    inline void has_site_energies(bool has_site_energies);  //!< declares whether or not there are site energies
    inline void has_charges(bool has_charges);              //!< declares whether or not there are charges
    inline void has_nbh_grades(bool has_charges);           //!< declares whether or not there are nbh_grades

    // Functions providing access to main fields of this class
    inline Vector3&        pos(int i);
    inline const Vector3&  pos(int i) const;
    inline double&         pos(int i, int a);
    inline const double&   pos(int i, int a) const;
    inline Vector3&        force(int i);
    inline const Vector3&  force(int i) const;
    inline double&         force(int i, int a);
    inline const double&   force(int i, int a) const;
    inline Vector3&        magmom(int i);
    inline const Vector3&  magmom(int i) const;
    inline double&         magmom(int i, int a);
    inline const double&   magmom(int i, int a) const;
    inline Vector3&        en_der_wrt_magmom(int i);
    inline const Vector3&  en_der_wrt_magmom(int i) const;
    inline double&         en_der_wrt_magmom(int i, int a);
    inline const double&   en_der_wrt_magmom(int i, int a) const;
    inline int&            type(int i);
    inline const int&      type(int i) const;
    inline double&         site_energy(int i);
    inline const double&   site_energy(int i) const;
    inline double&         charges(int i);
    inline const double&   charges(int i) const;
    inline double&         nbh_grades(int i);
    inline const double&   nbh_grades(int i) const;
    inline Vector3         direct_pos(int i) const;
    inline Matrix3         virial() const;

    Configuration();
    //! Construct a perfect crystal by repeating `replication_times` in each coordinate
    Configuration(  CrystalSystemType crystal_system,
                    double basic_lattice_parameter,
                    int replication_times = 1);
    //! Construct a perfect crystal by repeating `replication_times` in each coordinate
    Configuration(  CrystalSystemType crystal_system,
                    double basic_lattice_parameter1,
                    double basic_lattice_parameter2,
                    double basic_lattice_parameter3,
                    int replication_times = 1);
    ~Configuration();
    void destroy();                                 //!< Clears all configuration data and neighborhoods

    void* p_void_pair = nullptr;                    //!< a pointer to lammps pair object. Required if the configuration is constructed from lammps atoms and needed the for ghost-original communication function  

    void InitNbhs(const double _cutoff);            //!< neighborhood initialization procedure
    void ClearNbhs();
    Configuration GetCfgFromLocalEnv(int atom_ind, Matrix3 lattice);    // Returns configuration constructed by a local environment of atom with index 'atom_ind' within cutoff sphere of radius 'cutoff'

    //! compares two configurations. The features vector is allowed to be different
    bool operator==(const Configuration&) const;
    bool operator!=(const Configuration& cfg) const; //!< compares two configurations. The features vector is allowed to be different

    void Deform(const Matrix3& mapping);            //!< Apply linear transformation 'mapping' to the configuration
    void MoveAtomsIntoCell();                       //!< Make fractional coords between 0 and 1 by periodically equivalent transformations
    void CorrectSupercell();                        //!< Does following: 1. Construct an equivalent configuration by making it more orthogonal and make the supercell lower triangle; 2. Returns atoms from outside supercell to interior; 3. Make latice matrix lower triangle; 4. Forces and stresses preserved while transformations
    void ReplicateUnitCell(int, int, int);          //!<Replicates configuration
    void ReplicateUnitCell(templateMatrix3<int>);   //!<Replicates configuration
    void NearestNeighborDistance(std::vector<double>&); //!<Calculates distance between the nearest neighbors of the same type 
    inline std::set<int> GetTypes() const;          //!< returns set of types
    std::map<std::pair<int, int>, double> GetTypeMindists();                      //!< Calculates minimal distance between atoms in configuration
    double MinDist();                               //!< Calculates minimal distance between atoms in configuration
    double MinMagMom();                             //!< Calculates minimal magnetic moment in configuration
    double MaxMagMom();                             //!< Calculates maximum magnetic moment in configuration
    void ToAtomicUnits();                           //!< Converts coordinates, forces, etc., into atomic units
    void FromAtomicUnits();                         //!< Converts coordinates, forces, etc., into atomic units

    // I/O functions
    // Flags for Save()
    static const unsigned int SAVE_BINARY        = 0x1;
    static const unsigned int SAVE_NO_LOSS       = 0x2;
    static const unsigned int SAVE_DIRECT_COORDS = 0x4;
    static const unsigned int SAVE_GHOST_ATOMS   = 0x8;
    void Save(std::ofstream& ofs, unsigned int flags = 0) const;            //!< Saving to a file stream
    bool Load(std::ifstream& ifs);                                          //!< Loading one configuration from text filestream. Returns true if read, false if end-of-file.
    void AppendToFile(const std::string& filename, unsigned int flags = 0) const; //!< Append configuration to a file
    void SaveBin(std::ofstream& ofs) const;                                 //!< Saving, binary format

    // Import/Export functions to the other formats
    void LoadFromOUTCAR(const std::string& filename, bool relative_types = true);                       //!<    reading VASP output
    bool LoadLastFromOUTCAR(const std::string& filename, bool relative_types = true);                   //!<    reading VASP output
    bool LoadNextFromLAMMPSDump(std::ifstream&);                                                        //!<    reading LAMMPS output
    void WriteVaspPOSCAR(const std::string& filename, std::vector<int> types_mapping) const;       //!<    writing POSCAR for VASP
    void WriteLammpsDatafile(const std::string& filename) const;            //!<    writing cfg for LAMMPS
    static bool LoadDynamicsFromOUTCAR(std::vector<Configuration> &db,
                                       const std::string& filename, bool save_nonconverged = false, bool relative_types = true); //! Load all configurations in OUTCAR while relaxation or dynamics, saves into db
};

const std::map<const std::string, const int> MENDELEEV_STRING_TO_NUM = { { "H", 1 },{ "He", 2 },{ "Li", 3 },{ "Be", 4 },{ "B", 5 },{ "C", 6 },{ "N", 7 },{ "O", 8 },{ "F", 9 },{ "Ne", 10 },{ "Na", 11 },{ "Mg", 12 },{ "Al", 13 },{ "Si", 14 },{ "P", 15 },{ "S", 16 },{ "Cl", 17 },{ "Ar", 18 },{ "K", 19 },{ "Ca", 20 },{ "Sc", 21 },{ "Ti", 22 },{ "V", 23 },{ "Cr", 24 },{ "Mn", 25 },{ "Fe", 26 },{ "Co", 27 },{ "Ni", 28 },{ "Cu", 29 },{ "Zn", 30 },{ "Ga", 31 },{ "Ge", 32 },{ "As", 33 },{ "Se", 34 },{ "Br", 35 },{ "Kr", 36 },{ "Rb", 37 },{ "Sr", 38 },{ "Y", 39 },{ "Zr", 40 },{ "Nb", 41 },{ "Mo", 42 },{ "Tc", 43 },{ "Ru", 44 },{ "Rh", 45 },{ "Pd", 46 },{ "Ag", 47 },{ "Cd", 48 },{ "In", 49 },{ "Sn", 50 },{ "Sb", 51 },{ "Te", 52 },{ "I", 53 },{ "Xe", 54 },{ "Cs", 55 },{ "Ba", 56 },{ "La", 57 },{ "Ce", 58 },{ "Pr", 59 },{ "Nd", 60 },{ "Pm", 61 },{ "Sm", 62 },{ "Eu", 63 },{ "Gd", 64 },{ "Tb", 65 },{ "Dy", 66 },{ "Ho", 67 },{ "Er", 68 },{ "Tm", 69 },{ "Yb", 70 },{ "Lu", 71 },{ "Hf", 72 },{ "Ta", 73 },{ "W", 74 },{ "Re", 75 },{ "Os", 76 },{ "Ir", 77 },{ "Pt", 78 },{ "Au", 79 },{ "Hg", 80 },{ "Tl", 81 },{ "Pb", 82 },{ "Bi", 83 },{ "Po", 84 },{ "At", 85 },{ "Rn", 86 },{ "Fr", 87 },{ "Ra", 88 },{ "Ac", 89 },{ "Th", 90 },{ "Pa", 91 },{ "U", 92 },{ "Np", 93 },{ "Pu", 94 },{ "Am", 95 },{ "Cm", 96 },{ "Bk", 97 },{ "Cf", 98 },{ "Es", 99 },{ "Fm", 100 },{ "Md", 101 },{ "No", 102 },{ "Lr", 103 },{ "Rf", 104 },{ "Db", 105 },{ "Sg", 106 },{ "Bh", 107 },{ "Hs", 108 },{ "Mt", 109 },{ "Ds", 110 },{ "Rg", 111 },{ "Cn", 112 },{ "Nh", 113 },{ "Fl", 114 },{ "Mc", 115 },{ "Lv", 116 },{ "Ts", 117 },{ "Og", 118 } };

// Global function loads configurations from file and returns vector of loaded configurations
std::vector<Configuration> LoadCfgs(const std::string& filename, const int max_count=HUGE_INT);
// Global function loads configurations from file, distribute them among processors and returns vector of scatered configurations on each processor
std::vector<Configuration> MPI_LoadCfgs(const std::string& filename, const int max_count=HUGE_INT);


inline std::set<int> Configuration::GetTypes() const
{
    std::set<int> type_set;
    for (int i : types_)
        type_set.insert(i);
    return type_set;
}

inline int Configuration::size() const
{ 
    return (int)pos_.size();
} //!< returns the number of atoms

inline void Configuration::resize(int new_size)
{
    if (is_mpi_splited)
        ERROR("Cannot resize MPI-splitted configuration");
    if (new_size != size()) 
    {
        pos_.resize(new_size);
        types_.resize(new_size);
        if (has_forces()) forces_.resize(new_size);
        if (has_magmom()) magmom_.resize(new_size);
        if (has_en_der_wrt_magmom()) en_der_wrt_magmom_.resize(new_size);
        if (has_site_energies()) site_energies_.resize(new_size);
        if (has_charges()) charges_.resize(new_size);
        nbh_grades_.clear();
        ghost_inds.clear();
        ClearNbhs();
    }
}

inline Vector3& Configuration::pos(int i) {
#ifdef MLIP_DEBUG
    if (i > (int)pos_.size())
        ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
    return pos_[i];
}
inline Vector3 Configuration::direct_pos(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)pos_.size())
        ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
    return pos_[i] * lattice.inverse();
}
inline const Vector3& Configuration::pos(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)pos_.size())
        ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
    return pos_[i];
}
inline Vector3& Configuration::force(int i) {
#ifdef MLIP_DEBUG
    if (i > (int)forces_.size())
        ERROR("i > forces_.size()");
#endif // MLIP_DEBUG
    return forces_[i];
}
inline const Vector3& Configuration::force(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)forces_.size())
        ERROR("i > forces_.size()");
#endif // MLIP_DEBUG
    return forces_[i];
}
inline Vector3& Configuration::magmom(int i) {
#ifdef MLIP_DEBUG
    if (i > (int)magmom_.size())
        ERROR("i > magmom_.size()");
#endif // MLIP_DEBUG
    return magmom_[i];
}
inline const Vector3& Configuration::magmom(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)magmom_.size())
        ERROR("i > magmom_.size()");
#endif // MLIP_DEBUG
    return magmom_[i];
}
inline Vector3& Configuration::en_der_wrt_magmom(int i) {
#ifdef MLIP_DEBUG
    if (i > (int)en_der_wrt_magmom_.size())
        ERROR("i > en_der_wrt_magmom_.size()");
#endif // MLIP_DEBUG
    return en_der_wrt_magmom_[i];
}
inline const Vector3& Configuration::en_der_wrt_magmom(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)en_der_wrt_magmom_.size())
        ERROR("i > en_der_wrt_magmom_.size()");
#endif // MLIP_DEBUG
    return en_der_wrt_magmom_[i];
}
inline double& Configuration::en_der_wrt_magmom(int i, int a) {
#ifdef MLIP_DEBUG
    if (i > (int)en_der_wrt_magmom_.size())
        ERROR("i > en_der_wrt_magmom_.size()");
#endif // MLIP_DEBUG
    return en_der_wrt_magmom_[i][a];
}
inline const double& Configuration::en_der_wrt_magmom(int i, int a) const {
#ifdef MLIP_DEBUG
    if (i > (int)en_der_wrt_magmom_.size())
        ERROR("i > en_der_wrt_magmom_.size()");
#endif // MLIP_DEBUG
    return en_der_wrt_magmom_[i][a];
}
inline int& Configuration::type(int i) {
#ifdef MLIP_DEBUG
    if (i > (int)types_.size())
        ERROR("i > types_.size()");
#endif // MLIP_DEBUG
    return types_[i];
}
inline const int& Configuration::type(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)types_.size())
        ERROR("i > types_.size()");
#endif // MLIP_DEBUG
    return types_[i];
}
inline double& Configuration::site_energy(int i) {
#ifdef MLIP_DEBUG
    if (i > (int)site_energies_.size())
        ERROR("i > site_energies_.size()");
#endif // MLIP_DEBUG
    return site_energies_[i];
}
inline const double& Configuration::site_energy(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)site_energies_.size())
        ERROR("i > site_energies_.size()");
#endif // MLIP_DEBUG
    return site_energies_[i];
}
inline double& Configuration::charges(int i) {
#ifdef MLIP_DEBUG
    if (i > (int)charges_.size())
        ERROR("i > charges_.size()");
#endif // MLIP_DEBUG
    return charges_[i];
}
inline const double& Configuration::charges(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)charges_.size())
        ERROR("i > charges_.size()");
#endif // MLIP_DEBUG
    return charges_[i];
}
inline double& Configuration::nbh_grades(int i) {
#ifdef MLIP_DEBUG
    if (i > (int)nbh_grades_.size())
        ERROR("i > nbh_grades_.size()");
#endif // MLIP_DEBUG
    return nbh_grades_[i];
}
inline const double& Configuration::nbh_grades(int i) const {
#ifdef MLIP_DEBUG
    if (i > (int)nbh_grades_.size())
        ERROR("i > nbh_grades_.size()");
#endif // MLIP_DEBUG
    return nbh_grades_[i];
}
inline double& Configuration::pos(int i, int a) {
#ifdef MLIP_DEBUG
    if (i > (int)pos_.size())
        ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
    return pos_[i][a];
}
inline const double& Configuration::pos(int i, int a) const {
#ifdef MLIP_DEBUG
    if (i > (int)pos_.size())
        ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
    return pos_[i][a];
}
inline double& Configuration::force(int i, int a) {
#ifdef MLIP_DEBUG
    if (i > (int)forces_.size())
        ERROR("i > forces_.size()");
#endif // MLIP_DEBUG
    return forces_[i][a];
}
inline const double& Configuration::force(int i, int a) const {
#ifdef MLIP_DEBUG
    if (i > (int)forces_.size())
        ERROR("i > forces_.size()");
#endif // MLIP_DEBUG
    return forces_[i][a];
}
inline Matrix3 Configuration::virial() const {
    double vol = fabs(lattice.det());
    return stresses * (1/vol);
}

inline bool Configuration::has_energy() const           //!< checks whether there is energy
{ 
    return has_energy_; 
}                               
inline void Configuration::has_energy(bool has_energy)  //!< declares whether or not there is energy
{ 
    has_energy_ = has_energy; 
}               

inline bool Configuration::has_stresses() const 
{ 
    return has_stresses_; 
}                           //!< checks whether there are stresses
inline void Configuration::has_stresses(bool has_stresses) 
{ 
    has_stresses_ = has_stresses; 
}       //!< declares whether or not there are stresses

inline bool Configuration::has_forces() const 
{ 
    return !forces_.empty(); 
}                               //!< checks whether there are forces
inline void Configuration::has_forces(bool has_forces) 
{ 
    forces_.resize(has_forces ? size() : 0); 
}                               //!< declares whether or not there are forces

inline bool Configuration::has_site_energies() const 
{ 
    return !site_energies_.empty(); 
}                               //!< checks whether there are site energies
inline void Configuration::has_site_energies(bool has_site_energies)
{
    site_energies_.resize(has_site_energies ? size() : 0);
}                               //!< declares whether or not there are site energies

inline bool Configuration::has_charges() const
{
    return !charges_.empty();
}                               //!< checks whether there are nbh_grades
inline void Configuration::has_charges(bool has_charges)
{
    charges_.resize(has_charges ? size() : 0);
}                               //!< declares whether or not there are has_nbh_grades

inline bool Configuration::has_nbh_grades() const
{
    return !nbh_grades_.empty();
}                               //!< checks whether there are nbh_grades
inline void Configuration::has_nbh_grades(bool has_charges)
{
    nbh_grades_.resize(has_charges ? size() : 0);
}                               //!< declares whether or not there are has_nbh_grades

inline bool Configuration::has_magmom() const
{ 
    return !magmom_.empty(); 
}                               //!< checks whether there are magmom
inline void Configuration::has_magmom(bool has_magmom) 
{ 
    magmom_.resize(has_magmom ? size() : 0); 
}                               //!< declares whether or not there are en_der_wrt_magmom

inline bool Configuration::has_en_der_wrt_magmom() const 
{ 
    return !en_der_wrt_magmom_.empty(); 
}                               //!< checks whether there are en_der_wrt_magmom
inline void Configuration::has_en_der_wrt_magmom(bool has_en_der_wrt_magmom) 
{ 
    en_der_wrt_magmom_.resize(has_en_der_wrt_magmom ? size() : 0); 
}                               //!< declares whether or not there are en_der_wrt_magmom


#endif // MLIP_CONFIGURATION_H
