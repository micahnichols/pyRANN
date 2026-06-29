import numpy as np
import numpy.typing as npt
from numpy.random import rand

from numpy import ceil,floor,sign,array,zeros,round,pi, concatenate, ones, arccos
from numpy.linalg import inv,det,norm
from numpy import sqrt,cross,dot,array

from scipy.linalg import sqrtm,eig#,norm

from copy import deepcopy
import copy
import random

from typing import Any, Optional, Union, Tuple

# from .series import series
from . import series

class system:
    r"""
    A class representing a physical system with atoms, box dimensions, and various properties.

    Initializes a system with atom positions, types, box dimensions, and other optional properties.

    Parameters
    ----------
    box 
        The 3x3 matrix representing the simulation box. Default is the identity matrix.
    atoms
        A 3xN array representing the atom positions. This is required.
    types
        Atom types, either as a string, integer, or list. Default is an array of ones.
    elements
        Elements corresponding to atom types. If None, elements are generated.
    natoms
        Number of atoms in the system. If None, derived from `atoms`.
    timestep
        The current timestep of the simulation. Default is 0.
    energy
        The energy of the system in :math:`eV`.
    stress
        A 3x3 stress tensor array. Default is None.
    force
        A 3xN force array. Default is None.
    descriptor
        A description of the system. Default is None.

    Raises
    ------
    ValueError
        If `atoms` is not provided.
    """

    def __init__(self,
                 box: Optional[npt.ArrayLike] = None,
                 atoms: Optional[npt.ArrayLike] = None,
                 types: Optional[Union[str, int, npt.ArrayLike]] = None,
                 elements: Optional[Union[str, list]] = None,
                 natoms: Optional[Union[str, int]] = None,
                 timestep: Optional[Union[str, int]] = None,
                 energy: Optional[Union[str, float]] = None,
                 stress: Optional[Union[str, npt.ArrayLike]] = None,
                 force: Optional[Union[str, npt.ArrayLike]] = None,
                 descriptor: Optional[str] = None):

        self.box = box
        self._original_box = box
        self.atoms = atoms
        self._original_atoms = atoms
        self.types = types
        self._original_types = types
        self.elements = elements
        self._original_elements =elements
        self.natoms = natoms
        self._original_natoms = natoms
        self.timestep = timestep
        self._original_timestep = timestep
        self.energy = energy
        self._original_energy = energy
        self.stress = stress
        self._original_stress = stress
        self.force = force
        self._original_force = force
        self.descriptor = descriptor
        self._original_descriptor = descriptor

    def __str__(self) -> str:
        """
        Returns a string representation of the system.

        Returns
        -------
        str: A formatted string describing the system's box,
            atom positions, types, and other properties.
        """
        return '\n'.join(['box and atoms are column vectors',
                          f'System created from {self.descriptor}',
                          f'box = \n{self.box}',
                          f'natoms = {self.natoms}',
                          f'elements = {self.elements}',
                          f'timestep = {self.timestep}',
                          f'types = {self.types}',
                          f'atoms = \n{self.atoms}',
                          f'energy = {self.energy}',
                          f'forces = \n{self.force}',
                          f'stress = \n{self.stress}'])

    @property
    def box(self) -> np.ndarray:
        """
        Returns the system's box vectors.

        Returns
        -------
        np.ndarray: A 3x3 matrix representing the simulation box where x,
            y, and z are column vectors.
        """
        return self.__box

    @box.setter
    def box(self, value: Optional[npt.ArrayLike] = None):
        """
        Sets the system's box dimensions.

        Parameters
        ----------
        value: A 3x3 array representing the new box dimenstions
            dimensions.

        Raises
        ------
        AssertionError
            If the box is not a 3x3 matrix.
        """
        if value is not None:
            box = np.asarray(value)
        else:
            box = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        box = np.asarray(value, dtype=np.float64)
        assert box.shape == (3,3), 'Box must be 3x3 matrix'
        self.__box = box

    @property
    def atoms(self) -> np.ndarray:
        """
        Returns (3, natoms) shaped np.ndarray of the atom positions.

        Returns
        -------
        np.ndarray
            A 3xN array representing the positions of the atoms.
        """
        return self.__atoms

    @atoms.setter
    def atoms(self, value: Optional[npt.ArrayLike] = None):
        """
        Sets the atom positions.

        Parameters
        ----------
        value
            A 3xN array representing the new atom positions.

        Raises
        ------
        AssertionError
            If the atom array does not have 3 rows (representing 3D coordinates).
        """
        if value is not None:
            atoms = np.asarray(value)
        else:
            raise ValueError('System class must have atom positions')
        atoms = np.asarray(value, dtype=np.float64)
        assert atoms.shape[0] == 3, 'Atoms must be column vectors'
        self.__atoms = atoms

    @property
    def types(self) -> np.ndarray:
        """
        Returns the atom types.

        Returns
        -------
        np.ndarray
            A 1D array representing the atom types.
        """
        return self.__types

    @types.setter
    def types(self, value: Optional[Union[str, int, npt.ArrayLike]] = None):
        """
        Sets the atom types.

        Parameters
        ----------
        value
            Atom types, either as a string, integer, or array.
        """
        types = value
        if types is None:
            types = np.ones(self.atoms.shape[1], dtype=int)
        else:
            types = np.asarray(types, dtype=int)
        self.__types = types

    @property
    def elements(self) -> Optional[npt.ArrayLike]:
        """
        Returns the element types of the system. 
        If element types are unknown, the element type becomes "Type_n" where n is an integer
        representing the atom type.

        Returns
        -------
        np.ndarray
            Array of element types.
        """
        return self.__elements

    @elements.setter
    def elements(self, value: Optional[Union[str, npt.ArrayLike]] = None):
        """
        Sets the element types of the system with a string representation.
        If None, elements is set to an array of strings of "Type_n"
        where n is the element type integer.

        Parameters
        ----------
        value
            1-D array of element type strings. If None, array is generated.
        """
        if value is None:
            elements = np.asarray(self.types, dtype=str)
            elements = np.unique(elements)
            elements = np.array([f'Type_{i}' for i in elements])
        else:
            elements = np.asarray(value)
        self.__elements = elements

    @property
    def natoms(self) -> int:
        """
        Returns the number of atoms in the system.

        Returns
        -------
        int
            The number of atoms in the system.
        """
        return self.__natoms

    @natoms.setter
    def natoms(self, value: Optional[Union[str, int]] = None):
        """
        Sets the number of atoms in the system.

        Parameters
        ----------
        value
            The number of atoms in the system.
        """
        if value is None:
            natoms = self.atoms.shape[1]
        else:
            natoms = value
        self.__natoms = natoms

    @property
    def timestep(self) -> int:
        """
        Returns the timestep.

        Returns
        -------
        int
            The integer value of the timestep.
        """
        return self.__timestep

    @timestep.setter
    def timestep(self, value: Optional[Union[str, int]] = None):
        """
        Sets the timestep.

        Parameters
        ----------
        value
            Integer timestep
        """
        if value is None:
            timestep = 0
        timestep = value
        self.__timestep = timestep

    @property
    def energy(self) -> np.float64:
        """
        Returns the energy of the system

        Returns
        -------
        np.float64
            Float value of the energy.
        """
        return self.__energy

    @energy.setter
    def energy(self, value: Optional[Union[str, float]] = None):
        """
        Sets the system energy

        Parameters
        ----------
        value
            Float value of energy.
        """
        if value is not None:
            energy = value
        else:
            energy = 0
        self.__energy = energy

    @property
    def stress(self) -> Optional[npt.ArrayLike]:
        """
        Returns the stress tensor of the system if applicable.
        Returns None otherwise.

        Returns
        -------
        np.ndarray
            A 3x3 matrix representing the stress tensor.
        """
        return self.__stress

    @stress.setter
    def stress(self, value: Optional[Union[str, npt.ArrayLike]] = None):
        """
        Sets the stress of the system.

        Parameters
        ----------
        value
            3x3 Stress tensor of the system.
        """
        if value is not None:
            stress = np.asarray(value, dtype=np.float64)
            self._do_stress = True
        else:
            stress = None
            self._do_stress = False
        self.__stress = stress

    @property
    def force(self) -> Optional[npt.ArrayLike]:
        """
        Returns the x, y, and z force components of each atom if applicable.
        Returns None otherwise.

        Returns
        -------
        np.ndarray
            An array of shape (3, natoms) of the x, y, and z force components for each atom.
        """
        return self.__force

    @force.setter
    def force(self, value: Optional[Union[str, npt.ArrayLike]] = None):
        """
        Sets the force for each atom in the system.

        Parameters
        ----------
        value
            ArrayLike item representing the x, y, and z force components for each atom. 
            Must be of shape (3, natoms). If None, forces are not known.
        """
        if value is not None:
            force = np.asarray(value)
            assert force.shape == (3, self.natoms), (
                    'force must be of shape (3, natoms)')
            self._do_force = True
        else:
            force = None
            self._do_force = False
        self.__force = force

    @property
    def descriptor(self) -> Optional[str]:
        """
        Returns a description of the system. Helpful to determine the file loaded from or operations done on the system.

        Returns
        -------
        str
            User defined string describing the system.
        """
        return self.__descriptor

    @descriptor.setter
    def descriptor(self, value: Optional[str] = None):
        """
        Sets the description of the system.

        Parameters
        ----------
        value
            User defined string of the system.
        """
        self.__descriptor = value

    def createsupercell(self, new_box: Union[npt.ArrayLike, None] = None):
        """
        Creates a supercell by replicating the atoms in the original unit cell and placing them within a new box.

        This method takes the original box and atom positions and generates a new supercell with atoms placed according
        to the new box dimensions. It accounts for periodic boundary conditions and generates a new set of atom types
        and positions based on the replication.

        Parameters
        ----------
        new_box
            Column vector array of new box dimensions.

        Returns
        -------
        None
            This method updates the system's attributes: `types`, `atoms`, `box`, and `natoms`
            based on the new supercell.

        Raises
        ------
        ValueError
            If any unexpected issues arise during supercell creation.
        """

        avg_box_x = np.mean(self.box[0,:]) * (np.pi/1000)
        avg_box_y = np.mean(self.box[1,:]) * (np.pi/1000)
        avg_box_z = np.mean(self.box[2,:]) * (np.pi/1000)
        avg_box = np.array([avg_box_x, avg_box_y, avg_box_z])
        # self.atoms += avg_box
        self.atoms[0,:] += avg_box_x
        self.atoms[1,:] += avg_box_y
        self.atoms[2,:] += avg_box_z
        Direct = inv(self.box)@self.atoms
        Direct = Direct%1.0
        # Direct -= floor(Direct)
        new_atoms_matrix = self.box@Direct
        self.atoms = self.box@Direct
        if new_box is None:
            new_box = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        else:
            new_box = np.asarray(new_box)

        units = ceil(abs(inv(self.box)@new_box))*sign(inv(self.box)@new_box)
        vol = int(det(inv(self.box)@new_box)*1.1)
        A = zeros((3,(vol+10)*self.atoms.shape[1]))
        Types = zeros(((vol+10)*self.atoms.shape[1]))
        n=0
        C000 = array([0,0,0]).T
        C100 = units[:,0]
        C010 = units[:,1]
        C001 = units[:,2]
        C110 = C100+C010
        C101 = C100+C001
        C011 = C010+C001
        C111 = C001+C110
        "min to max values of the box corners give us the bounds for the make lattice command"
        Corners = array([C000,C100,C010,C001,C110,C101,C011,C111]).T   
        bounds = round(array([Corners.min(axis=1),Corners.max(axis=1)]))
        bounds[0,:] -= 2
        bounds[1,:] += 2
        for i in range(int(bounds[0,0]),int(bounds[1,0])):
            for j in range(int(bounds[0,1]),int(bounds[1,1])):
                for k in range(int(bounds[0,2]),int(bounds[1,2])):
                    for l in range(self.atoms.shape[1]):

                        # avg_box_x = np.mean(self.box[:,0]) * (np.pi/1000)
                        # avg_box_y = np.mean(self.box[:,1]) * (np.pi/1000)
                        # avg_box_z = np.mean(self.box[:,2]) * (np.pi/1000)
                        # self.atoms[:,0] += avg_box_x
                        # self.atoms[:,1] += avg_box_y
                        # self.atoms[:,2] += avg_box_z
                        # Direct = inv(self.box)@self.atoms
                        # Direct -= floor(Direct)
                        # self.atoms = self.box@Direct
                        x = self.box[:,0]*i+self.box[:,1]*j+self.box[:,2]*k+self.atoms[:,l]
                        if all(inv(new_box)@x<1) and all(inv(new_box)@x>=0):
                            A[:,n]=x
                            Types[n]=self.types[l]
                            n=n+1  

        self.box = new_box+0
        self._original_box = self.box+0
        # self._original_box = new_box+0
        # avg_box_x = np.mean(self.box[:,0]) * (np.pi/10000)
        # avg_box_y = np.mean(self.box[:,1]) * (np.pi/10000)
        # avg_box_z = np.mean(self.box[:,2]) * (np.pi/10000)

        Types = Types[0:n]

        self.types = Types
        self._original_types = Types+0
        # temp_atoms = A[:,0:n]+0
        # temp_atoms[:,0] += avg_box_x
        # temp_atoms[:,1] += avg_box_y
        # temp_atoms[:,2] += avg_box_z
        # Direct = inv(self.box)@self.atoms
        # Direct -= floor(Direct)
        # temp_atoms = self.box@Direct
        # self.atoms = temp_atoms+0
        self.atoms = A[:,0:n]+0
        # self.atoms[:,0] += avg_box_x
        # self.atoms[:,1] += avg_box_y
        # self.atoms[:,2] += avg_box_z
        # Direct = inv(self.box)@self.atoms
        # Direct -= floor(Direct)
        # self.atoms = self.box@Direct
        # self._original_atoms = A[:,0:n]+0
        self._original_atoms = self.atoms+0
        self.natoms = self.atoms.shape[1]
        self._original_natoms = self.atoms.shape[1]+0

    def export(self,
               filename: str,
               filetype: Union[str, None] = None,
               style: Union[str, None] = None):
        """
        Exports the system data to a specified file in a given format.

        This method allows the user to export the system's atoms and box dimensions to various file formats,
        such as POSCAR, LAMMPS data, and others. The user can choose the output file's name, type, and
        coordinate style (e.g., Cartesian, direct, or crystal).

        Parameters
        ----------
        filename
            String of the filename to write.
        filetype
            String of the file extension. Not needed if extension is given in filetype.
            filetypes are 'poscar', 'pos', 'data', 'dat', 'dump', 'qe', 'quantum_espresso', 'vasp', or 'INCAR'
            Note: Some of these filetypes may not be implemented yet.
        style
            String of type of atom coordinates desired (direct, crystal, or cartesian).

        Returns
        -------
        None
            This method writes the system data to the specified file in the given format.

        Raises
        ------
        ValueError
            If an invalid `filetype` or `style` is provided.

        Notes
        ----
        Todo
            qe
                Implement Quantum Expresso
            vasp
                Implement VASP
        """

        if filename is None:
            filename = 'Atomic_System'
        else:
            if filetype is None:
                ext_keys = ['poscar', 'pos', 'data', 'dat', 'dump', 'qe', 'quantum_espresso', 'vasp', 'INCAR']
                if filename.split('.')[-1].lower() in ext_keys:
                    filetype = filename.split('.')[-1]
                else:
                    raise ValueError(f'Either filetype must be given, '
                    f'or the file extension must be one of the following:\n{ext_keys}')
            else:
                filename = '.'.join([filename, filetype])
        if style is None:
            style = 'cartesian'
        else:
            styles = np.array(['direct', 'crystal', 'cartesian'])
            if style.lower() not in styles:
                raise ValueError('Style must be direct, crystal, or cartesian')
            else:
                pass

        if filetype.lower() == 'pos':
            filetype = 'poscar'
        elif filetype.lower() == 'dat':
            filetype = 'data'
        elif filetype.lower() == 'quantum_espresso':
            filetype = 'qe'
        # TODO - ADD IN INFORMATION FOR QE AND VASP?
        elif filetype.lower() == 'vasp':
            # filetype = 'INCAR'
            filetype = 'poscar'
        elif filetype.lower() == 'incar':
            filetype = 'poscar'


        if filetype.lower() == 'poscar':
            file = open(filename,"w")
            # natoms = Lattice.shape[1]
            file.write("#Atomistic data file created by python\n")
            file.write("1.0\n")
            len_col0 = max([len(str(f'{self.box[0,i]:+.16f}')) for i in range(3)])
            len_col1 = max([len(str(f'{self.box[1,i]:+.16f}')) for i in range(3)])
            len_col2 = max([len(str(f'{self.box[2,i]:+.16f}')) for i in range(3)])
            # file.write("   %.16f   %.16f   %.16f\n" % (self.box[0,0],self.box[1,0],self.box[2,0]))
            # file.write("   %.16f   %.16f   %.16f\n" % (self.box[0,1],self.box[1,1],self.box[2,1]))
            # file.write("   %.16f   %.16f   %.16f\n" % (self.box[0,2],self.box[1,2],self.box[2,2]))
            file.write(f'  {self.box[0,0]:>{len_col0}.16f}'
                       f'   {self.box[1,0]:>{len_col1}.16f}'
                       f'   {self.box[2,0]:>{len_col2}.16f}\n')
            file.write(f'  {self.box[0,1]:>{len_col0}.16f}'
                       f'   {self.box[1,1]:>{len_col1}.16f}'
                       f'   {self.box[2,1]:>{len_col2}.16f}\n')
            file.write(f'  {self.box[0,2]:>{len_col0}.16f}'
                       f'   {self.box[1,2]:>{len_col1}.16f}'
                       f'   {self.box[2,2]:>{len_col2}.16f}\n')
            # typesum = []
            unique_types = np.unique(self.types)
            typesum = np.array([len(self.types[self.types==i]) for i in unique_types])
            element_str = ' '.join(self.elements)
            element_int_str = ' '.join(map(str, typesum))
            file.write(f'{element_str}\n')
            file.write(f'{element_int_str}\n')
            # for i in range(len(self.elements)):
            #     file.write(self.elements[i]+' ')
                # typesum.append(sum(self.types==i+1))
            # file.write("\n")
            # for i in range(len(self.elements)):
            #     file.write("%d " % typesum[i])
            # file.write("\n")
            # file.write("%s\n" % style)
            file.write(f'{style}\n')
            if (style=='direct' or style=='Direct' or style=='Crystal' or style == 'crystal'):
                Lattice = inv(self.box)@self.atoms
                Lattice = Lattice%1.0
            else:
                Lattice = self.atoms
            I = np.argsort(self.types)
            for i in range(self.natoms):
                # file.write("\t {0:f} {1:f} {2:f} {3:s}\n" .format(Lattice[0,I[i]],
                #                                                   Lattice[1,I[i]],
                #                                                   Lattice[2,I[i]],
                #                                                   self.elements[int(self.types[I[i]])-1]))
                # file.write("   %.16f   %.16f   %.16f %s\n" % (Lattice[0,I[i]],
                #                                           Lattice[1,I[i]],
                #                                           Lattice[2,I[i]],
                #                                           self.elements[int(self.types[I[i]])-1]))
                file.write(f'  {Lattice[0,I[i]]:>{len_col0}.16f}'
                           f'   {Lattice[1,I[i]]:>{len_col1}.16f}'
                           f'   {Lattice[2,I[i]]:>{len_col2}.16f}'
                           f' {self.elements[int(self.types[I[i]])-1]}\n')
            file.close()

        elif filetype == 'data':
            (xlo,ylo,zlo)=(0,0,0)
            xhi = norm(self.box[:,0])
            xy = dot(self.box[:,1],self.box[:,0]/norm(self.box[:,0]))
            yhi = norm(cross(self.box[:,0]/norm(self.box[:,0]),self.box[:,1]))
            xz = dot(self.box[:,2],self.box[:,0]/norm(self.box[:,0]))
            yz = (dot(self.box[:,2],self.box[:,1])-xy*xz)/yhi
            zhi = sqrt(dot(self.box[:,2],self.box[:,2])-xz**2-yz**2)
            new_box = array([[xlo, xhi],[ylo, yhi],[zlo, zhi]])
            tilt = array([xy, xz, yz]);
            R = (array([[xhi, xy, xz],[0, yhi, yz],[0, 0, zhi]])
                 @array([cross(self.box[:,1],self.box[:,2]),
                         cross(self.box[:,2],self.box[:,0]),
                         cross(self.box[:,0],self.box[:,1])])
                 /det(array([[xhi, xy, xz],[0, yhi, yz],[0, 0, zhi]])))
            x = R@self.atoms
            lammpsbox = new_box
            # (atoms,lammpsbox,tilt,R)=triclinic2lammps(Atoms,Box)
            file = open(filename,"w")
            # natoms = atoms.shape[1]
            ntypes = len(np.unique(self.types))
            file.write("#Atomistic data file created by python\n\n")
            file.write("%d\tatoms\n\n" % self.natoms)
            file.write("%d\t atom types\n\n" % ntypes)
            file.write("%.16f   %.16f xlo xhi\n" % (lammpsbox[0,0],lammpsbox[0,1]))
            file.write("%.16f   %.16f ylo yhi\n" % (lammpsbox[1,0],lammpsbox[1,1]))
            file.write("%.16f   %.16f zlo zhi\n" % (lammpsbox[2,0],lammpsbox[2,1]))
            file.write("%.16f   %.16f  %.16f xy xz yz\n" % (tilt[0],tilt[1],tilt[2]))
            file.write("Atoms\n\n")
            for i in range(self.natoms):
                file.write("\t {0:d} {1:.0f} {2:f} {3:f} {4:f}\n" .format(i+1,self.types[i],x[0,i],x[1,i],x[2,i]))
            file.close()

        elif filetype == 'dump':
            (xlo,ylo,zlo)=(0,0,0)
            xhi = norm(self.box[:,0])
            xy = dot(self.box[:,1],self.box[:,0]/norm(self.box[:,0]))
            yhi = norm(cross(self.box[:,0]/norm(self.box[:,0]),self.box[:,1]))
            xz = dot(self.box[:,2],self.box[:,0]/norm(self.box[:,0]))
            yz = (dot(self.box[:,2],self.box[:,1])-xy*xz)/yhi
            zhi = sqrt(dot(self.box[:,2],self.box[:,2])-xz**2-yz**2)
            new_box = array([[xlo, xhi],[ylo, yhi],[zlo, zhi]])
            tilt = array([xy, xz, yz]);
            R = (array([[xhi, xy, xz],[0, yhi, yz],[0, 0, zhi]])
                 @array([cross(self.box[:,1],self.box[:,2]),
                         cross(self.box[:,2],self.box[:,0]),
                         cross(self.box[:,0],self.box[:,1])])
                 /det(array([[xhi, xy, xz],[0, yhi, yz],[0, 0, zhi]])))
            x = R@self.atoms
            lammpsbox = new_box
            if tilt[0]>0:
                lammpsbox[0][1] += tilt[0]
            else:
                lammpsbox[0][0] += tilt[0]
            if tilt[1]>0:
                lammpsbox[0][1] += tilt[1]
            else:
                lammpsbox[0][0] += tilt[1]
            if tilt[2]>0:
                lammpsbox[1][1] += tilt[2]
            else:
                lammpsbox[1][0] += tilt[2]

            file = open(filename,"w")
            file.write("ITEM: TIMESTEP\n")
            file.write("%d\n" % (self.timestep))
            file.write("ITEM: NUMBER OF ATOMS\n")
            # natoms=Atoms.shape[1]
            file.write("%d\n" % self.natoms)
            origin = np.zeros((3))
            # (x,box,tilt,R)=triclinic2lammps(Atoms,Box)
            # if self.stress is None:
            #     do_stress = False
            # else:
            #     do_stress = True
            # if self.force is None:
            #     do_force = False
            # else:
            #     do_force = True
            if self._do_stress:
                file.write("ITEM: BOX BOUNDS xy xz yz pp pp pp stress\n")
            else:
                file.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            for i in range(3):
                if self._do_stress:
                    file.write("%.16f %.16f %.16f %.16f %.16f %.16f\n" % (lammpsbox[i,0],lammpsbox[i,1],tilt[i],
                                                        self.stress[i,0],self.stress[i,1],self.stress[i,2]))
                else:
                    file.write("%.16f %.16f %.16f\n" % (lammpsbox[i,0],lammpsbox[i,1],tilt[i],))
            # TODO - ADD IN STRESS ATTRIBUTES
            # file.write("ITEM: ATOMS id type x y z fx fy fz\n")
            if not self._do_force:
                file.write("ITEM: ATOMS id type x y z\n")
            else:
                file.write("ITEM: ATOMS id type x y z fx fy fz\n")
            for i in range(self.natoms):
                if not self._do_force:
                    file.write("%d %d %.16f %.16f %.16f\n" % (i+1, self.types[i], x[0,i], x[1,i], x[2,i]))
                else:
                    file.write("%d %d %.16f %.16f %.16f %.16f %.16f %.16f\n" % (i+1, self.types[i], x[0,i], x[1,i], x[2,i],
                                                              self.force[0,i], self.force[1,i], self.force[2,i]))
            file.close()

    def thermal_wave(self,
                     amplitude: float,
                     a: Union[float, None] = None,
                     c: Union[float, None] = None,
                     wavelength: Union[float, int, None] = None,
                     frequency: Union[float, int, None] = None,
                     alpha: Union[float, int, None] = None,
                     direction: Union[npt.ArrayLike, None] = None,
                     nsims: Union[str, int, None] = 100,
                     shear: Union[str, int, float, None] = 0,
                     N: int = None,
                     ss_dict: Union[dict, None] = None):
        """
        Simulates thermal perturbations on the atomic system by applying random displacements and box strain.

        This method generates a series of configurations by thermally perturbing the atomic
            positions and box dimensions. The perturbations are applied to a base system, and 
            the resulting atomic configurations are stored in a series for further analysis or export.

        Args:
            temp: Maximum random perturbation of atoms.
                Currently, this should be a percentage of the given simulation cell's lattice parameter (in decimal
                format).
            strain: Maximum strain given as a percentage (in decimal format) of the given simulation
                cell's lattice parameter.
            nsims: The number of unique structures to produce.
            N: Number of atoms - Ignore this. Not sure why I made this an argument at the moment.
            ss_dict: dictionary of keywords for doing thermal perturbation on solid solution.
                ss_keys = ['solute_type', 'num_neighbors', 'num_atoms']

        Returns:
            series: A series containing the generated configurations, each with perturbed
                atomic positions and box dimensions.

        Raises:
            ValueError: If the number of simulations (`nsims`) is not provided or invalid
                keywords are found in `ss_dict`.
        """
        # old_atoms = deepcopy(self.atoms)
        # old_box = deepcopy(self.box)
        # old_types = deepcopy(self.types)
        # old_types = copy.copy(self.types)
        # old_natoms = deepcopy(self.natoms)
        # old_elements = deepcopy(self.elements)
        if direction is None:
            direction = np.random.normal(size=3)
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.asarray(direction)
            assert direction.shape == (3,), 'The direction vector must have a shape of (3,).'
            direction = direction / np.linalg.norm(direction)
        if wavelength is None and a is None and c is None:
            box_project = np.abs(np.dot(self.box.T, direction))
            wavelength = np.sum(box_project)
        elif wavelength is None and a is not None:
            if c is not None:
                # print(f'working correctly\n')
                box_x = a * (self.box[:,0] / np.linalg.norm(self.box[:,0]))
                box_y = a * (self.box[:,1] / np.linalg.norm(self.box[:,1]))
                box_z = c * (self.box[:,2] / np.linalg.norm(self.box[:,2]))
                temp_box = np.array([box_x, box_y, box_z]).T
                box_project = np.abs(np.dot(temp_box.T, direction))
                wavelength = np.sum(box_project)
            else:
                box_x = a * (self.box[:,0] / np.linalg.norm(self.box[:,0]))
                box_y = a * (self.box[:,1] / np.linalg.norm(self.box[:,1]))
                box_z = a * (self.box[:,2] / np.linalg.norm(self.box[:,2]))
                temp_box = np.array([box_x, box_y, box_z]).T
                box_project = np.abs(np.dot(temp_box.T, direction))
                wavelength = np.sum(box_project)

        if frequency is None:
            frequency = 1.0
        if alpha is None:
            alpha = 1.0
        if nsims is None:
            raise ValueError('Please specify the number of structures to create by '
            'thermally perturbing the base system.')
        else:
            nsims = int(nsims)
        if N is None:
            N = self.natoms
        if ss_dict is not None:
            ss_keys = ['solute_type', 'num_neighbors', 'num_atoms']
            good_keys = []
            for i in ss_dict.keys():
                if i not in ss_keys:
                    raise ValueError(f'{i} is not a valid solidsolution keyword.\nValid keywords are {ss_keys}')
                else:
                    if i == 'solute_type':
                        solute_type = ss_dict[i]
                    elif i == 'num_neighbors':
                        num_neighbors = ss_dict[i]
                    elif i == 'num_atoms':
                        num_atoms = ss_dict[i]
                    # elif i == 'replicate':
                    #     replicate = ss_dict[i]
                    else:
                        solute_type = None
                        num_neighbors = 1
                        num_atoms = 1
                        # replicate = None
            # self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, replicate=replicate)


        inv_box = inv(self.box)
        series_list = []
        t = np.linspace(0, 1, nsims)
        for i in range(1, nsims+1):
            self.atoms = self._original_atoms+0
            self.box = self._original_box+0
            self.types = self._original_types+0
            self.natoms = self._original_natoms+0
            self.elements = self._original_elements
            k = 2 * np.pi / wavelength
            omega = 2 * np.pi * frequency
            projections = np.dot(self.atoms.T, direction)
            displacement_magnitudes = amplitude * np.exp(-alpha*t[i-1]) * np.sin(k * projections - omega * t[i-1])
            displacements = displacement_magnitudes[:, np.newaxis] * direction
            self.atoms += displacements.T
            direct = inv_box@self.atoms
            wrap_direct = direct%1.0
            self.atoms = self.box@wrap_direct
            series_list.append(system(atoms=self.atoms+0, box=self.box+0, types=self.types+0,
                                      elements=self.elements, natoms=self.natoms+0, timestep=i))


        # for i in range(1,nsims+1):

        #     # self.atoms = old_atoms
        #     # self.box = old_box
        #     # self.types = old_types
        #     # self.natoms = old_natoms
        #     # self.elements = old_elements
        #     self.atoms = self._original_atoms+0
        #     self.box = self._original_box+0
        #     self.types = self._original_types+0
        #     self.natoms = self._original_natoms+0
        #     self.elements = self._original_elements

        #     # self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, replicate=replicate)
        #     if ss_dict is not None:
        #         self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, num_atoms=num_atoms)

        #     rng = np.random.default_rng()
        #     ss = rng.uniform(size=6, low=np.array([-1.,-1.,-1.,-1.,-1.,-1.])*strain, high=np.array([1.,1.,1.,1.,1.,1.])*strain)
        #     shear_strain = rng.uniform(size=6, low=np.array([-1.,-1.,-1.,-1.,-1.,-1.])*shear, high=np.array([1.,1.,1.,1.,1.,1.])*shear)
        #     # ss = (rand(6)-0.5)*strain*2
        #     # box1 = new_box+0
        #     # self.box = self.box+0
        #     xmag = norm(self.box[:,0])
        #     ymag = norm(self.box[:,1])
        #     zmag = norm(self.box[:,2])
        #     self.box[0,0] += ss[0]*xmag
        #     self.box[1,1] += ss[1]*ymag
        #     self.box[2,2] += ss[2]*zmag
        #     self.box[0,1] += shear_strain[3]*ymag
        #     self.box[0,2] += shear_strain[4]*zmag
        #     self.box[1,2] += shear_strain[5]*zmag
        #     # self.box[0,1] += ss[3]*ymag
        #     # self.box[0,2] += ss[4]*zmag
        #     # self.box[1,2] += ss[5]*zmag

        #     # self.box[0,0] += ss[0]*xmag
        #     # self.box[1,1] += ss[1]*ymag
        #     # self.box[2,2] += ss[2]*zmag
        #     # self.box[0,1] += ss[3]*ymag
        #     # self.box[0,2] += ss[4]*zmag
        #     # self.box[1,2] += ss[5]*zmag
        #     # self.atoms = self.box@(inv(old_box)@old_atoms)

        #     self.atoms = self.box@(inv(self._original_box)@self._original_atoms)

        #     seq = list(range(0,self.natoms))
        #     index = random.sample(seq,N)
        #     for j in range(N):
        #         rng = np.random.default_rng()
        #         vec = rng.uniform(size=3, low=np.array([-1.,-1.,-1.]), high=np.array([1.,1.,1.]))
        #         # x = (random.random()-0.5)
        #         # y = (random.random()-0.5)
        #         # z = (random.random()-0.5)
        #         # vec = np.array([x, y, z])*temp/np.sqrt(3)*2
        #         vec *= temp/np.sqrt(3)*2
        #         self.atoms[:,index[j]] += vec
        #     # Direct = inv(self.box)@self.atoms
        #     # Direct -= floor(Direct)
        #     # self.atoms = self.box@Direct

        #     series_list.append(system(atoms=self.atoms+0, box=self.box+0, types=self.types+0, elements=self.elements, natoms=self.natoms+0, timestep=i))

        # count = 1
        # for i in series_list:
        #     i.export(f'thermal_tests/{count}.poscar')
        #     count += 1

        # self.atoms = old_atoms
        # self.box = old_box
        # self.types = old_types
        # self.natoms = old_natoms
        # self.elements = old_elements
        descriptor = f'{self.descriptor}_{self.thermal.__name__}'
        self.atoms = self._original_atoms+0
        self.box = self._original_box+0
        self.types = self._original_types+0
        self.natoms = self._original_natoms+0
        return series.series(series_list, descriptor=descriptor)

    def thermal_test(self,
                temp: Union[str, int, float, None],
                strain: Union[str, int, float, None] = 0,
                shear: Union[str, int, float, None] = 0,
                nsims: Union[str, int, None] = 100,
                N: int = None,
                ss_dict: Union[dict, None] = None):
        """
        Simulates thermal perturbations on the atomic system by applying random displacements and box strain.

        This method generates a series of configurations by thermally perturbing the
            atomic positions and box dimensions. The perturbations are applied to a 
            base system, and the resulting atomic configurations are stored in a series for further analysis or export.

        Args:
            temp: Maximum random perturbation of atoms.
                Currently, this should be a percentage of the given simulation cell's lattice parameter (in decimal
                format).
            strain: Maximum strain given as a percentage (in decimal format) of the given simulation
                cell's lattice parameter.
            nsims: The number of unique structures to produce.
            N: Number of atoms - Ignore this. Not sure why I made this an argument at the moment.
            ss_dict: dictionary of keywords for doing thermal perturbation on solid solution.
                ss_keys = ['solute_type', 'num_neighbors', 'num_atoms']

        Returns:
            series: A series containing the generated configurations, each with perturbed
                atomic positions and box dimensions.

        Raises:
            ValueError: If the number of simulations (`nsims`) is not provided or invalid
                keywords are found in `ss_dict`.
        """
        # old_atoms = deepcopy(self.atoms)
        # old_box = deepcopy(self.box)
        # old_types = deepcopy(self.types)
        # old_types = copy.copy(self.types)
        # old_natoms = deepcopy(self.natoms)
        # old_elements = deepcopy(self.elements)
        if nsims is None:
            raise ValueError('Please specify the number of structures to create by thermally '
            'perturbing the base system.')
        else:
            nsims = int(nsims)
        if N is None:
            N = self.natoms
        if ss_dict is not None:
            ss_keys = ['solute_type', 'num_neighbors', 'num_atoms']
            good_keys = []
            for i in ss_dict.keys():
                if i not in ss_keys:
                    raise ValueError(f'{i} is not a valid solidsolution keyword.\nValid keywords are {ss_keys}')
                else:
                    if i == 'solute_type':
                        solute_type = ss_dict[i]
                    elif i == 'num_neighbors':
                        num_neighbors = ss_dict[i]
                    elif i == 'num_atoms':
                        num_atoms = ss_dict[i]
                    # elif i == 'replicate':
                    #     replicate = ss_dict[i]
                    else:
                        solute_type = None
                        num_neighbors = 1
                        num_atoms = 1
                        # replicate = None
            # self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, replicate=replicate)


        series_list = []
        for i in range(1,nsims+1):

            # self.atoms = old_atoms
            # self.box = old_box
            # self.types = old_types
            # self.natoms = old_natoms
            # self.elements = old_elements
            self.atoms = self._original_atoms+0
            self.box = self._original_box+0
            self.types = self._original_types+0
            self.natoms = self._original_natoms+0
            self.elements = self._original_elements

            # self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, replicate=replicate)
            if ss_dict is not None:
                self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, num_atoms=num_atoms)

            rng = np.random.default_rng()
            ss = rng.uniform(size=6, low=np.array([-1.,-1.,-1.,-1.,-1.,-1.])*strain,
                             high=np.array([1.,1.,1.,1.,1.,1.])*strain)
            shear_strain = rng.uniform(size=6, low=np.array([-1.,-1.,-1.,-1.,-1.,-1.])*shear,
                                       high=np.array([1.,1.,1.,1.,1.,1.])*shear)
            # ss = (rand(6)-0.5)*strain*2
            # box1 = new_box+0
            # self.box = self.box+0
            xmag = norm(self.box[:,0])
            ymag = norm(self.box[:,1])
            zmag = norm(self.box[:,2])
            self.box[0,0] += ss[0]*xmag
            self.box[1,1] += ss[1]*ymag
            self.box[2,2] += ss[2]*zmag
            self.box[0,1] += shear_strain[3]*ymag
            self.box[0,2] += shear_strain[4]*zmag
            self.box[1,2] += shear_strain[5]*zmag
            # self.box[0,1] += ss[3]*ymag
            # self.box[0,2] += ss[4]*zmag
            # self.box[1,2] += ss[5]*zmag

            # self.box[0,0] += ss[0]*xmag
            # self.box[1,1] += ss[1]*ymag
            # self.box[2,2] += ss[2]*zmag
            # self.box[0,1] += ss[3]*ymag
            # self.box[0,2] += ss[4]*zmag
            # self.box[1,2] += ss[5]*zmag
            # self.atoms = self.box@(inv(old_box)@old_atoms)

            self.atoms = self.box@(inv(self._original_box)@self._original_atoms)

            seq = list(range(0,self.natoms))
            index = random.sample(seq,N)
            for j in range(N):
                rng = np.random.default_rng()
                vec = rng.uniform(size=3, low=np.array([-1.,-1.,-1.]), high=np.array([1.,1.,1.]))
                # x = (random.random()-0.5)
                # y = (random.random()-0.5)
                # z = (random.random()-0.5)
                # vec = np.array([x, y, z])*temp/np.sqrt(3)*2
                vec *= temp/np.sqrt(3)*2
                self.atoms[:,index[j]] += vec
            # Direct = inv(self.box)@self.atoms
            # Direct -= floor(Direct)
            # self.atoms = self.box@Direct

            series_list.append(system(atoms=self.atoms+0, box=self.box+0, types=self.types+0,
                                      elements=self.elements, natoms=self.natoms+0, timestep=i))

        # count = 1
        # for i in series_list:
        #     i.export(f'thermal_tests/{count}.poscar')
        #     count += 1

        # self.atoms = old_atoms
        # self.box = old_box
        # self.types = old_types
        # self.natoms = old_natoms
        # self.elements = old_elements
        descriptor = f'{self.descriptor}_{self.thermal.__name__}'
        self.atoms = self._original_atoms+0
        self.box = self._original_box+0
        self.types = self._original_types+0
        self.natoms = self._original_natoms+0
        return series.series(series_list, descriptor=descriptor)
    def thermal(self,
                temp: Union[str, int, float, None],
                strain: Union[str, int, float, None] = 0,
                nsims: Union[str, int, None] = 100,
                N: int = None,
                ss_dict: Union[dict, None] = None):
        """
        Simulates thermal perturbations on the atomic system by applying random displacements and box strain.

        This method generates a series of configurations by thermally perturbing
        the atomic positions and box dimensions. The perturbations are applied
        to a base system, and the resulting atomic configurations are stored in
        a series for further analysis or export.

        Parameters
        ----------
        temp
            Maximum random perturbation of atoms in Angstroms.
        strain
            Maximum strain given as a percentage (in decimal format) of the given simulation
            cell's lattice parameter.
        nsims
            The number of unique structures to produce.
        N
            Number of atoms to perturb. Useful if you only want to perturb a few random atoms. 
        ss_dict
            dictionary of keywords for doing thermal perturbation on solid solution.
            ss_keys = ['solute_type', 'num_neighbors', 'num_atoms']

        Returns
        -------
        series
            A series containing the generated configurations, each with
            perturbed atomic positions and box dimensions.

        Raises
        ------
        ValueError
            If the number of simulations (`nsims`) is not provided or invalid keywords are found in `ss_dict`.
        """
        # old_atoms = deepcopy(self.atoms)
        # old_box = deepcopy(self.box)
        # old_types = deepcopy(self.types)
        # old_types = copy.copy(self.types)
        # old_natoms = deepcopy(self.natoms)
        # old_elements = deepcopy(self.elements)
        if nsims is None:
            raise ValueError('Please specify the number of structures to create by '
            'thermally perturbing the base system.')
        else:
            nsims = int(nsims)
        if N is None:
            N = self.natoms
        if ss_dict is not None:
            ss_keys = ['solute_type', 'num_neighbors', 'num_atoms']
            good_keys = []
            for i in ss_dict.keys():
                if i not in ss_keys:
                    raise ValueError(f'{i} is not a valid solidsolution keyword.\nValid keywords are {ss_keys}')
                else:
                    if i == 'solute_type':
                        solute_type = ss_dict[i]
                    elif i == 'num_neighbors':
                        num_neighbors = ss_dict[i]
                    elif i == 'num_atoms':
                        num_atoms = ss_dict[i]
                    # elif i == 'replicate':
                    #     replicate = ss_dict[i]
                    else:
                        solute_type = None
                        num_neighbors = 1
                        num_atoms = 1
                        # replicate = None
            # self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, replicate=replicate)


        series_list = []
        for i in range(1,nsims+1):

            # self.atoms = old_atoms
            # self.box = old_box
            # self.types = old_types
            # self.natoms = old_natoms
            # self.elements = old_elements
            self.atoms = self._original_atoms+0
            self.box = self._original_box+0
            self.types = self._original_types+0
            self.natoms = self._original_natoms+0
            self.elements = self._original_elements

            # self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, replicate=replicate)
            if ss_dict is not None:
                self.solidsolution(solute_type=solute_type, num_neighbors=num_neighbors, num_atoms=num_atoms)

            rng = np.random.default_rng()
            ss = rng.uniform(size=6, low=np.array([-1.,-1.,-1.,-1.,-1.,-1.])*strain,
                             high=np.array([1.,1.,1.,1.,1.,1.])*strain)
            # ss = (rand(6)-0.5)*strain*2
            # box1 = new_box+0
            # self.box = self.box+0
            xmag = norm(self.box[:,0])
            ymag = norm(self.box[:,1])
            zmag = norm(self.box[:,2])
            self.box[0,0] += ss[0]*xmag
            self.box[1,1] += ss[1]*ymag
            self.box[2,2] += ss[2]*zmag
            self.box[0,1] += ss[3]*ymag
            self.box[0,2] += ss[4]*zmag
            self.box[1,2] += ss[5]*zmag
            # self.box[0,0] += ss[0]*xmag
            # self.box[1,1] += ss[1]*ymag
            # self.box[2,2] += ss[2]*zmag
            # self.box[0,1] += ss[3]*ymag
            # self.box[0,2] += ss[4]*zmag
            # self.box[1,2] += ss[5]*zmag
            # self.atoms = self.box@(inv(old_box)@old_atoms)

            self.atoms = self.box@(inv(self._original_box)@self._original_atoms)

            seq = list(range(0,self.natoms))
            index = random.sample(seq,N)
            for j in range(N):
                rng = np.random.default_rng()
                vec = rng.uniform(size=3, low=np.array([-1.,-1.,-1.]), high=np.array([1.,1.,1.]))
                # x = (random.random()-0.5)
                # y = (random.random()-0.5)
                # z = (random.random()-0.5)
                # vec = np.array([x, y, z])*temp/np.sqrt(3)*2
                # vec *= temp/np.sqrt(3)*2
                vec *= temp
                self.atoms[:,index[j]] += vec
            Direct = inv(self.box)@self.atoms
            Direct = Direct%1.0
            # Direct -= floor(Direct)
            self.atoms = self.box@Direct

            series_list.append(system(atoms=self.atoms+0, box=self.box+0, types=self.types+0,
                                      elements=self.elements, natoms=self.natoms+0, timestep=i))

        # count = 1
        # for i in series_list:
        #     i.export(f'thermal_tests/{count}.poscar')
        #     count += 1

        # self.atoms = old_atoms
        # self.box = old_box
        # self.types = old_types
        # self.natoms = old_natoms
        # self.elements = old_elements
        descriptor = f'{self.descriptor}_{self.thermal.__name__}'
        self.atoms = self._original_atoms+0
        self.box = self._original_box+0
        self.types = self._original_types+0
        self.natoms = self._original_natoms+0
        return series.series(series_list, descriptor=descriptor)

    # def solidsolution(self,
    #                   solute_type: int = None,
    #                   solute_element: Union[str, int, None] = None,
    #                   num_neighbors: int = 1,
    #                   replicate: Union[int, npt.ArrayLike, None] = None):

    def solidsolution(self,
                      solute_type: int = None,
                      num_atoms: int=1,
                      solute_element: Union[str, int, None] = None,
                      num_neighbors: int = 1):
        """
        Adds a solid solution to the atomic system by introducing solute atoms into the lattice.

        This method randomly selects a position in the system and changes the type of
        num_neighbor closest atoms corresponding to the solute. The solute atoms
        are added to the system's atomic types and elements.

        Parameters
        ----------
        solute_type
            Integer value for the atom type of the solute.
        num_atoms
            Integer value for the number of solute atoms.
        solute_element
            Element type of the solute.
        num_neighbors
            Integer value representing the number of neighbors to a random point
            to change to the solute type.

        Returns
        -------
        None
            The method modifies the `types` attribute of the atomic system by
            assigning the selected solute type to the chosen nearest neighbor atoms.

        Raises
        ------
        ValueError
            If the solute type or element is not valid.
        """
        if solute_type is None:
            solute_type = max(self.types)+1

        if solute_element is None:
            solute_element = str(solute_type)
            self.elements = np.append(self.elements, f'Type_{solute_type}')
        else:
            if isinstance(solute_element, int):
                solute_element = str(solute_element)
            self.elements = np.append(self.elements, solute_element)

        # save = []
        # for i in range(self.natoms):
        #     per = i/self.natoms
        #     if per < 2

        j=0
        while j < num_atoms:
            rng = np.random.default_rng()
            # self.createsupercell(new_box=self.box*replicate)
            point = rng.uniform(size=3, low=[0,0,0], high=[self.box[0,0], self.box[1,1], self.box[2,2]])
            point = np.asarray(point)
            neigh = np.zeros(num_neighbors, dtype=int)
            Vec = self.atoms-point.reshape(3,1)
            VecD = inv(self.box)@Vec
            Vecwrap = self.box@(VecD-np.round(VecD))
            dist = norm(Vecwrap,axis=0)
            I = np.argsort(dist)
            neigh = I[1:num_neighbors+1]
            distance = dist[I]
            ind = neigh
            for i in ind:
                j += 1
                self.types[i] = solute_type

    @staticmethod
    def findspacing(Box1r,atoms1):
        """
        Finds the spacing between atoms in a given plane within the box.

        This method calculates the spacing between atoms that are confined to a specified plane, defined by the
        system's box. It identifies the plane with the largest atom spacing and returns its position.

        Args:
            Box1r (ndarray): The 3x3 matrix representing the simulation box.
            atoms1 (ndarray): The matrix of atom positions (3xN, where N is the number of atoms).

        Returns:
            float: The position of the plane with the largest atom spacing in the simulation box.
        """
        zlen = norm(Box1r[:,2])
        plane = zlen/2
        atomsd = inv(Box1r)@atoms1
        mask = np.all(np.array([atomsd[2,:]>0.4,atomsd[2,:]<0.6]),axis=0)
        Ds = np.diff(np.sort(atoms1[2,mask]))
        I = Ds.argmax()
        Is = np.argsort(atoms1[2,mask])
        atomsred = atoms1[:,mask]
        X1 = atomsred[:,Is[I]]
        X2 = atomsred[:,Is[I+1]]
        m1 = atoms1.T@(X1.reshape(3,1)-atoms1)
        m2 = atoms1.T@(X2.reshape(3,1)-atoms1)
        II1 = np.argmin(np.abs(m1))
        II2 = np.argmin(np.abs(m2))
        planevec = (atoms1[:,II1]+atoms1[:,II2])/2
        plane = planevec.T@Box1r[2,:]/norm(Box1r[2,:])
        return plane

    # def surface(atoms,types,box,x1dir,y1dir,z1dir):
    def surface(self,
                direction: npt.ArrayLike):
        """
        Creates a surface by modifying the atomic system and box dimensions based on a specified direction.

        This method uses the input direction to define the orientation of the system,
        transforms the system to a new basis, and generates a supercell by filling
        the box with atoms. It then applies a transformation to the system's atomic
        positions and box to create a surface-like structure. The result is a
        modified system with a new set of atoms and box dimensions.

        Parameters
        ----------
        direction
            An array of column vectors representing the new simulation cell.
            Free surface will be on the x? plane.

        Returns
        -------
        None
            The method modifies the `atoms` and `box` attributes of the system in place.

        Raises
        ------
        Exception
            If the input direction results in an invalid configuration
            where the cross product leads to a zero vector.
        """
        direction = np.asarray(direction)
        x1dir = direction[:,0]
        y1dir = direction[:,1]
        z1dir = direction[:,2]
        i1 = cross(x1dir,y1dir)
        s1 = dot(i1,z1dir)
        if s1==0:
            raise Exception('invalid z1dir')
        elif s1<0:
            z1dir = -z1dir
        x2 = array([1,0,0])
        y2 = array([0,1,0])
        z2 = array([0,0,1])
        x1dir=self.box@x1dir
        y1dir=self.box@y1dir
        z1dir=self.box@z1dir
        x1h = x1dir/norm(x1dir)
        y1h = y1dir-dot(x1h,y1dir)*x1h
        y1h = y1h/norm(y1h)
        z1h = z1dir-dot(x1h,z1dir)*x1h-dot(y1h,z1dir)*y1h
        z1h = z1h/norm(z1h)
        Trans1 = array([x2,y2,z2]).T@inv(array([x1h,y1h,z1h]).T)
        Box1 = array([x1dir,y1dir,z1dir]).T
        Box1r = Trans1@Box1
        unitcell1r = Trans1@self.box
        basis1r = Trans1@self.atoms
        basis1r = basis1r+np.pi/10000
        # ntypes = max(types)+1
        ntypes = len(np.unique(self.types))
        # (atoms1,Type) = createsupercell(unitcell1r,basis1r,types,Box1r)
        # self.box = unitcell1r
        # self.atoms = basis1r
        # new_box = Box1r
        def createsupercell2(unitcell,atoms,types,box):
            """
            Inputs: Unitcell 3x3 matrix. atoms, matrix with column vectors of atom
            coordinates inside unitcell. Box 3x3 matrix to fill with atoms.
            """
            units = ceil(abs(inv(unitcell)@box))*sign(inv(unitcell)@box)
            vol = int(det(inv(unitcell)@box)*1.1)
            A = zeros((3,(vol+10)*atoms.shape[1]))
            Types = zeros(((vol+10)*atoms.shape[1]))
            n=0
            C000 = array([0,0,0]).T
            C100 = units[:,0]
            C010 = units[:,1]
            C001 = units[:,2]
            C110 = C100+C010
            C101 = C100+C001
            C011 = C010+C001
            C111 = C001+C110
            "min to max values of the box corners give us the bounds for the make lattice command"
            Corners = array([C000,C100,C010,C001,C110,C101,C011,C111]).T
            bounds = round(array([Corners.min(axis=1),Corners.max(axis=1)]))
            bounds[0,:] -= 2
            bounds[1,:] += 2
            for i in range(int(bounds[0,0]),int(bounds[1,0])):
                for j in range(int(bounds[0,1]),int(bounds[1,1])):
                    for k in range(int(bounds[0,2]),int(bounds[1,2])):
                        for l in range(atoms.shape[1]):
                            x = unitcell[:,0]*i+unitcell[:,1]*j+unitcell[:,2]*k+atoms[:,l]
                            if all(inv(box)@x<1) and all(inv(box)@x>=0):
                                A[:,n]=x
                                Types[n]=types[l]
                                n=n+1
            A = A[:,0:n]
            Types = Types[0:n]
            return (A,Types)
        (atoms1,Type) = createsupercell2(unitcell1r,basis1r,self.types,Box1r)
        # atoms1, Type = createsupercell2(self.box, self.atoms, self.types, Box1r)
        self.atoms = atoms1
        # self.types = Type
        # self.natoms = self.atoms.shape[1]

        # self.createsupercell(new_box)
        atoms1 = atoms1-pi/10000
        # atoms1 = self.atoms-np.pi/10000
        plane = self.findspacing(Box1r,atoms1)
        # plane = findspacing(Box1r,atoms1)
        mask = atoms1[2,:]<plane
        atoms2 = self.atoms[:,mask]
        self.atoms = atoms1
        self.box = Box1r
        # return (atoms2,Type,Box1r)%     

    # def vacancy(atoms, types, box, num_neighbors=1, replicate=[1,1,1]):
    def vacancy(self,
                num_neighbors: int):
        """
        Removes atoms from the system to create vacancies by removing atoms close to randomly selected points.

        This method randomly selects points within the simulation box and identifies
        the nearest neighbors to those points. It then removes a specified number
        of nearest neighbors from the atomic system, effectively creating vacancies
        in the structure.

        Parameters
        ----------
        num_neighbors
            The number of neighbors to delete from a randomly selected points.

        Returns
        -------
        None 
            The method modifies the `atoms` and `types` attributes of the system
            in place, removing the specified number of atoms.
        """
        rng = np.random.default_rng()
        # final_box = np.array([replicate])*box
        # new_atoms, new_types = createsupercell(box, atoms, types, final_box)
        point = rng.uniform(size=3, low=[0, 0, 0], high=[self.box[0,0], self.box[1,1], self.box[2,2]])
        # point = np.asarray(point)
        point = np.array([point])

        neigh = np.zeros(num_neighbors, dtype=int)
        Vec = self.atoms-point.reshape(3,1)
        VecD = inv(self.box)@Vec
        Vecwrap = self.box@(VecD-np.round(VecD))
        dist = norm(Vecwrap,axis=0)
        I = np.argsort(dist)
        neigh = I[1:num_neighbors+1]
        distance = dist[I]
        ind = neigh
        # dist, ind = KNNsearch(atoms,box,num_neighbors,point)

        # final_atoms = np.zeros((new_atoms.shape[0], new_atoms.shape[1]-num_neighbors))
        # final_types = np.zeros(final_atoms.shape[1], dtype=int)
        final_atoms = np.zeros((self.atoms.shape[0], self.atoms.shape[1]-num_neighbors))
        final_types = np.zeros(self.atoms.shape[1]-num_neighbors, dtype=int)
        count = 0
        for i in range(self.atoms.shape[1]):
            if i not in ind:
                final_atoms[:,count] = self.atoms[:,i]
                final_types[count] = self.types[i]
                count += 1
            else:
                continue
        self.atoms = final_atoms
        self._original_atoms = self.atoms
        self.types = final_types
        self._original_types = self.types
        self.natoms = self.atoms.shape[1]
        self._original_natoms = self.natoms
        # return (final_atoms, final_types, final_box)

    @staticmethod
    def makelattice3(unitcell,atoms,types,box):
        """
        Generates a lattice structure by replicating a unit cell and placing atoms within the given box.

        This method constructs a lattice by replicating the atoms within a unit
            cell to fill a 3D box. It computes the lattice vectors, iterates over
            the box to place atoms at appropriate positions, and returns the final
            atomic positions and their associated types.

        Args:
            unitcell (ndarray): A 3x3 matrix representing the unit cell of the structure.
            atoms (ndarray): A matrix of atomic positions (3xN), where each column represents
                the position of an atom in the unit cell.
            types (ndarray): A vector of atomic types corresponding to the atoms in the unit cell.
            box (ndarray): A 3x3 matrix representing the dimensions of the box in which the lattice is constructed.

        Returns:
            tuple: A tuple containing:
                - `A` (ndarray): A matrix of the atomic positions in the lattice
                    (3xM, where M is the total number of atoms).
                - `Types` (ndarray): A vector of atomic types corresponding to the atoms in the lattice.

        Notes:
            The method uses the inverse of the unit cell to determine the appropriate
                scaling and position of atoms within the box.
        """
        # """
        # Inputs: Unitcell 3x3 matrix. atoms, matrix with column vectors of atom
        # coordinates inside unitcell. Box 3x3 matrix to fill with atoms.
        # """
        units = ceil(abs(inv(unitcell)@box))*sign(inv(unitcell)@box)
        vol = int(det(inv(unitcell)@box)*1.1)
        A = zeros((3,(vol+10)*atoms.shape[1]))
        Types = zeros(((vol+10)*atoms.shape[1]))
        n=0
        C000 = array([0,0,0]).T
        C100 = units[:,0]
        C010 = units[:,1]
        C001 = units[:,2]
        C110 = C100+C010
        C101 = C100+C001
        C011 = C010+C001
        C111 = C001+C110
        "min to max values of the box corners give us the bounds for the make lattice command"
        Corners = array([C000,C100,C010,C001,C110,C101,C011,C111]).T
        bounds = round(array([Corners.min(axis=1),Corners.max(axis=1)]))
        bounds[0,:] -= 2
        bounds[1,:] += 2
        for i in range(int(bounds[0,0]),int(bounds[1,0])):
            for j in range(int(bounds[0,1]),int(bounds[1,1])):
                for k in range(int(bounds[0,2]),int(bounds[1,2])):
                    for l in range(atoms.shape[1]):
                        x = unitcell[:,0]*i+unitcell[:,1]*j+unitcell[:,2]*k+atoms[:,l]
                        if all(inv(box)@x<1) and all(inv(box)@x>=0):
                            A[:,n]=x
                            Types[n]=types[l]
                            n=n+1
        A = A[:,0:n]
        Types = Types[0:n]
        return (A,Types)

    # def generate_gsfe(inputfile,inputqescript,outputfile,outputdirectory,output_style,N,x1dir,y1dir,z1dir,reps):
    def gsfe(self,
            nsims: Union[int, str] = 100,
            direction: npt.ArrayLike = np.array([[1,0,0],[0,1,0],[0,0,1]]),
            reps: npt.ArrayLike = np.array([1,1,12])):
        """
        Generates a series of simulations for computing the Generalized Stacking Fault Energy.

        This method creates a series of simulations by generating different
        configurations of a system's atomic positions, shifting the
        positions and box based on specified replication parameters.
        It uses the `makelattice3` method to construct the lattice and
        applies transformations to simulate different strain states or
        surface configurations. The direction should be a 3x3 matrix with
        x, y, and z as column vectors.

        Parameters
        ----------
        nsims
            The total number of unique structures to generate moving along the fault.
        direction
            Array of column vectors representing the miller indicies of the given
            simulation cell. The stacking fault plane should be the x direction.
        reps
            An array of size (1,3) representing the number of replications of the final
            simulation cell in the x, y, and z directions.

        Returns
        -------
        series
            A series object containing the configurations generated in
            each simulation, with atomic positions, box, and types.

        Raises
        ------
        Exception
            If the direction vectors lead to an invalid configuration
            (i.e., a left-handed coordinate system).

        Notes
        -----
            The method iteratively modifies the atomic positions and the
            simulation box to generate different configurations. It also
            creates a series of simulations that can be used to analyze
            the GSFE by comparing different atomic configurations under
            strain. The final series of configurations is returned as a
            `series` object.
        """
        direction = np.asarray(direction)
        reps = np.asarray(reps)
        Xr = reps[0]
        Yr = reps[1]
        Zr = reps[2]
        x1dir = direction[:,0]
        y1dir = direction[:,1]
        z1dir = direction[:,2]
        # [Xr,Yr,Zr] = reps
        # (basis,types,unitcell,elements,style)=importposcar(inputfile)
        i1 = cross(x1dir,y1dir)
        s1 = dot(i1,z1dir)
        if s1==0:
            raise Exception('invalid z1dir')
        elif s1<0:
            z1dir = -z1dir
        x2 = array([1,0,0])
        y2 = array([0,1,0])
        z2 = array([0,0,1])
        x1dir=self.box@x1dir
        y1dir=self.box@y1dir
        z1dir=self.box@z1dir
        x1h = x1dir/norm(x1dir)
        y1h = y1dir-dot(x1h,y1dir)*x1h
        y1h = y1h/norm(y1h)
        z1h = z1dir-dot(x1h,z1dir)*x1h-dot(y1h,z1dir)*y1h
        z1h = z1h/norm(z1h)
        Trans1 = array([x2,y2,z2]).T@inv(array([x1h,y1h,z1h]).T)
        Box1 = array([x1dir*Xr,y1dir*Yr,z1dir*Zr]).T
        # Box1 = array([x1dir,y1dir,z1dir]).T
        Box1r = Trans1@Box1
        unitcell1r = Trans1@self.box
        basis1r = Trans1@self.atoms
        basis1r = basis1r+pi/10000
        # ntypes = max(types)+1
        ntypes = len(np.unique(self.types))
        (atoms1,Type) = self.makelattice3(unitcell1r,basis1r,self.types,Box1r)
        atoms1 = atoms1-pi/10000
        plane = self.findspacing(Box1r,atoms1)
        shift = Trans1@x1dir/nsims

        self.atoms = atoms1
        # self._original_atoms = self.atoms+0
        self.box = Box1r
        # self._original_box = self.box+0
        self.types = Type
        # self._original_types = self.types+0
        # self._original_elements = deepcopy(self.elements)
        self.natoms = self.atoms.shape[1]
        # self._original_natoms = self.natoms+0

        # file = open(inputqescript,'r')
        # qelines = file.readlines()
        # file.close()
        # origin = np.array([0,0,0])
        # folders = outputdirectory.split('/')
        # for n in range(len(folders)-1):
        #     if not os.path.isdir(folders[n]):
        #         os.mkdir(folders[n])
        #     os.chdir(folders[n])

        # i=0
        # initialdirectory = os.getcwd()
        # if not os.path.isdir(folders[-1]+str(i)):
        #     os.mkdir(folders[-1]+str(i))
        # os.chdir(folders[-1]+str(i))
        # writeposcar(outputfile+'.'+str(i)+'.poscar',atoms1,Type,Box1r,elements,output_style)
        # (lbox,tilt)=lammpsbox(Box1r,origin)
        # writedata(outputfile+'.'+str(i)+'.data',atoms1,Type+1,ntypes,lbox,tilt)
        # writeqecoords(outputfile+'.'+str(i)+'.x',qelines,Box1r,atoms1,Type,elements)
        # os.chdir('..')
        # writedata(outputfile+'.'+str(i)+'.data',atoms1,Type+1,ntypes,lbox,tilt)
        series_list = []
        for i in range(1,nsims+1):
            # self.atoms = self._original_atoms+0
            # self.box = self._original_box+0
            # self.types = self._original_types+0
            # self.natoms = self._original_natoms+0
            # self.elements = self._original_elements
            # if not os.path.isdir(folders[-1]+str(i)):
            #     os.mkdir(folders[-1]+str(i))
            # os.chdir(folders[-1]+str(i))

            series_list.append(system(atoms=self.atoms+0, box=self.box+0, types=self.types,
                                      elements=self.elements, natoms=self.natoms+0, timestep=i))
            # Box1r[:,2]+=shift
            # self.box[:,2] += shift
            self.box[:,2] = self.box[:,2]+shift
            # mask = atoms1[2,:]>plane
            mask = self.atoms[2,:]>plane
            # atoms1[:,mask]+=shift.reshape(3,1)
            self.atoms[:,mask] += shift.reshape(3,1)
            # writeposcar(outputfile+'.'+str(i)+'.poscar',atoms1,Type,Box1r,elements,output_style)
            # (lbox,tilt)=lammpsbox(Box1r,origin)
            # writedata(outputfile+'.'+str(i)+'.data',atoms1,Type+1,ntypes,lbox,tilt)
            # writeqecoords(outputfile+'.'+str(i)+'.x',qelines,Box1r,atoms1,Type,elements)
            # os.chdir('..')
            # writedata(outputfile+'.'+str(i)+'.data',atoms1,Type+1,ntypes,lbox,tilt)
        # os.chdir(initialdirectory)% 

        descriptor = f'{self.descriptor}_{self.gsfe.__name__}'
        self.atoms = self._original_atoms
        self.box = self._original_box
        self.types = self._original_types
        self.elements = self._original_elements
        self.natoms = self._original_natoms
        return series.series(series_list, descriptor=descriptor)

    def strain(self,
               strain_points: int = 100,
               maximum_strain: np.float64 = 0.05):

        '''
        Strains the simulation cell in even increments from `-maximum_strain` to `maximum_strain`.

        Strains the simulation cell in even increments up to "maximum_strain"
        in the following ways: volumetric strain, strain along x, strain
        along y, strain along z, yz shear strain, xz shear strain, xy shear
        strain, strain along (111), xy in-plane strain, xy in-plane shear
        strain, and global strain. Strain points indicates how many different
        simulation cells to create for each type of strain. The "maximum_strain"
        parameter is the decimal form of a percentage (0.1 for 10% maximum strain).

        Parameters
        ----------
        strain_points
            The number of unique structures to create for each strain mode (14 strain modes
            currently used?)
        maximum_strain
            Maximum percentage (in decimal format) of the lattice parameter of the given 
            simulation cell to strain the box.

        Returns
        -------
        A series class object
        '''

        # if (str(os.path.exists('input.xml'))=='False'):
            # sys.exit("ERROR: Input file input.xml not found!\n")

        #maximum_strain = input("\nEnter maximum Lagrangian strain [smax] >>>> ")
        # maximum_strain=0.15
        # if (maximum_strain == 0):
        #     work_directory = 'workdir'
        #     if (os.path.exists(work_directory)): shutil.rmtree(work_directory)
        #     os.mkdir(work_directory)
        #     os.system("cp input.xml workdir/input-01.xml")
        #     output_info = open('workdir/INFO-elastic-constants',"w")
        #    print >>output_info,\
        #        "\nMaximum Lagrangian strain       = 0",\
        #        "\nNumber of strain values         = 1",\
        #        "\nVolume of equilibrium unit cell = 1.0 [a.u]^3",\
        #        "\nDeformation code                = 000000",\
        #        "\nDeformation label               = single\n"
            # output_info.close()
            # output_eta = open('workdir/strain-01',"w")
        #    print >>output_eta, "0.00"
            # output_eta.close()
        #    print
            # sys.exit("Single unstrained calculation\n")

        if (1 < maximum_strain or maximum_strain < 0):
            sys.exit("ERROR: Maximum Lagrangian strain is out of range [0-1]!\n")
        #strain_points = \
        #              input("\nEnter the number of strain values in [-smax,smax] >>>> ")
        # strain_points=99
        # strain_points = int(abs(strain_points))
        if (3 > strain_points or strain_points > 101):
            sys.exit("ERROR: Number of strain values is out of range [3-99]!\n")

        # print
        # print "------------------------------------------------------------------------"
        # print " List of deformation codes for strains in Voigt notation"
        # print "------------------------------------------------------------------------"
        # print "  0 =>  ( eta,  eta,  eta,    0,    0,    0)  | volume strain "
        # print "  1 =>  ( eta,    0,    0,    0,    0,    0)  | linear strain along x "
        # print "  2 =>  (   0,  eta,    0,    0,    0,    0)  | linear strain along y "
        # print "  3 =>  (   0,    0,  eta,    0,    0,    0)  | linear strain along z "
        # print "  4 =>  (   0,    0,    0,  eta,    0,    0)  | yz shear strain"
        # print "  5 =>  (   0,    0,    0,    0,  eta,    0)  | xz shear strain"
        # print "  6 =>  (   0,    0,    0,    0,    0,  eta)  | xy shear strain"
        # print "  7 =>  (   0,    0,    0,  eta,  eta,  eta)  | shear strain along (111)"
        # print "  8 =>  ( eta,  eta,    0,    0,    0,    0)  | xy in-plane strain "
        # print "  9 =>  ( eta, -eta,    0,    0,    0,    0)  | xy in-plane shear strain"
        # print " 10 =>  ( eta,  eta,  eta,  eta,  eta,  eta)  | global strain"
        # print " 11 =>  ( eta,    0,    0,  eta,    0,    0)  | mixed strain"
        # print " 12 =>  ( eta,    0,    0,    0,  eta,    0)  | mixed strain"
        # print " 13 =>  ( eta,    0,    0,    0,    0,  eta)  | mixed strain"
        # print " 14 =>  ( eta,  eta,    0,  eta,    0,    0)  | mixed strain"
        # print "------------------------------------------------------------------------"
        codes = np.array([0,1,2,3,4,5,6,7,8,9,10])
        systems_list = []
        timestep_count = 1
        for mwn in codes:
            self.box = self._original_box+0
            self.atoms = self._original_atoms+0
            deformation_code = mwn
            # deformation_code=XXX
            #deformation_code = input("\nEnter deformation code >>>> ")
            if (0 > deformation_code or deformation_code > 14):
                sys.exit("ERROR: Deformation code is out of range [0-14]!\n")

            if (deformation_code == 0 ): dc='EEE000'
            if (deformation_code == 1 ): dc='E00000'
            if (deformation_code == 2 ): dc='0E0000'
            if (deformation_code == 3 ): dc='00E000'
            if (deformation_code == 4 ): dc='000E00'
            if (deformation_code == 5 ): dc='0000E0'
            if (deformation_code == 6 ): dc='00000E'
            if (deformation_code == 7 ): dc='000EEE'
            if (deformation_code == 8 ): dc='EE0000'
            if (deformation_code == 9 ): dc='Ee0000'
            if (deformation_code == 10): dc='EEEEEE'
            if (deformation_code == 11): dc='E00E00'
            if (deformation_code == 12): dc='E000E0'
            if (deformation_code == 13): dc='E0000E'
            if (deformation_code == 14): dc='EE0E00'

            #-------------------------------------------------------------------------------

            # input_obj = open("input.xml","r")
            # input_doc = etree.parse(input_obj)
            # input_rut = input_doc.getroot()

            # xml_scale = list(map(float,input_doc.xpath('/input/structure/crystal/@scale')))
            # if (xml_scale == []):
            #     ref_scale = 1.0
            # else:
            #     ref_scale = float(xml_scale[0])
            ref_scale = 1.0

            # str_stretch = input_doc.xpath('/input/structure/crystal/@stretch')
            # if (str_stretch ==[]):
            #     xml_stretch = [1.,1.,1.]
            # else: xml_stretch=numpy.array(list(map(float,str_stretch[0].split())))
            xml_stretch = [1.,1.,1.]

            # lst_basevect = input_doc.xpath('//basevect/text()')
            # xml_basevect = []
            # for ind_basevect in lst_basevect:
            #     xml_basevect.append(list(map(float,ind_basevect.split())))

            # axis_matrix = numpy.array(xml_basevect)
            axis_matrix = self.box.T+0
            determinant = np.linalg.det(axis_matrix)
            volume = abs(determinant*ref_scale**3\
                *xml_stretch[0]*xml_stretch[1]*xml_stretch[2])

            # work_directory = 'workdir'
            # if (len(sys.argv) > 1): work_directory = sys.argv[1]
            # if (os.path.exists(work_directory)): shutil.rmtree(work_directory)
            # os.mkdir(work_directory)
            # os.chdir(work_directory)

            # output_info = open('INFO-elastic-constants',"w")
            #print >>output_info, "\nMaximum Lagrangian strain       = ", maximum_strain,\
            #                     "\nNumber of strain values         = ", strain_points,\
            #                     "\nVolume of equilibrium unit cell = ", volume, "[a.u]^3",\
            #                     "\nDeformation code                = ", deformation_code,\
            #                     "\nDeformation label               = ", dc, "\n"
            # output_info.close()

            #-------------------------------------------------------------------------------

            delta=strain_points-1
            convert=1
            if (strain_points <= 1):
                strain_points=1
                convert=-1
                delta=1

            eta_step=2*maximum_strain/delta

            #-------------------------------------------------------------------------------

            for i in range(0,strain_points):

            #-------------------------------------------------------------------------------

                eta=i*eta_step-maximum_strain*convert

                # if (i+1 < 10): strainfile = 'strain-0'+str(i+1)
                # else: strainfile = 'strain-'+str(i+1)
                # output_str = open(strainfile,"w")
                # fmt = '%11.8f'
                # print >>output_str, (fmt%eta)
                # output_str.close()

                if (abs(eta) < 0.000001): eta=0.000001
                ep=eta
                if (eta < 0.0): em=abs(eta)
                else: em=-eta

            #-------------------------------------------------------------------------------

                e=[]
                for j in range(6):
                    ev=0
                    if  (dc[j:j+1] == 'E' ): ev=ep
                    elif(dc[j:j+1] == 'e' ): ev=em
                    elif(dc[j:j+1] == '0' ): ev=0
                    else: dc; sys.exit("ERROR: deformation code not allowed!")
                    e.append(ev)

            #-------------------------------------------------------------------------------

                # eta_matrix=mat([[ e[0]   , e[5]/2., e[4]/2.],
                #                 [ e[5]/2., e[1]   , e[3]/2.],
                #                 [ e[4]/2., e[3]/2., e[2]   ]])

                eta_matrix=np.array([[ e[0]   , e[5]/2., e[4]/2.],
                                [ e[5]/2., e[1]   , e[3]/2.],
                                [ e[4]/2., e[3]/2., e[2]   ]])

                # one_matrix=mat([[  1.0,  0.0,  0.0],
                #                 [  0.0,  1.0,  0.0],
                #                 [  0.0,  0.0,  1.0]])

                one_matrix=np.array([[  1.0,  0.0,  0.0],
                                [  0.0,  1.0,  0.0],
                                [  0.0,  0.0,  1.0]])

            #-------------------------------------------------------------------------------

                norma=1.0
                inorma=0
                eps_matrix=eta_matrix

                if (np.linalg.norm(eta_matrix) > 0.7):sys.exit("ERROR: too large deformation!")

                while ( norma > 1.e-10 ):
                    x=eta_matrix-0.5*dot(eps_matrix,eps_matrix)
                    norma=np.linalg.norm(x-eps_matrix)
                    eps_matrix=x
                    inorma=inorma+1

                def_matrix=one_matrix+eps_matrix
                new_atoms_matrix = self.atoms+0

                new_axis_matrix=np.transpose(dot(def_matrix,np.transpose(axis_matrix)))

                Direct = inv(self.box)@self.atoms
                Direct = Direct%1.0
                # Direct -= floor(Direct)
                new_atoms_matrix = new_axis_matrix.T@Direct
                # new_atoms_matrix=np.transpose(dot(def_matrix,np.transpose(new_atoms_matrix)))
                # new_atoms_matrix = self.atoms.T+def_matrix
                nam=new_axis_matrix
                self.atoms = new_atoms_matrix+0

                self.box = new_axis_matrix.T+0
                systems_list.append(system(atoms=self.atoms+0, box=self.box+0, types=self.types+0,
                                           elements=self.elements, natoms=self.natoms+0, timestep=timestep_count))
                timestep_count += 1

            #-------------------------------------------------------------------------------

                #xbv = input_doc.xpath('//crystal/basevect')
                #fmt = '%22.16f'
                #for j in range(3):
                #    xbv[j].text = str(fmt%nam[j,0])+str(fmt%nam[j,1])+str(fmt%nam[j,2])+" "
                #if (i+1 < 10): outputfile = 'input-'+str(i+1)+'.xml'
                #else: outputfile = 'input-'+str(i+1)+'.xml'
                #output_obj = open(outputfile,"wb")
                #output_obj.write(etree.tostring(input_rut, method='xml',
                #                                           #pretty_print=True,
                #                                           xml_declaration=False,
                #                                           encoding='UTF-8'))
                #output_obj.close()

            #-------------------------------------------------------------------------------

            # os.chdir('../')
            #print

        descriptor = f'{self.descriptor}_{self.strain.__name__}'
        return series.series(systems_list, descriptor='strain')

    def change_type(self,
                    new_types: Union[npt.ArrayLike, int]):
        '''
        Changes the integer type for selected atoms.

        Changes the integer atom type for the system object. "new_types" should
        be array-like where the first position represents the new type for
        current atom type 1. For example, if a system contains Ti as type 
        and Al as type 2, they could be flipped using "new_types=[2,1]".
        "new_types" should have the same length as the desired number of
        atom types.

        Parameters
        ----------
        new_types
            An array-like item that represents new atom types.

        Returns
        -------
        None
            This method updates the system's types attribute.
        '''

        new_types = np.asarray(new_types)
        old_types = self.types+0
        Type = np.array([new_types[i-1] for i in old_types])
        self.types = Type+0
        self._original_types = self.types+0
