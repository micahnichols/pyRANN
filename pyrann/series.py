import os
import sys
import numpy as np
import numpy.typing as npt
import textwrap

from numpy import ceil, sign, array, zeros, round
from numpy.linalg import inv, det, norm
from numpy import sqrt, cross, dot, array

from typing import Union, Optional
from copy import deepcopy
# from .system import system 
from . import system

class series:
    """
    A class to represent a series of atomistic systems, which can be exported to various file formats.

    Parameters
    ----------
    systems
        A list or array of atomistic systems to be contained in the series.
    descriptor
        A descriptor to label the series (e.g., a string identifier).

    """
    def __init__(self,
                 systems: Optional[npt.ArrayLike] = None,
                 descriptor: Optional[str] = None):
        """
        Initializes a `series` object.

        Parameters:
        systems: A list or array of atomistic systems to be contained in the series.
        descriptor: A descriptor to label the series (e.g., a string identifier).
        """
        if systems is not None:
            systems = np.asarray(systems)
            timesteps = [i.timestep for i in systems]
        self.systems = systems
        self.descriptor = descriptor
        self.timesteps = timesteps

    def __str__(self) -> str:
        """
        Returns a string representation of all systems in the series.

        The output includes information such as box dimensions, atom count, element types, atom positions, and more.

        Returns
        -------
        str
            A formatted string containing details of each system in the series.
        """
        # string = ['\n'.join([f'box = \n{i.box}', f'natoms = {i.natoms}', f'elements = {i.elements}',
        #                      f'timestep = {i.timestep}', f'types = {i.types}',
        #                      f'atoms = \n{i.atoms}']) for i in self.systems]
        string = [str(i) for i in self.systems]
        return '\n\n---------------------------------------------------------------------------------\n\n'.join(string)

    @property
    def descriptor(self) -> str:
        """
        Gets the descriptor for the series.

        Returns
        -------
        str
            The current descriptor for the series.
        """
        return self.__descriptor
    @descriptor.setter
    def descriptor(self, value: str):
        """
        Sets the descriptor for the series.

        Parameters
        ----------
        value
            The new descriptor to set for the series.
        """
        descriptor = value
        self.__descriptor = descriptor

    @property
    def timesteps(self) -> np.ndarray:
        """
        Returns 1-D array of timesteps for all systems.

        Returns
        -------
        np.ndarray
            1-D array of timesteps.
        """
        return self.__timesteps

    @timesteps.setter
    def timesteps(self, value: npt.ArrayLike):
        """
        Sets the timesteps for all systems

        Parameters
        ----------
        value
            1-D array of timesteps for all systems in class
        """
        self.__timesteps = value

    @property
    def systems(self) -> np.ndarray:
        """
        Returns 1-D array of system objects in the series object

        Returns
        -------
        np.ndarray
            1-D array of system objects in the series object
        """
        return self.__systems
    
    @systems.setter
    def systems(self, value: npt.ArrayLike):
        """
        Sets the list of systems for the series object

        Parameters
        ----------
        value
            ArrayLike item consisting of system objects
        """
        if all(isinstance(i, system.system) for i in value):
            systems = value
        else:
            raise ValueError('systems must be an ArrayLike object '
                             'and contain only system class instances')
        self.__systems = systems

    def export(self,
               filename: Union[str, None],
               directory: Union[str, None] = None,
               filetype: Union[str, None] = None,
               style: Union[str, None] = None):
        """
        Exports the systems in the series to a specified file format.

        Parameters
        ----------
        filename
            The name of the file to export (without extension).
        directory
            The directory where the file will be saved. If None, saves in the current directory.
        filetype
            The format of the file (e.g., 'poscar', 'data'). If None, inferred from filename.
        style
            The style for representing atomic positions (e.g., 'cartesian', 'direct').

        Raises
        ------
        ValueError
            If an unsupported filetype is provided, or if the style is invalid.

        Returns
        -------
        None
        """
        nsims = len(self.systems)
        if filename is None:
            filename = 'Atomic_System'
        else:
            if filetype is None:
                ext_keys = ['poscar', 'pos', 'data', 'dat', 'dump', 'qe', 'quantum_espresso', 'vasp', 'INCAR', 'cfg']
                if filename.split('.')[-1].lower() in ext_keys:
                    filetype = filename.split('.')[-1]
                else:
                    raise ValueError(f'Either filetype must be given, or the file extension must '
                    f'be one of the following:\n{ext_keys}')
            else:
                filename = '.'.join([filename, filetype])
        if directory is None:
            directory = ''
        else:
            sys.path.append(os.getcwd())
            if not os.path.isdir(directory):
                os.mkdir(directory)
            directory = f'{directory}/'
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
        elif filetype.lower() == 'cfg':
            filetype = 'cfg'


        if filetype.lower() == 'poscar':
            original_filename = filename
            for mwn in range(nsims):
                filename = f'{directory}{original_filename.split(".")[0]}_{mwn}.{original_filename.split(".")[-1]}'
                file = open(filename,"w")
                # natoms = Lattice.shape[1]
                file.write("#Atomistic data file created by python\n")
                file.write("1.0\n")
                file.write("%f   %f   %f\n" % (self.systems[mwn].box[0,0],
                                               self.systems[mwn].box[1,0],
                                               self.systems[mwn].box[2,0]))
                file.write("%f   %f   %f\n" % (self.systems[mwn].box[0,1],
                                               self.systems[mwn].box[1,1],
                                               self.systems[mwn].box[2,1]))
                file.write("%f   %f   %f\n" % (self.systems[mwn].box[0,2],
                                               self.systems[mwn].box[1,2],
                                               self.systems[mwn].box[2,2]))
                typesum = []
                for i in range(len(self.systems[mwn].elements)):
                    file.write(self.systems[mwn].elements[i]+' ')
                    typesum.append(sum(self.systems[mwn].types==i+1))
                file.write("\n")
                for i in range(len(self.systems[mwn].elements)):
                    file.write("%d " % typesum[i])
                file.write("\n")
                file.write("%s\n" % style)
                if (style=='direct' or style=='Direct' or style=='Crystal' or style == 'crystal'):
                    Lattice = inv(self.systems[mwn].box)@self.systems[mwn].atoms
                else:
                    Lattice = self.systems[mwn].atoms
                I = np.argsort(self.systems[mwn].types)
                for i in range(self.systems[mwn].natoms):
                    file.write("\t {0:f} {1:f} {2:f} {3:s}\n" .format(
                        Lattice[0,I[i]],Lattice[1,I[i]],Lattice[2,I[i]],
                        self.systems[mwn].elements[int(self.systems[mwn].types[I[i]])-1])
                               )
                file.close()

        elif filetype == 'data':
            original_filename = filename
            for mwn in range(nsims):
                filename = f'{directory}{original_filename.split(".")[0]}_{mwn}.{original_filename.split(".")[-1]}'
                (xlo,ylo,zlo)=(0,0,0)
                xhi = norm(self.systems[mwn].box[:,0])
                xy = dot(self.systems[mwn].box[:,1],self.systems[mwn].box[:,0]/norm(self.systems[mwn].box[:,0]))
                yhi = norm(cross(self.systems[mwn].box[:,0]/norm(self.systems[mwn].box[:,0]),
                                 self.systems[mwn].box[:,1]))
                xz = dot(self.systems[mwn].box[:,2],self.systems[mwn].box[:,0]/norm(self.systems[mwn].box[:,0]))
                yz = (dot(self.systems[mwn].box[:,2],self.systems[mwn].box[:,1])-xy*xz)/yhi
                zhi = sqrt(dot(self.systems[mwn].box[:,2],self.systems[mwn].box[:,2])-xz**2-yz**2)
                new_box = array([[xlo, xhi],[ylo, yhi],[zlo, zhi]])
                tilt = array([xy, xz, yz]);
                R = (array([[xhi, xy, xz],[0, yhi, yz],[0, 0, zhi]])
                     @array([cross(self.systems[mwn].box[:,1],
                                   self.systems[mwn].box[:,2]),
                             cross(self.systems[mwn].box[:,2],
                                   self.systems[mwn].box[:,0]),
                             cross(self.systems[mwn].box[:,0],
                                   self.systems[mwn].box[:,1])])
                     /det(array([[xhi, xy, xz],[0, yhi, yz],[0, 0, zhi]])))
                x = R@self.systems[mwn].atoms
                lammpsbox = new_box
                # (atoms,lammpsbox,tilt,R)=triclinic2lammps(Atoms,Box)
                file = open(filename,"w")
                # natoms = atoms.shape[1]
                ntypes = len(np.unique(self.systems[mwn].types))
                file.write("#Atomistic data file created by python\n\n")
                file.write("%d\tatoms\n\n" % self.systems[mwn].natoms)
                file.write("%d\t atom types\n\n" % ntypes)
                file.write("%f   %f xlo xhi\n" % (lammpsbox[0,0],lammpsbox[0,1]))
                file.write("%f   %f ylo yhi\n" % (lammpsbox[1,0],lammpsbox[1,1]))
                file.write("%f   %f zlo zhi\n" % (lammpsbox[2,0],lammpsbox[2,1]))
                file.write("%f   %f  %f xy xz yz\n" % (tilt[0],tilt[1],tilt[2]))
                file.write("Atoms\n\n")
                for i in range(self.systems[mwn].natoms):
                    file.write("\t {0:d} {1:.0f} {2:f} {3:f} {4:f}\n" .format(i+1,
                                                                              self.systems[mwn].types[i],
                                                                              x[0,i],x[1,i],x[2,i]))
                file.close()

        elif filetype == 'dump':
            original_filename = filename
            filename = f'{directory}{original_filename}'

           # if filename is None:
           #     filename = 'Atomic_series.dump'
           # else:
           #     if filename.lower().split('.')[-1] != 'dump':
           #         filename = '.'.join([filename.split('.')[0], 'dump'])

            file = open(filename,"w")
           # nsims = sims.size
            nsims = len(self.systems)
            sims = len(self.systems)
            energy = np.zeros(nsims)
            energy_weight = np.ones(nsims)
            # force = np.zeros((nsims,3))
            force_weight = np.ones(nsims)
            # stress1 = np.zeros((nsims,3))
            for nn in range(nsims):
               file.write("ITEM: TIMESTEP energy, energy_weight, force_weight, nsims\n")
               file.write("%d %f %f %f %d\n" % (self.systems[nn].timestep,
                                                self.systems[nn].energy,
                                                energy_weight[nn],force_weight[nn],nsims))
               file.write("ITEM: NUMBER OF ATOMS\n")
               natoms = self.systems[nn].natoms
               file.write("%d\n" % natoms)
               origin = np.zeros((3))
               Box = self.systems[nn].box
               # X = sims[nn]['x']
               X = self.systems[nn].atoms

               (xlo,ylo,zlo)=(0,0,0)
               xhi = norm(Box[:,0])
               xy = dot(Box[:,1],Box[:,0]/norm(Box[:,0]))
               yhi = norm(cross(Box[:,0]/norm(Box[:,0]),Box[:,1]))
               xz = dot(Box[:,2],Box[:,0]/norm(Box[:,0]))
               yz = (dot(Box[:,2],Box[:,1])-xy*xz)/yhi
               zhi = sqrt(dot(Box[:,2],Box[:,2])-xz**2-yz**2)
               box = array([[xlo, xhi],[ylo, yhi],[zlo, zhi]])
               tilt = array([xy, xz, yz]);
               R = (array([[xhi, xy, xz],[0, yhi, yz],[0, 0, zhi]])
                    @array([cross(Box[:,1],Box[:,2]),
                            cross(Box[:,2],Box[:,0]),
                            cross(Box[:,0],Box[:,1])])
                    /det(array([[xhi, xy, xz],[0, yhi, yz],[0, 0, zhi]])))
               x = R@X
               lammpsbox = box
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
               # return (x,lammpsbox,tilt,R)%

               # (x,box,tilt,R)=triclinic2lammps(X,Box)
               # if self.systems[nn].stress is None:
               #     do_stress = False 
               # else:
               #     do_stress = True
                   stress = self.systems[nn].stress
               if self.systems[nn]._do_stress:
                   file.write("ITEM: BOX BOUNDS xy xz yz pp pp pp stress\n")
               else:
                   file.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
               # stress = sims[nn]['stress']
               # stress = stress1[nn]
               for i in range(3):
                   # file.write("%f %f %f %f %f %f\n" % (box[i,0],box[i,1],tilt[i],stress[i,0],stress[i,1],stress[i,2]))
                   # file.write("%f %f %f %f %f %f\n" % (box[i,0],box[i,1],tilt[i],stress[0],stress[1],stress[2]))
                   if self.systems[nn]._do_stress:
                       file.write("%f %f %f %f %f %f\n" % (lammpsbox[i,0],
                                                           lammpsbox[i,1],
                                                           tilt[i],stress[0],
                                                           stress[1],stress[2]))
                   else:
                       file.write("%f %f %f\n" % (lammpsbox[i,0],lammpsbox[i,1],tilt[i]))
               file.write("ITEM: ATOMS id type x y z fx fy fz\n")
               # f = sims[nn]['f']
               # do_force = True
               # f = self.systems[nn].force
               # if f is None:
               #     do_force = False
               # ids = sims[nn]['id']
               # types = sims[nn]['type']
               for i in range(natoms):
                   # if self.systems[nn].force != None:
                   if self.systems[nn]._do_force:
                       file.write("%d %d %f %f %f %f %f %f\n" % (i+1, self.systems[nn].types[i],
                                                                 x[0][i], x[1][i], x[2][i],
                                                                 f[0][i], f[1][i], f[2][i]))
                   else:
                       # TODO - FIX TO WHERE FORCES ARE NOT WRITTEN IF _do_force == False
                       file.write("%d %d %f %f %f %f %f %f\n" % (i+1, self.systems[nn].types[i],
                                                                 x[0][i], x[1][i], x[2][i],
                                                                 0.0, 0.0, 0.0))
            file.close()
        elif filetype == 'cfg':
            original_filename = filename
            filename = f'{directory}{original_filename}'

           # if filename is None:
           #     filename = 'Atomic_series.dump'
           # else:
           #     if filename.lower().split('.')[-1] != 'dump':
           #         filename = '.'.join([filename.split('.')[0], 'dump'])

            f = open(filename,"w")
           # nsims = sims.size
            nsims = len(self.systems)
            sims = len(self.systems)
            energy = np.zeros(nsims)
            energy_weight = np.ones(nsims)
            # force = np.zeros((nsims,3))
            force_weight = np.ones(nsims)
            # stress1 = np.zeros((nsims,3))
            for nn in range(nsims):
                ids = [i+1 for i in range(self.systems[nn].natoms)]
                f.write('BEGIN_CFG\n')
                f.write(' Size\n')
                f.write(f'    {self.systems[nn].natoms}\n')
                f.write(' Supercell\n')
                for i in range(3):
                    f.write("%.16f      %.16f      %.16f\n" % (self.systems[nn].box[0,i],
                                                               self.systems[nn].box[1,i],
                                                               self.systems[nn].box[2,i]))
                # if isinstance(self.systems[nn].force, np.ndarray):
                if self.systems[nn]._do_force:
                    f.write(' AtomData:  id type       cartes_x      '
                    'cartes_y      cartes_z           fx          fy          fz\n')
                    for j in range(self.systems[nn].natoms):
                        f.write("             %d    %d       %.16f      %.16f      "
                        "%.16f    %.16f    %.16f    %.16f\n" % (ids[j], 
                                                                self.systems[nn].types[j]-1,
                                                                self.systems[nn].atoms[0,j],
                                                                self.systems[nn].atoms[1,j],
                                                                self.systems[nn].atoms[2,j],
                                                                self.systems[nn].force[0,j],
                                                                self.systems[nn].force[1,j],
                                                                self.systems[nn].force[2,j]))
                else:
                    f.write(' AtomData:  id type       cartes_x      cartes_y      cartes_z\n')
                    for j in range(self.systems[nn].natoms):
                        f.write("             %d    %d       %.16f      %.16f      %.16f"
                                "\n" % (ids[j],
                                        self.systems[nn].types[j]-1,
                                        self.systems[nn].atoms[0,j],
                                        self.systems[nn].atoms[1,j],
                                        self.systems[nn].atoms[2,j]))
                f.write(' Energy\n')
                f.write("        %.16f\n" % (self.systems[nn].energy))
                # if self.systems[nn].stress:
                if self.systems[nn]._do_stress:
                    f.write(' PlusStress:  xx          yy          zz          yz          xz          xy\n')
                    f.write("        %.16f    %.16f    %.16f    %.16f    %.16f    %.16f"
                    "\n" % (self.systems[nn].stress[0,0],
                            self.systems[nn].stress[1,1],
                            self.systems[nn].stress[2,2],
                            self.systems[nn].stress[1,2],
                            self.systems[nn].stress[0,2],
                            self.systems[nn].stress[0,1]))
                f.write('END_CFG\n')
                f.write('\n')
            f.close()

    # TODO - IMPLEMENT VASP FUNCTIONALITY

    #def write_vasp(self,
    #               INCAR: Union[str, None] = None,
    #               KPOINTS: Union[str, None] = None):
    #    """
    #    Writes VASP DFT inputs
    #    """
    #    if INCAR != None:
    #        with open(INCAR, 'r') as f:
    #            INCAR = f.read()
    #    else:
    #        INCAR = textwrap.dedent('''\
    #        System = Fe bcc fero
    #        # Parameters
    #            PREC = Accurate
    #            LASPH = True
    #            LORBIT = 10

    #        #Electronic relaxation
    #            ALGO = Normal

    #        #Electronic Relaxation
    #           ENCUT  =    520.00
    #           NELM   =    300    #max SCF steps
    #           NELMIN =      4    #min SCF steps
    #           EDIFF  =   1E-7    #stopping-criterion for ELM
    #           LREAL = F    # RM and RH


    #        #DOS related values:
    #            ISMEAR = 0
    #            SIGMA = 0.2

    #        #Write flags
    #            LWAVE  =      F    #write WAVECAR
    #            LCHARG =      F    #write CHGCAR
    #            LELF   =      F    #write electronic localiz. function (ELF)

    #        #Ionic relaxation
    #        #   NSW    =     100    #number of steps for IOM
    #            ISIF   =      2
    #            IBRION =      -1   # no update
    #        #   POTIM  =    0.1    #time-step for ionic-motion
    #        #   ISYM   =      0    #0-off
    #        ''')
    #        with open('INCAR', 'w') as f:
    #            f.write(INCAR)
    #    if KPOINTS != None:
    #        with open(KPOINTS, 'r') as f:
    #            KPOINTS = f.read()
    #    else:
    #        KPOINTS = textwrap.dedent('''\
    #        K-Spacing Value to Generate K-Mesh: 0.020
    #        0
    #        Monkhorst-Pack
    #          5 5 5
    #        0.0  0.0  0.0
    #        ''')
    #        with open('KPOINTS', 'w') as f:
    #            f.write(KPOINTS)
    #    nsims = len(self.systems)
    #    # print(nsims)
    #    sys.path.append(os.getcwd())
    #    self.export(filename='POSCAR',
    #                filetype='poscar',
    #                directory='all_poscars')
    #    for i in range(nsims):
    #        directory = f'input_{i+1}'
    #        if not os.path.isdir(directory):
    #            os.mkdir(directory)
    #        os.system(f'cp INCAR KPOINTS {directory}/.')
    #        os.chdir(directory)
    #        self.systems[i].export(filename='POSCAR',
    #                               filetype='poscar')
    #        os.system('mv POSCAR.poscar POSCAR')
    #        os.chdir('../')


