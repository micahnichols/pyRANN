from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Union, Optional
# from . import system
from .system import system
from .series import series as series1

def load(filename: str,
         filetype: Optional[str] = None,
         timestep: Optional[Union[str, int, npt.ArrayLike]] = None,
         series: bool = False):
    '''
    Loads in a poscar file, LAMMPS data file, LAMMPS dump file, or cfg file and assigns
        the corresponding atoms, box, types, elements, number of atoms, and timestep to a
        system object. Stresses and forces will be added soon

    Args:
        filename : A string representation of the desired file to be loaded.
        filetype : Type of file to be loaded. Supported filetypes are poscar,
            data, dump, and cfg. If no filetype is given, it will be inferred from the 
            filename
        timestep: A string or integer representation of the timestep for the atomic
            system being loaded. Defaults to 0.
        series: Boolean value to determine if the file being loaded contains
            multiple timesteps.
    Returns:
        System Object or Series Object: A system object if series == False, otherwise a series object
    Raises:
        ValueError if the filetype is unsupported.
    '''
    if timestep is None:
        timestep = 0
    else:
        timestep = np.asarray(timestep)
    ext_keys = ['poscar', 'pos', 'data', 'dat', 'dump', 'cfg']
    if filetype is None:
        ext = filename.split('.')[-1]
        if ext.lower() in ext_keys:
            filetype = ext
        else:
            raise ValueError(f'Either filetype must be given or the file extension '
            'must be one of the following:\n{ext_keys}')
    else:
        filename = '.'.join([filename, filetype])
    if series:
        if filetype.lower() != 'dump' and filetype.lower() != 'cfg':
            raise ValueError('Currently, only a dump file or a cfg file containing multiple timesteps '
            'can be loaded as a series. This may change in the future.')

    if filetype.lower() == 'pos' or filetype.lower() == 'poscar':
        file = open(filename,'r')
        header = file.readline().strip()
        scale = file.readline().strip()
        box = np.empty((3,3))
        for i in range(3):
            line345 = file.readline().strip()
            bs = line345.split()
            box[0,i]=float(bs[0])
            box[1,i]=float(bs[1])
            box[2,i]=float(bs[2])
        elements = file.readline().strip().split()
        ntypesstr = file.readline().strip().split()
        ntypesint = np.array([int(i) for i in ntypesstr])
        ntypes = len(ntypesstr)
        typelen = np.empty(ntypes,dtype=np.uint8)
        # count = 0
        for i in range(ntypes):
            typelen[i] = int(ntypesstr[i])
        style = file.readline().strip().split()[0]
        direct = 0
        if (style=='direct' or style=='Direct' or style=='Crystal' or style == 'crystal'):
            direct = 1
        natoms = sum(typelen)
        Atoms = np.zeros([3,natoms])
        # types = -np.ones([natoms])
        types = np.zeros([natoms])
        count = np.array([np.sum(ntypesint[:j+1]) for j in range(len(ntypesint))])
        count2 = 0
        for i in range(natoms):
            line = file.readline().strip()
            lines = line.split()
            for j in range(3):
                Atoms[j,i] = lines[j]
            if len(lines) > 3:
                for j in range(ntypes):
                    if elements[j]==lines[3]:
                        types[i]=j+1
                if types[i]==-1:
                    print("unidentified element!")
            else:
                if i < count[count2]:
                    types[i]= count2+1
                else:
                    count2 += 1
                    types[i] = count2+1
        file.close()
        if (direct):
            Atoms = box@Atoms
        descriptor = filename
        return  system(atoms=Atoms, box=box, types=types, elements=elements, descriptor=descriptor)

    elif filetype == 'data' or filetype == 'dat':
        file = open(filename,'r')
        box = np.empty((3,3))
        origin = np.empty(3)
        atom_count = 0
        for line_num, content in enumerate(file):
            if not content.strip().split():
                continue
            elif content.strip().split()[0] == '#' or line_num == 1:
                header = content
                sec_header = content
            elif content.strip().split()[-1] == 'atoms':
                natoms = int(content.strip().split()[0])
                Id = np.empty(natoms)
                types = np.empty(natoms)
                atoms = np.empty((3, natoms))
            elif content.strip().split()[-1] == 'types' and content.strip().split()[1] == 'atom':
                ntypes = int(content.strip().split()[0])
            elif content.strip().endswith('xhi'):
                origin[0] = content.strip().split()[0]
                box[0,0] = content.strip().split()[1]
            elif content.strip().endswith('yhi'):
                origin[1] = content.strip().split()[0]
                box[1,1] = content.strip().split()[1]
            elif content.strip().endswith('zhi'):
                origin[2] = content.strip().split()[0]
                box[2,2] = content.strip().split()[1]
            elif content.strip().endswith('xy xz yz'):
                box[0,1] = content.strip().split()[0]
                box[0,2] = content.strip().split()[1]
                box[1,2] = content.strip().split()[2]
            elif content.strip().split()[0] == 'Atoms':
                sec_header = 'Atoms'
            elif content.strip().split() and sec_header == 'Atoms':
                (Id[atom_count],
                types[atom_count],
                atoms[0,atom_count],
                atoms[1,atom_count],
                atoms[2,atom_count],
                 *_) = np.float64(content.strip().split())
                atom_count += 1
            else:
                continue
        # header = file.readline()
        # file.readline()
        # line2 = file.readline().strip().split()
        # natoms = int(line2[0])
        # file.readline()
        # line3 = file.readline().strip().split()
        # ntypes = int(line3[0])
        # box = np.empty((3,3))
        # origin = np.empty(3)
        # file.readline()
        # for i in range(3):
        #     bs = file.readline().strip().split()
        #     origin[i]=bs[0]
        #     box[i,i]=bs[1]
        # line7=file.readline().strip()
        # if (not line7=="Atoms"):
        #     t = line7.split()
        #     box[0,1]=t[0]
        #     box[0,2]=t[1]
        #     box[1,2]=t[2]
        #     line7=file.readline()
        # file.readline()
        # Id = np.empty(natoms)
        # types = np.empty(natoms)
        # atoms = np.empty((3,natoms))
        # for i in range(natoms):
        #     line = file.readline().strip().split()
        #     (Id[i],types[i],atoms[0,i],atoms[1,i],atoms[2,i])=np.float64(line)
        file.close()
        descriptor = filename
        return system(atoms=atoms, box=box, types=types, descriptor=descriptor)
        # return (atoms,box,types,Id)

    elif filetype == 'dump' and not series:
        # TODO - ADD IN STRESSES
        # nsims = int(np.genfromtxt(filename, skip_header=1, max_rows=1, usecols=-1, dtype=int))
        nsims = 1
        file = open(filename,'r')
        Data_arr = []
        colstr_arr = []
        box_arr = []
        natoms_arr = []
        timestep_arr = []
        energy_arr = []
        if nsims == 0:
            nsims += 1
        for mwn in range(0, nsims):
            file.readline()
            # line2 = file.readline()
            line2 = file.readline().strip().split()
            # nsims = line2.strip().split()[-1]
            file.readline()
            line4 = file.readline()
            timestep = int(line2[0])
            if len(line2) > 1:
                energy = float(line2[1])
            else:
                energy = 0
            natoms = int(line4)
            file.readline()
            box = np.zeros([3,2])
            new_box = np.zeros([3,3])
            tilt = np.zeros([3])
            for i in range(3):
                line678 = file.readline()
                # bs = line678.split(' ')
                bs = line678.split()
                box[i,0]=float(bs[0])
                box[i,1]=float(bs[1])
                if (len(bs)>=3):
                    tilt[i]=float(bs[2])
            new_box[0,0] = ((box[0,1]-max([0, tilt[0], tilt[1], tilt[0]+tilt[1]]))
                            - (box[0,0]-min([0, tilt[0], tilt[1], tilt[0]+tilt[1]])))
            new_box[1,0] = tilt[0]
            new_box[1,1] = (box[1,1]-max(0,tilt[2])) - (box[1,0]-min(0,tilt[2]))
            new_box[2,0] = tilt[1]
            new_box[2,1] = tilt[2]
            new_box[2,2] = box[2,1]-box[2,0]

            line9 = file.readline()
            colstr = line9.replace('ITEM: ATOMS ','')
            colvec = colstr.strip().split(' ')
            cols = len(colvec)
            Data = np.zeros([natoms,cols])
            for i in range(natoms):
                line = file.readline()
                # lines = line.split(' ')
                lines = line.split()
                for j in range(cols):
                    Data[i,j]=float(lines[j])
            Data_arr.append(Data)
            colstr_arr.append(colstr)
            box_arr.append(new_box.T)
            natoms_arr.append(natoms)
            timestep_arr.append(timestep)
            energy_arr.append(energy)
        file.close()

        Data_arr = np.array([Data_arr])[0]
        per_atom_keywords = ['id', 'mass', 'type', 'x', 'y', 'z', 'xs', 'ys', 'zs', 'fx', 'fy', 'fz', 'vx', 'vy', 'vz']
        per_atom_dict = dict.fromkeys(per_atom_keywords)
        # print(f'{colvec = }')
        # print(f'{per_atom_keywords = }')
        for i in colvec:
            # print(f'{i = }')
            if i in per_atom_keywords:
                per_atom_index = colvec.index(i)
                per_atom_dict[i] = Data_arr[0,:,per_atom_index].T
        per_atom_dict = {k: v for k, v in per_atom_dict.items() if v is not None}
        # print(per_atom_dict)
        # Data_arr = np.array([Data_arr])[0]
        colstr_arr = np.array([colstr_arr])[0]
        box_arr = np.array([box_arr])[0]
        natoms_arr = np.array([natoms_arr])[0]
        # timestep_arr = np.array([timestep_arr])[0]
        timestep_arr = timestep_arr[0]
        energy_arr = np.array([energy_arr])[0]
        # atoms = Data_arr[0,:,2:].T
        atoms = np.array([v for k, v in per_atom_dict.items()
                          if k.startswith('x') or k.startswith('y') or k.startswith('z')])
        # print(f'atoms = \n{atoms.shape}')
        # types = Data_arr[0,:,1].T
        box = box_arr[0]
        descriptor = filename
        if 'fx' in per_atom_dict.keys():
            force = np.array([per_atom_dict['fx'], per_atom_dict['fy'], per_atom_dict['fz']])
            # print(f'{force = }')
            # print(f'{force.shape = }')
        else:
            force = None
        if 'type' in per_atom_dict.keys():
            # types = np.array([per_atom_dict['type']])
            types = per_atom_dict['type']
        else:
            types = np.ones(atoms.shape[1])
        # TODO - add in id, mass, and velocities to system description possibly
        return system(atoms=atoms, box=box, timestep=timestep_arr, types=types, descriptor=descriptor, force=force)

    elif filetype.lower() == 'cfg' and series:
        nsims = 0
        maxsize = 0
        file = open(filename)
        while True:
            line = file.readline()
            if not line:
                break
            if line=="BEGIN_CFG\n":
                nsims=nsims+1
            if line==" Size\n":
                line = file.readline()
                size=int(line)
                if size>maxsize:
                    maxsize=size
        file.close()
        nsims = 0
        with open (filename, 'r') as file:
            for line in file:
                if line=='BEGIN_CFG\n':
                    nsims += 1
            file.close()
        # print(nsims)
        file = open(filename,'r')
        dt = np.dtype([('index', int, (1,)), ('box', float, (3,3)),
                       ('natoms',int, (1,)),('x',float,(maxsize,3)),
                       ('f',float,(maxsize,3)),('id',int,maxsize),
                       ('type',int,(maxsize,)),('stress',float,(3,3)),
                       ('energy',float,(1,)),('energy_weight',float,(1,)),
                       ('force_weight',float,(1,))])
        # print(f'{dt = }')
        sims = np.empty((nsims),dtype=dt)
        series_list = []
        do_force = False
        do_stress = False
        for nn in range(nsims):
            # print(nn)
            sims[nn]['index']=nn
            line = file.readline()
            while line!="BEGIN_CFG\n":
                line = file.readline()
            isopen = 1
            while isopen:
                # print(f'{line}\n')
                sims[nn]['energy_weight']=1
                sims[nn]['force_weight']=1
                line = file.readline()
                if line==' Size\n':
                    # print('size')
                    line=file.readline()
                    sims[nn]['natoms']=int(line)
                elif line==' Supercell\n':
                    # print(' Supercell\n')
                    for i in range(3):
                        line=file.readline()
                        ls = line.split()
                        for j in range(3):
                            sims[nn]['box'][j,i]=float(ls[j])
                elif line[:10]==' AtomData:':
                    if line.strip().split()[-1] in ['fx', 'fy', 'fz']:
                        do_force = True
                    else:
                        do_force = False
                    # print(' AtomData\n')
                    for i in range(sims[nn]['natoms'][0]):
                        line = file.readline()
                        ls = line.split()
                        sims[nn]['id'][i]=int(ls[0])
                        sims[nn]['type'][i]=int(ls[1])+1
                        sims[nn]['x'][i,:]=(float(ls[2]),float(ls[3]),float(ls[4]))
                        if do_force:
                            sims[nn]['f'][i,:]=(float(ls[5]),float(ls[6]),float(ls[7]))
                elif line[:12]==" PlusStress:":
                    do_stress = True
                    # print(' PlusStress\n')
                    line = file.readline()
                    ls = line.strip().split()
                    stressvoight = (float(ls[0]),float(ls[1]),float(ls[2]),float(ls[3]),float(ls[4]),float(ls[5]))
                    sims[nn]['stress'][0,0]=stressvoight[0]
                    sims[nn]['stress'][1,1]=stressvoight[1]
                    sims[nn]['stress'][2,2]=stressvoight[2]
                    sims[nn]['stress'][0,1]=stressvoight[3]
                    sims[nn]['stress'][1,0]=stressvoight[3]
                    sims[nn]['stress'][0,2]=stressvoight[4]
                    sims[nn]['stress'][2,0]=stressvoight[4]
                    sims[nn]['stress'][1,2]=stressvoight[5]
                    sims[nn]['stress'][2,1]=stressvoight[5]
                elif line==" Energy\n":
                    # print(' Energy\n')
                    line = file.readline()
                    sims[nn]['energy']=float(line)
                elif line=='END_CFG\n':
                    # file.readline()
                    # file.readline()
                    # print('END_CFG\n')
                    isopen=0
                elif line[:8]==" Feature":
                    # print(' Feature')
                    # file.readline()
                    pass
                elif line=='BEGIN_CFG\n':
                    # print('BEGIN_CFG\n')
                    file.readline()
                    # file.readline()
                elif line=='\n' and nn!=nsims-1:
                    # print('BLANK LINE')
                    file.readline()
                else:
                    # print('What')
                    # print("unexpected cfg content!")
                    isopen=0

            timestep = nn
            types = sims[timestep]['type'][:sims[timestep]['natoms'][0]]
            atoms = sims[timestep]['x'][:sims[timestep]['natoms'][0]].T
            box = sims[timestep]['box']
            energy = sims[timestep]['energy'][0]
            if do_force:
                force = sims[timestep]['f'][:sims[timestep]['natoms'][0]].T
            else:
                force = None
            if do_stress:
                stress = sims[timestep]['stress']
            else:
                stress = None
            series_list.append(system(atoms=atoms, box=box, types=types,
                                      timestep=nn, energy=energy, force=force, stress=stress))

        # TODO - ADD IN STRESSES, FORCES, ETC
        # print(sims[-2]['x'][:sims[-2]['natoms'][0]].T)
        # print(sims[-2]['type'][:sims[-2]['natoms'][0]])
        types = sims[timestep]['type'][:sims[timestep]['natoms'][0]]
        atoms = sims[timestep]['x'][:sims[timestep]['natoms'][0]].T
        box = sims[timestep]['box']
        # return sims
        descriptor = filename
        # return system(atoms=atoms, box=box, types=types, descriptor=descriptor)
        return series1(series_list, descriptor=descriptor)

    # TODO - ADD IN STRESSES
    elif filetype.lower() == 'dump' and series:
        with open(filename, 'r') as f:
            content = f.read()
        nsims = content.count('TIMESTEP')
        # print(f'{nsims = }')
        # nsims = int(np.genfromtxt(filename, skip_header=1, max_rows=1, usecols=-1, dtype=int))
        file = open(filename,'r')
        Data_arr = []
        colstr_arr = []
        box_arr = []
        natoms_arr = []
        timestep_arr = []
        energy_arr = []
        series_list = []
        for mwn in range(0, nsims):
            file.readline()
            # line2 = file.readline()
            line2 = file.readline().strip().split()
            # nsims = line2.strip().split()[-1]
            # print(f'{nsims = }')
            file.readline()
            line4 = file.readline()
            timestep = int(line2[0])
            try:
                energy = float(line2[1])
            except:
                energy = 0.0
            natoms = int(line4)
            file.readline()
            box = np.zeros([3,2])
            new_box = np.zeros([3,3])
            tilt = np.zeros([3])
            for i in range(3):
                line678 = file.readline()
                # print(f'{line678 = }')
                # bs = line678.split(' ')
                bs = line678.split()
                # print(f'{bs = }')
                box[i,0]=float(bs[0])
                box[i,1]=float(bs[1])
                if (len(bs)>=3):
                    tilt[i]=float(bs[2])
            new_box[0,0] = ((box[0,1]-max([0, tilt[0], tilt[1], tilt[0]+tilt[1]]))
                            - (box[0,0]-min([0, tilt[0], tilt[1], tilt[0]+tilt[1]])))
            new_box[1,0] = tilt[0]
            new_box[1,1] = (box[1,1]-max(0,tilt[2])) - (box[1,0]-min(0,tilt[2]))
            new_box[2,0] = tilt[1]
            new_box[2,1] = tilt[2]
            new_box[2,2] = box[2,1]-box[2,0]

            line9 = file.readline()
            colstr = line9.replace('ITEM: ATOMS ','')
            colvec = colstr.split(' ')
            cols = len(colvec)
            Data = np.zeros([natoms,cols])
            for i in range(natoms):
                line = file.readline()
                # lines = line.split(' ')
                lines = line.split()
                # print(f'{lines = }')
                for j in range(cols):
                    Data[i,j]=float(lines[j])
            Data_arr.append(Data)
            # print(f'------------------------\n{Data[:,2:5].T = }\n---------------------------\n')
            colstr_arr.append(colstr)
            box_arr.append(new_box)
            natoms_arr.append(natoms)
            timestep_arr.append(timestep)
            energy_arr.append(energy)
            per_atom_keywords = ['id', 'mass', 'type', 'x', 'y', 'z', 'xs',
                                 'ys', 'zs', 'fx', 'fy', 'fz', 'vx', 'vy', 'vz']
            per_atom_dict = dict.fromkeys(per_atom_keywords)
            # print(f'{colvec = }')
            # print(f'{per_atom_keywords = }')
            colvec = [i.strip() for i in colvec]
            for i in colvec:
                # print(f'{i = }')
                if i in per_atom_keywords:
                    per_atom_index = colvec.index(i)
                    per_atom_dict[i] = Data[:,per_atom_index].T
            per_atom_dict = {k: v for k, v in per_atom_dict.items() if v is not None}
            atoms = np.array([v for k, v in per_atom_dict.items()
                              if k.startswith('x') or k.startswith('y') or k.startswith('z')])
            # print(f'{atoms = }')
            if 'fx' in per_atom_dict.keys():
                force = np.array([per_atom_dict['fx'], per_atom_dict['fy'], per_atom_dict['fz']])
                # print(f'{force = }')
                # print(f'{force.shape = }')
            else:
                # force = np.zeros((natoms, 3))
                # force = np.zeros((3, natoms))
                force = None
            if 'type' in per_atom_dict.keys():
                # types = np.array([per_atom_dict['type']])
                types = per_atom_dict['type']
            else:
                types = np.ones(atoms.shape[1])
            series_list.append(system(atoms=atoms, box=new_box.T, types=types, natoms=natoms,
                                      timestep=timestep, energy=energy, force=force, descriptor=filename))
        # file.close()
        # Data_arr = np.array([Data_arr])[0]
        # colstr_arr = np.array([colstr_arr])[0]
        # box_arr = np.array([box_arr])[0]
        # natoms_arr = np.array([natoms_arr])[0]
        # timestep_arr = np.array([timestep_arr])[0]
        # energy_arr = np.array([energy_arr])[0]
        # return (Data_arr,colstr_arr,box_arr,natoms_arr,timestep_arr, energy_arr
        return  series1(series_list, descriptor=filename)
