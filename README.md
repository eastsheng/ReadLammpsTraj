## ReadLammpsTraj
- Read lammps dump trajectory

### Installation

- github:

  ```bash
  git clone https://github.com/eastsheng/ReadLammpsTraj
  cd ReadLammpsTraj
  pip install .
  ```

- pip

  ```bash
  pip install ReadLammpsTraj
  ```

  

### Usage

```python
Help on ReadLammpsTraj in module ReadLammpsTraj object:

class ReadLammpsTraj(builtins.object)
 |  ReadLammpsTraj(f)
 |  
 |  Read lammps trajectory file
 |  
 |  Methods defined here:
 |  
 |  TwoD_Density(self, nframe, mxyz, atomtype_n, Nx=1, Ny=1, Nz=1, mass_or_number='mass', id_type='mol')
 |      nframe: n-th frame
 |      mxyz: mass x y z
 |      natoms: tot number of atoms
 |      atomtype_n: type of molecules,list,natoms=[1,36], the 1 is the first atom type and 36 is the last one atom type
 |      Nx,Ny,Nz: layer number of x , y, z for calculating density, which is relate to the precision of density,
 |      and default is 1, that is, the total density.
 |      mass_or_number: "mass: mass density; number: number density"
 |      id_type:"mol" or "atom" for atomtype_n
 |  
 |  __init__(self, f)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  calc_PMF(self, r, gr, T)
 |      Calculating the potential of mean force (PMF) by ln RDF
 |      Parameters:
 |      - r: radial distance
 |      - gr: radial distribution functions (RDF)
 |      - T: Temerature
 |      return PMF
 |  
 |  calc_bulk_density(self, nframe, modify=False)
 |      calculate bulk mass density from lammpstrj
 |      Parameters:
 |      - nframe: number of frame
 |      - modify: need to modify mass, default False, modify={"C": 16.043}
 |      Return a density value
 |  
 |  calc_coordination_number(self, rho, r, gr)
 |      Applied the trapezoidal rule to integrate 
 |      the RDF cumulatively and stores the CN function
 |      Parameters:
 |      - rho: number density
 |      - r: radial distance
 |      - gr: radial distribution functions (RDF)
 |      return coordination number (cn)
 |  
 |  density(self, nframe, id_range, mass_dict, Nz=100, id_type='atom', density_type='mass', direction='z')
 |      Calculating the density
 |      Parameters:
 |      - nframe: number of frame
 |      - id_range: calculated id range of atoms [1,2]
 |      - mass_dict: masses dict
 |      - Nz: number of bins
 |      - id_type: "atoms" or "mol"
 |      - density_type: "mass" or "number"
 |      - direction: x, or y, or z
 |      return coord, rho
 |  
 |  dividing(self, L0, L1, lbin)
 |  
 |  dump(self, nframe, dumpfile=False)
 |  
 |  dump_minish(self, mframe, nframe, interval=1, dumpfile=False)
 |      dump minish lammpstrj file
 |      Parameters:
 |      - mframe: start number of frame
 |      - nframe: end number of frame
 |      - interval: interval of frame
 |      - dumpfile: lammpstrj file name
 |  
 |  dump_unwrap(self, mframe, nframe, interval=1, dumpfile=False)
 |      dump unwrap lammpstrj
 |      Parameters:
 |      - mframe: start number of frame
 |      - nframe: end number of frame
 |      - interval: interval of frame
 |      - dumpfile: lammpstrj file name
 |  
 |  msd(self, atomtype, mframe, nframe, interval, outputfile=False)
 |      calculating msd
 |      Parameters:
 |      - atomtype: atomtype, list, example: [1,2]
 |      - mframe: start number of frame
 |      - nframe: end number of frame
 |      - interval: interval number of frame
 |      - outputfile: msd file
 |  
 |  oneframe_alldensity(self, nframe, mxyz, Nbin, mass_dict=False, density_type='mass', direction='z')
 |      calculating density of all atoms......
 |      mxyz: array of mass, x, y, and z;
 |      Nbin: number of bins in x/y/z-axis
 |      mass_dict: masses of atoms ,default=False
 |      density_type: calculated type of density
 |  
 |  oneframe_moldensity(self, nframe, mxyz, Nbin, id_range, mass_dict=False, id_type='mol', density_type='mass', direction='z')
 |      calculating density of some molecules......
 |      mxyz: array of mass, x, y, and z;
 |      Nbin: number of bins in x/y/z-axis
 |      id_range: range of molecule/atom id;
 |      mass_dict: masses of atoms ,default=False
 |      id_type: according to the molecule/atom id, to recognize atoms, args: mol, atom
 |      density_type: calculated type of density
 |  
 |  rdf(self, mframe, nframe, interval, atomtype1, atomtype2, cutoff=12, Nb=120, rdffile=False)
 |      calculate rdf from lammpstrj
 |      Parameters:
 |      - mframe: start number of frame
 |      - nframe: end number of frame
 |      - interval: interval of frame
 |      - atomtype1: selected atom type1, a list, example, [1,2]
 |      - atomtype2: selected atom type2, a list, example, [1,2]
 |      - cutoff: cutoff, default 12 Angstrom
 |      - Nb: number of bins
 |      - rdffile: lammpstrj file name
 |  
 |  read_box(self, nframe)
 |      read box from header
 |      Parameters:
 |      nframe: number of frame
 |      Return a dict
 |  
 |  read_elements(self)
 |  
 |  read_header(self, nframe)
 |      read header of nth-frame
 |      Parameters:
 |      - nframe: number of frame
 |      Return a list
 |  
 |  read_info(self)
 |  
 |  read_item(self, item)
 |  
 |  read_lengths(self, nframe)
 |  
 |  read_mxyz(self, nframe, modify=False)
 |      read mass, and x, y, z coordinates of nth frame from traj...
 |      nframe: number of frame 
 |      modify: need to modify mass, default False, modify={"C": 16.043}
 |  
 |  read_mxyz_add_mass(self, nframe, atomtype_list, mass_list)
 |  
 |  read_num_of_frames(self)
 |  
 |  read_steps(self)
 |  
 |  read_traj(self, nframe)
 |      read data of nth frame from traj...
 |      nframe: number of frame
 |  
 |  read_types(self)
 |  
 |  read_vol(self, nframe)
 |      read and calculate vol from header
 |      Parameters:
 |      nframe: number of frame
 |      Return a vol value unit/A^3
 |  
 |  read_xyz(self, nframe)
 |      read x, y, z coordinates of nth frame from traj...
 |      nframe: number of frame
 |  
 |  trj2lmp(self, nframe, lmp, relmp)
 |      write lammps data from lammpstrj and original lmp
 |      Parameters:
 |      - nframe: number of frame
 |      - lmp: original lammps data
 |      - relmp: final writed lammps data
 |  
 |  zoning(self, sorted_traj, axis_range, direc='y')
 |      Divide a coordinate interval along a direction, such as, x or y or z
 |      sorted_traj: sorted lammps traj, pandas dataframe format, it includes at least 'id mol type x y z'
 |      axis_range: Divide interval, a list, such as, axis_range = [0,3.5], unit/Angstrom
 |      direc: The direction to be divided, default direc="y"
 |  
 |  zoning_molecule(self, sorted_traj, axis_range, direc='y')
 |      Divide a coordinate interval along a direction, such as, x or y or z
 |      sorted_traj: sorted lammps traj, pandas dataframe format, it includes at least 'id mol type x y z'
 |      axis_range: Divide interval, a list, such as, axis_range = [0,3.5], unit/Angstrom
 |      direc: The direction to be divided, default direc="y"
 |  
 |  zoning_water_in_hydrate(self, sorted_traj, axis_range, direc='y')
 |      Divide a coordinate interval along a direction, such as, x or y or z for hydrate/water big molecules
 |      sorted_traj: sorted lammps traj, pandas dataframe format, it includes at least 'id mol type x y z'
 |      axis_range: Divide interval, a list, such as, axis_range = [0,3.5], unit/Angstrom
 |      direc: The direction to be divided, default direc="y"
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)

None
```


