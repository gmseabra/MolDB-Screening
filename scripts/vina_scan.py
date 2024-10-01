#!/usr/bin/env python
"""
    Creates decoys for given complexes using Vina.
"""
import multiprocessing as mp
from functools import partial
from pathlib import Path
import time
import numpy as np
from vina import Vina
from tqdm import tqdm


def dock_ligands(target, center, size, ligand, n_cpus=4, exh=8, n_poses=1, e_range=5.0):
    """ Creates decoys for a given complex using Vina.
        ligand:         path to ligand PDBQT file
        target:         path to the target protein pdbqt file
        center:         [X,Y,Z] coords for grid center
        size:           [dx, dy, dz] dimensions of the grid box 
        n_cpus:         number of cpus to use
        exh:            exhaustiveness
        n_poses:        number of poses to generate
        e_range:        energy range to use when saving the decoys
    """

    target_file = Path(target)
    ligand_file = Path(ligand)
    ligand_name = ligand_file.stem

    docker = Vina(sf_name='Vina', cpu=n_cpus, verbosity=0)
    docker.set_receptor(str(target_file))

    docker.set_ligand_from_file(str(ligand_file))
    docker.compute_vina_maps(center, size)
    docker.dock(exhaustiveness=exh, n_poses=n_poses)

    # save the results
    docker.write_poses(f"out/{ligand_name}.pdbqt",
                       n_poses,
                       energy_range=e_range,
                       overwrite=True)

    return

if __name__ == "__main__":

    start = time.time()

    # Control variables
    EXHAUSTIVENESS     = 32
    N_POSES            =  1
    PARALLEL_PROCESSES = 50
    VINA_CPUS          =  4
    ENERGY_RANGE       =  5.0

    # -- Grid Definition
    center_x = 23.266
    center_y = 56.891
    center_z = 86.524

    size_x = 18.0
    size_y = 18.0
    size_z = 18.0

    print("Docking library with Vina:")
    print(f"  Exhaustiveness={EXHAUSTIVENESS}, N poses={N_POSES}")
    print(f"  Parallel processes={PARALLEL_PROCESSES}, Vina CPUs={VINA_CPUS}, Energy range={ENERGY_RANGE}")

    root_dir = Path.cwd()
    lig_library = Path("/blue/lic/seabra/databases/screening/Enamine/HLL-460/sample_10K")

    Path(root_dir,"out").mkdir(exist_ok=True)
    target  = Path("/blue/lic/seabra/li/znf146/vina_scan/3V3L_A_prepared.pdbqt")
    ligands = list(lig_library.glob("**/*.pdbqt"))

    print(f"  Found: Target: {target} \n"
          f"         {len(ligands)} ligands in {lig_library}.\n")

    # Prints the citation for Vina
    Vina().cite()
    
    # Generate decoys for each complex. Use multiprocessing to speed up the process.
    dock_ligands_partial = partial(dock_ligands, 
                                    target,
                                    [center_x,center_y,center_z],
                                    [size_x,size_y,size_z],
                                    n_cpus=VINA_CPUS,
                                    exh=EXHAUSTIVENESS,
                                    n_poses=N_POSES,
                                    e_range=ENERGY_RANGE)
    
    with mp.Pool(processes=PARALLEL_PROCESSES) as pool:
        with tqdm(total=len(ligands)) as pbar:
            for _ in pool.imap_unordered(dock_ligands_partial, ligands):
                pbar.update()
   



    finish = time.time()
    elapsed = finish - start

    print( f"Done. Elapsed time={elapsed:10.2f} seconds." )
