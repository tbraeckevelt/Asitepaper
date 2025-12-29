from ase.io import read
import numpy as np

unique_dist = []
for i in range(4000):
    atoms = read("opt_atoms_"+str(i)+".xyz")  # data set is not included in this repository
    dist_atoms = np.round(atoms.get_all_distances(mic = True), 2)
    flag = True
    for dist_mat in unique_dist:
        if (np.abs(dist_mat - dist_atoms)<0.02).all():
            flag = False
    if flag:
        for dist_mat in unique_dist:
            print(np.max(np.abs(dist_mat-dist_atoms)))
        unique_dist.append(dist_atoms)
        print(len(unique_dist))


#results for CsPbI3 with NVT cell at 600 K:
# gamma: 24 (+1 but that structure was 1 kJ/mol higher in energy)
# Csdelta: 1
# FAdelta: 6