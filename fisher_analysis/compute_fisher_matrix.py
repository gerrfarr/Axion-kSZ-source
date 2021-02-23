import numpy as np
from copy import copy
from axion_kSZ_source.auxiliary.helper_functions import is_array


def make_fisher_matrix(cosmo_derivs, nuisance_derivs, z_vals, covmats):
    fisher_dim =len(cosmo_derivs) + len(nuisance_derivs) * len(z_vals)
    fisher_matrix =np.zeros((fisher_dim, fisher_dim))

    for z_i in range(len(z_vals)):
        inv_cov =np.linalg.inv(covmats[z_i])

        for i in range(len(cosmo_derivs)):
            for j in range(i, len(cosmo_derivs)):
                f=np.dot(cosmo_derivs[i,z_i], np. dot(inv_cov, cosmo_derivs[j,z_i]) )
                fisher_matrix[i,j]+=f
                if i!=j:
                    fisher_matrix[j,i]+=f

            for j in range(len(nuisance_derivs)):
                pos=len(cosmo_derivs)+z_i*len(nuisance_derivs)+j

                fisher_matrix[i,pos]=np.dot(cosmo_derivs[i,z_i], np. dot(inv_cov, nuisance_derivs[j,z_i]))
                fisher_matrix[pos,i]=fisher_matrix[i, pos]

        for i in range(len(nuisance_derivs)):
            for j in range(i, len(nuisance_derivs)):
                pos_i=len(cosmo_derivs)+z_i*len(nuisance_derivs)+i
                pos_j=len(cosmo_derivs)+z_i*len(nuisance_derivs)+j

                fisher_matrix[pos_i,pos_j]=np.dot(nuisance_derivs[i,z_i], np. dot(inv_cov, nuisance_derivs[j,z_i]) )
                if pos_i!=pos_j:
                    fisher_matrix[pos_j,pos_i]=fisher_matrix[pos_i,pos_j]

    return fisher_matrix


def get_deriv_sets(all_derivs, params, param_step_sizes):
    if params[0] in param_step_sizes.keys() and is_array(param_step_sizes[params[0]]):
        these_derivs = all_derivs[0:len(param_step_sizes[params[0]])]
        other_derivs = all_derivs[len(param_step_sizes[params[0]]):]
        if other_derivs.shape[0] == 0:
            sets = []
            for i in range(len(param_step_sizes[params[0]])):
                if not np.all(these_derivs[i] == np.nan):
                    sets.append([these_derivs[i]])
            return sets
        else:
            sets = get_deriv_sets(other_derivs, params[1:], param_step_sizes)
            output_sets = []
            for set in sets:
                for i in range(len(param_step_sizes[params[0]])):
                    if not np.all(these_derivs[i] == np.nan):
                        new_set = copy(set)
                        new_set.insert(0, these_derivs[i])
                        output_sets.append(new_set)

            return output_sets

    else:
        if all_derivs.shape[0] == 1:
            return [[all_derivs[0]]]
        else:
            sets = get_deriv_sets(all_derivs[1:], params[1:], param_step_sizes)
            for set in sets:
                set.insert(0, all_derivs[0])
            return sets
