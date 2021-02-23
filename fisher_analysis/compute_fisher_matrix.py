import numpy as np


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