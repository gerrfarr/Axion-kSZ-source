from ..theory.cosmology import Cosmology, CosmologyCustomH

def is_array(item, cl=None):
    try:
        if len(item)>1 and (cl is None or isinstance(item, cl)):
            return True
        else:
            return True
    except TypeError as ex:
        return False
    except AttributeError as ex:
        return False

def generate_cosmo_identifier(cosmo):
    """

    :type cosmo: Cosmology
    """
    return hash((cosmo.h, cosmo.omegaCDM, cosmo.omegaB, cosmo.omega_axion, cosmo.m_axion, cosmo.n_s, cosmo.A_s, type(cosmo) == CosmologyCustomH))