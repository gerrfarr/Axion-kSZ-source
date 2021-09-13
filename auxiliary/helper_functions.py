from ..theory.cosmology import Cosmology, CosmologyCustomH
import hashlib
import dataclasses
import json

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

def json_default(thing):
    try:
        return dataclasses.asdict(thing)
    except TypeError:
        pass
    raise TypeError(f"object of type {type(thing).__name__} not serializable")

def json_dumps(thing):
    return json.dumps(
        thing,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(',', ':'),
    )

def generate_cosmo_identifier(cosmo):
    """

    :type cosmo: Cosmology
    """
    return hashlib.md5(json_dumps((cosmo.h, cosmo.omegaCDM, cosmo.omegaB, cosmo.omega_axion, cosmo.m_axion, cosmo.n_s, cosmo.A_s, type(cosmo) == CosmologyCustomH)).encode('utf-8')).digest()