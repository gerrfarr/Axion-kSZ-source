import shutil
from ..theory.cosmology import Cosmology
import subprocess, shlex
import os
from .hubble_interpolation import HubbleInterpolation
from .power_interpolation import LinearPowerInterpolation
from .growth_interpolation import GrowthInterpolation

class AxionCAMBWrapper(object):
    camb_files=["_params.ini", "_evolution.dat", "_devolution.dat", "_a_vals.dat", "_matterpower_out.dat", "_transfer_out.dat"]
    def __init__(self, outpath, fileroot, log_path):
        """

        :type outpath: string
        :type log_path: string
        :type fileroot: string
        """
        self.__outpath = outpath
        self.__fileroot = fileroot
        self.__log_path = log_path

        from ..auxiliary.configs import param_file_path, axion_camb_path
        self.__camb_path = axion_camb_path
        self.__param_path = param_file_path


    def __call__(self, cosmo):
        """

        :type cosmo: Cosmology
        """

        success=True
        with open(self.__log_path, "a+", buffering=1) as log_file:
            command_line = self.__camb_path.replace(" ", "\ ")+" "+self.__param_path.replace(" ", "\ ")+" 1 {} 2 {} 3 {} 4 {} 5 {} 6 {} 7 {} 8 T 9 {} > {}".format(cosmo.omegaB, cosmo.omegaCDM, cosmo.omega_axion, cosmo.m_axion, cosmo.H0, cosmo.n_s, cosmo.A_s, self.__fileroot.replace(" ", "\ "), self.__log_path.replace(" ", "\ "))

            try:
                p = subprocess.run(shlex.split(command_line))
                if p.returncode != 0:
                    raise Exception("Subprocess finished with return code {}".format(p.returncode))
            except Exception as ex:
                log_file.write("CAMB failed with message: "+str(ex)+"\n")
                success=False
                raise Exception("CAMB failed!")
            finally:
                for path_appendix in self.camb_files:
                    try:
                        shutil.move(os.getcwd() + "/" + self.__fileroot + path_appendix, self.__outpath + self.__fileroot + path_appendix)
                    except FileNotFoundError as ex:
                        log_file.write("File {} could not be moved. It does not exist.\n".format(self.__fileroot + path_appendix))
                        success = False
                    else:
                        log_file.write("File {} successfully moved!\n".format(self.__fileroot + path_appendix))

        return success

    def get_linear_power(self):
        return LinearPowerInterpolation(self.__outpath+self.__fileroot+"_matterpower_out.dat")

    def get_growth(self):
        return GrowthInterpolation(self.__outpath+self.__fileroot)

    def get_hubble(self):
        return HubbleInterpolation(self.__outpath+self.__fileroot+"_a_vals.dat")

