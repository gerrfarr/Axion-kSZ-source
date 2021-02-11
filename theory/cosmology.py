import numpy as np
from scipy.interpolate import interp1d as interpolate
from ..axion_camb_wrappers.hubble_interpolation import HubbleInterpolation

class Cosmology(object):
    ##DEFINE CONSTANTS
    ZMAX = 1100  # redshift at suface of last scattering
    H0INV = 3000  # h/H0INV=H0 in units where c=1 (also: Hubble radius)
    T0 = 2.73e6  # CMB temperature in muK
    SIGMA_T = 6.6e-25  # cm^2 Thompson cross section
    Mpc_to_cm = 3.086e24  # cm/Mpc
    Mpc_to_m = 3.08568e22  # m/Mpc
    E_to_invL = 1 / 1.9746359e-7  # m^-1/eV
    RHO_C = 1.86e-29  # g/cm^3/h^2 (rho_B=Omega_Bh2 RHO_C)
    m_planck_L = 1.91183e57  # 1/Mpc
    delta_crit = 1.686
    m_sun = 1.988e33  # g
    m_proton = 1.6726219e-24  # g

    def __init__(self):


        self.__h=0.6737
        self.__omega_cdm=0.1198
        self.__omega_b=0.02233
        self.__n_s=0.9652
        self.__tau=0.0540
        self.__A_s=np.exp(3.043)/1.0e10

        self.__OmegaR = 9.28656e-5

        self.__m_axion=1.0e-24
        self.__omega_axion=0.0

    @staticmethod
    def generate(m_axion=None, omega_axion=None, h=None, omega_cdm=None, omega_b=None, n_s=None, A_s=None, read_H_from_file=False):
        if not read_H_from_file:
            cos = Cosmology()
        else:
            cos = CosmologyCustomH()

        if h is not None:
            cos.__h=h

        if omega_cdm is not None:
            cos.__omega_cdm = omega_cdm

        if omega_b is not None:
            cos.__omega_b = omega_b

        if n_s is not None:
            cos.__n_s = n_s

        if A_s is not None:
            cos.__A_s = A_s

        if m_axion is not None:
            cos.__m_axion = m_axion

        if omega_axion is not None:
            cos.__omega_axion = omega_axion

        return cos

    @property
    def OmegaM(self):
        return (self.__omega_b+self.__omega_cdm)/self.__h**2

    @property
    def omegaM(self):
        return self.__omega_b+self.__omega_cdm

    @property
    def OmegaB(self):
        return self.__omega_b / self.__h**2

    @property
    def omegaB(self):
        return self.__omega_b

    @property
    def OmegaCDM(self):
        return self.__omega_cdm / self.__h**2

    @property
    def omegaCDM(self):
        return self.__omega_cdm

    @property
    def OmegaLambda(self):
        return 1.0-self.OmegaM-self.__OmegaR

    @property
    def H0(self):
        return 100.0*self.__h

    @property
    def h(self):
        return self.__h

    @property
    def n_s(self):
        return self.__n_s

    @property
    def tau(self):
        return self.__tau

    @property
    def A_s(self):
        return self.__A_s

    @property
    def log_1e10A_s(self):
        return np.log(1.0e10*self.__A_s)

    @property
    def m_axion(self):
        return self.__m_axion

    @property
    def omega_axion(self):
        return self.__omega_axion

    @property
    def rho_crit(self):
        """

        :return: critical density in units of eV
        """
        return 8.11299e-11*self.__h**2

    @property
    def rho_mean(self):
        """

        :return: mean matter density in units of h^{-1} M_sun/(h^{-1} Mpc)^3 = h^2 M_sun/Mpc^3
        """
        return self.omegaM*self.RHO_C*self.Mpc_to_cm**3/self.m_sun

    def E(self, z):
        return np.sqrt(self.__OmegaR * (1 + z)**4 + self.OmegaM * (1 + z)**3 + self.OmegaLambda + (1 - self.OmegaM - self.OmegaLambda) * (1 + z)**2)

    def H(self, z):
        return self.H0*self.E(z)


    def __repr__(self):
        return "h={:.3f}, omega_cdm={:.3f}, omega_b={:.3f}, n_s={:.3f}, tau={:.3f}, log_1e10A_s={:.3f}, m_axion={:.3E}, omega_axion={:.3f}".format(self.__h, self.__omega_cdm, self.__omega_b, self.__n_s, self.__tau, self.log_1e10A_s, self.__m_axion, self.__omega_axion)

    def __copy__(self):
        return Cosmology.generate(m_axion=self.m_axion, omega_axion=self.omega_axion, h=self.h, omega_cdm=self.omegaCDM, omega_b=self.omegaB, n_s=self.n_s, A_s=self.A_s, read_H_from_file=False)

class CosmologyCustomH(Cosmology):
    def __init__(self):
        super().__init__()
        self.__H_interp=None

    def set_H_interpolation(self, H_interp):
        """

        :type H_interp: HubbleInterpolation
        """
        self.__H_interp = H_interp

    def H(self, z):
        try:
            return self.__H_interp(z)*3000.0*100
        except TypeError:
            raise Exception("Interpolation of H is not defined. Set H first using function set_H_interpolation(H_interp)")

    def E(self, z):
        return self.H(z)/self.H0

    @staticmethod
    def generate(m_axion=None, omega_axion=None, h=None, omega_cdm=None, omega_b=None, n_s=None, A_s=None, read_H_from_file=True):
        return Cosmology.generate(m_axion, omega_axion, h, omega_cdm, omega_b, n_s, A_s, read_H_from_file)

    def __copy__(self):
        return Cosmology.generate(m_axion=self.m_axion, omega_axion=self.omega_axion, h=self.h, omega_cdm=self.omegaCDM, omega_b=self.omegaB, n_s=self.n_s, A_s=self.A_s, read_H_from_file=True)