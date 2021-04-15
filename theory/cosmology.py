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
    def generate(m_axion=None, omega_axion=None, axion_frac=None, h=None, omega_cdm=None, omega_b=None, n_s=None, A_s=None, read_H_from_file=False):
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

        if omega_axion is not None and axion_frac is None:
            cos.__omega_axion = omega_axion
        elif axion_frac is not None:
            assert (axion_frac <= 1.0 and axion_frac>=0.0)
            cos.axion_frac=axion_frac
        elif omega_axion is not None and axion_frac is not None:
            raise Exception("Can not simultaneously specify omega_axion and axion_frac.")


        return cos

    @property
    def OmegaM(self):
        return (self.__omega_b+self.__omega_cdm+self.__omega_axion)/self.__h**2

    @property
    def omegaM(self):
        return self.__omega_b+self.__omega_cdm+self.__omega_axion

    @property
    def OmegaB(self):
        return self.__omega_b / self.__h**2

    @property
    def omegaB(self):
        return self.__omega_b

    @omegaB.setter
    def omegaB(self, new):
        self.__omega_b = new

    @property
    def omegaDM(self):
        return self.omegaM-self.omegaB

    @omegaDM.setter
    def omegaDM(self, new):
        axion_frac = self.axion_frac
        self.__omega_cdm = (1-axion_frac)*new
        self.__omega_axion = axion_frac*new

    @property
    def OmegaCDM(self):
        return self.__omega_cdm / self.__h**2

    @property
    def omegaCDM(self):
        return self.__omega_cdm

    @omegaCDM.setter
    def omegaCDM(self, new):
        self.__omega_cdm = new

    @property
    def OmegaLambda(self):
        return 1.0-self.OmegaM-self.__OmegaR

    @property
    def H0(self):
        return 100.0*self.__h

    @H0.setter
    def H0(self, new):
        self.__h = new/100.0

    @property
    def h(self):
        return self.__h

    @h.setter
    def h(self, new):
        self.__h=new

    @property
    def n_s(self):
        return self.__n_s

    @n_s.setter
    def n_s(self, new):
        self.__n_s = new

    @property
    def tau(self):
        return self.__tau

    @tau.setter
    def tau(self, new):
        self.__tau = new

    @property
    def A_s(self):
        return self.__A_s

    @A_s.setter
    def A_s(self, new):
        self.__A_s = new

    @property
    def log_1e10A_s(self):
        return np.log(1.0e10*self.__A_s)

    @log_1e10A_s.setter
    def log_1e10A_s(self, new):
        self.__A_s = np.exp(new)/1.0e10

    @property
    def m_axion(self):
        return self.__m_axion

    @m_axion.setter
    def m_axion(self, new):
        self.__m_axion = new

    @property
    def omega_axion(self):
        return self.__omega_axion

    @omega_axion.setter
    def omega_axion(self, new):
        self.__omega_axion = new

    @property
    def axion_frac(self):
        return self.omega_axion/self.omegaDM

    @axion_frac.setter
    def axion_frac(self, new):
        assert(new<=1.0 and new>=0.0)
        new_omega_axion = self.omegaDM*new
        self.__omega_cdm = self.omegaDM*(1-new)
        self.__omega_axion = new_omega_axion

    @property
    def log_axion_frac(self):
        return np.log(self.axion_frac)

    @log_axion_frac.setter
    def log_axion_frac(self, new):
        self.axion_frac = np.exp(new)

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
        return "h={:.4f}, omega_cdm={:.4f}, omega_b={:.4f}, n_s={:.4f}, tau={:.4f}, log_1e10A_s={:4f}, m_axion={:.3E}, omega_axion={:.4f}".format(self.__h, self.__omega_cdm, self.__omega_b, self.__n_s, self.__tau, self.log_1e10A_s, self.__m_axion, self.__omega_axion)

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
            return self.__H_interp(z)*3000.0*100.0
        except TypeError:
            raise Exception("Interpolation of H is not defined. Set H first using function set_H_interpolation(H_interp)")

    def E(self, z):
        return self.H(z)/self.H0

    @staticmethod
    def generate(m_axion=None, omega_axion=None, axion_frac=None, h=None, omega_cdm=None, omega_b=None, n_s=None, A_s=None, read_H_from_file=True):
        return Cosmology.generate(m_axion, omega_axion, axion_frac, h, omega_cdm, omega_b, n_s, A_s, read_H_from_file)

    def __copy__(self):
        return Cosmology.generate(m_axion=self.m_axion, omega_axion=self.omega_axion, h=self.h, omega_cdm=self.omegaCDM, omega_b=self.omegaB, n_s=self.n_s, A_s=self.A_s, read_H_from_file=True)