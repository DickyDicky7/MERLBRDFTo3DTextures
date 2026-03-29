import os
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import math
import math
import cv2
import cv2
import sys
import sys
import time
import time
from typing import Dict, Any, List
from typing import Dict, Any, List

pMax: int = 3
"""
pMax: int = 3
"""
Pi: float = math.pi
"""
Pi: float = math.pi
"""

def I0(x: float) -> float:
    val: float = 0.0
#   val: float = 0.0
    x2i: float = 1.0
#   x2i: float = 1.0
    ifact: int = 1
#   ifact: int = 1
    i4: int = 1
#   i4: int = 1
    i: int
#   i: int
    for i in range(10):
#   for i in range(10):
        if i > 1: ifact *= i
#       if i > 1: ifact *= i
        val += x2i / (i4 * (ifact**2))
#       val += x2i / (i4 * (ifact**2))
        x2i *= x * x
#       x2i *= x * x
        i4 *= 4
#       i4 *= 4
    return val
#   return val

def LogI0(x: float) -> float:
    if x > 12:
#   if x > 12:
        return x + 0.5 * (-math.log(2 * Pi) + math.log(1 / x) + 1 / (8 * x))
#       return x + 0.5 * (-math.log(2 * Pi) + math.log(1 / x) + 1 / (8 * x))
    else:
#   else:
        return math.log(I0(x=x))
#       return math.log(I0(x=x))

def Mp(cosThetaI: float, cosThetaO: float, sinThetaI: float, sinThetaO: float, v: float) -> float:
    a: float = cosThetaI * cosThetaO / v
#   a: float = cosThetaI * cosThetaO / v
    b: float = sinThetaI * sinThetaO / v
#   b: float = sinThetaI * sinThetaO / v
    mp: float
#   mp: float
    if v <= 0.1:
#   if v <= 0.1:
        mp = math.exp(LogI0(x=a) - b - 1 / v + 0.6931471805599453 + math.log(1 / (2 * v)))
#       mp = math.exp(LogI0(x=a) - b - 1 / v + 0.6931471805599453 + math.log(1 / (2 * v)))
    else:
#   else:
        mp = (math.exp(-b) * I0(x=a)) / (math.sinh(1 / v) * 2 * v)
#       mp = (math.exp(-b) * I0(x=a)) / (math.sinh(1 / v) * 2 * v)
    return mp
#   return mp

def FrDielectric(cosThetaI: float, etaI: float, etaT: float) -> float:
    cosThetaI = max(-1.0, min(1.0, cosThetaI))
#   cosThetaI = max(-1.0, min(1.0, cosThetaI))
    entering: bool = cosThetaI > 0.0
#   entering: bool = cosThetaI > 0.0
    if not entering:
#   if not entering:
        etaI, etaT = etaT, etaI
#       etaI, etaT = etaT, etaI
        cosThetaI = abs(cosThetaI)
#       cosThetaI = abs(cosThetaI)

    sinThetaI: float = math.sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI))
#   sinThetaI: float = math.sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI))
    sinThetaT: float = etaI / etaT * sinThetaI
#   sinThetaT: float = etaI / etaT * sinThetaI

    if sinThetaT >= 1:
#   if sinThetaT >= 1:
        return 1.0
#       return 1.0
    cosThetaT: float = math.sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT))
#   cosThetaT: float = math.sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT))
    Rparl: float = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT))
#   Rparl: float = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT))
    Rperp: float = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT))
#   Rperp: float = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT))
    return (Rparl * Rparl + Rperp * Rperp) / 2.0
#   return (Rparl * Rparl + Rperp * Rperp) / 2.0

def Ap(cosThetaO: float, eta: float, h: float, T: npt.NDArray[np.float64]) -> List[npt.NDArray[np.float64]]:
    ap: List[npt.NDArray[np.float64]] = [np.zeros(shape=3) for _ in range(pMax + 1)]
#   ap: List[npt.NDArray[np.float64]] = [np.zeros(shape=3) for _ in range(pMax + 1)]
    cosGammaO: float = math.sqrt(max(0.0, 1.0 - h * h))
#   cosGammaO: float = math.sqrt(max(0.0, 1.0 - h * h))
    cosTheta: float = cosThetaO * cosGammaO
#   cosTheta: float = cosThetaO * cosGammaO
    f: float = FrDielectric(cosThetaI=cosTheta, etaI=1.0, etaT=eta)
#   f: float = FrDielectric(cosThetaI=cosTheta, etaI=1.0, etaT=eta)
    ap[0] = np.array(object=[f, f, f])
#   ap[0] = np.array(object=[f, f, f])
    ap[1] = (1.0 - f)**2 * T
#   ap[1] = (1.0 - f)**2 * T
    p: int
#   p: int
    for p in range(2, pMax):
#   for p in range(2, pMax):
        ap[p] = ap[p - 1] * T * f
#       ap[p] = ap[p - 1] * T * f
    ap[pMax] = ap[pMax - 1] * f * T / (np.ones(shape=3) - T * f)
#   ap[pMax] = ap[pMax - 1] * f * T / (np.ones(shape=3) - T * f)
    return ap
#   return ap

def Phi(p: int, gammaO: float, gammaT: float) -> float:
    return 2 * p * gammaT - 2 * gammaO + p * Pi
#   return 2 * p * gammaT - 2 * gammaO + p * Pi

def Logistic(x: float, s: float) -> float:
    x = abs(x)
#   x = abs(x)
    return math.exp(-x / s) / (s * (1 + math.exp(-x / s))**2)
#   return math.exp(-x / s) / (s * (1 + math.exp(-x / s))**2)

def LogisticCDF(x: float, s: float) -> float:
    return 1 / (1 + math.exp(-x / s))
#   return 1 / (1 + math.exp(-x / s))

def TrimmedLogistic(x: float, s: float, a: float, b: float) -> float:
    return Logistic(x=x, s=s) / (LogisticCDF(x=b, s=s) - LogisticCDF(x=a, s=s))
#   return Logistic(x=x, s=s) / (LogisticCDF(x=b, s=s) - LogisticCDF(x=a, s=s))

def Np(phi: float, p: int, s: float, gammaO: float, gammaT: float) -> float:
    dphi: float = phi - Phi(p=p, gammaO=gammaO, gammaT=gammaT)
#   dphi: float = phi - Phi(p=p, gammaO=gammaO, gammaT=gammaT)
    while dphi > Pi: dphi -= 2 * Pi
#   while dphi > Pi: dphi -= 2 * Pi
    while dphi < -Pi: dphi += 2 * Pi
#   while dphi < -Pi: dphi += 2 * Pi
    return TrimmedLogistic(x=dphi, s=s, a=-Pi, b=Pi)
#   return TrimmedLogistic(x=dphi, s=s, a=-Pi, b=Pi)

class HairBSDF:
    eta: float
#   eta: float
    sigma_a: npt.NDArray[np.float64]
#   sigma_a: npt.NDArray[np.float64]
    beta_m: float
#   beta_m: float
    beta_n: float
#   beta_n: float
    alpha: float
#   alpha: float
    v: npt.NDArray[np.float64]
#   v: npt.NDArray[np.float64]
    s: float
#   s: float
    sin2kAlpha: npt.NDArray[np.float64]
#   sin2kAlpha: npt.NDArray[np.float64]
    cos2kAlpha: npt.NDArray[np.float64]
#   cos2kAlpha: npt.NDArray[np.float64]

    def __init__(self, eta: float, sigma_a: npt.NDArray[np.float64], beta_m: float, beta_n: float, alpha: float) -> None:
        self.eta = eta
#       self.eta = eta
        self.sigma_a = np.array(object=sigma_a)
#       self.sigma_a = np.array(object=sigma_a)
        self.beta_m = beta_m
#       self.beta_m = beta_m
        self.beta_n = beta_n
#       self.beta_n = beta_n
        self.alpha = alpha
#       self.alpha = alpha

        self.v = np.zeros(shape=pMax + 1)
#       self.v = np.zeros(shape=pMax + 1)
        self.v[0] = (0.726 * beta_m + 0.812 * beta_m**2 + 3.7 * beta_m**20)**2
#       self.v[0] = (0.726 * beta_m + 0.812 * beta_m**2 + 3.7 * beta_m**20)**2
        self.v[1] = 0.25 * self.v[0]
#       self.v[1] = 0.25 * self.v[0]
        self.v[2] = 4 * self.v[0]
#       self.v[2] = 4 * self.v[0]
        p: int
#       p: int
        for p in range(3, pMax + 1):
#       for p in range(3, pMax + 1):
            self.v[p] = self.v[2]
#           self.v[p] = self.v[2]

        self.s = math.sqrt(math.pi / 8.0) * (0.265 * beta_n + 1.194 * beta_n**2 + 5.372 * beta_n**22)
#       self.s = math.sqrt(math.pi / 8.0) * (0.265 * beta_n + 1.194 * beta_n**2 + 5.372 * beta_n**22)

        self.sin2kAlpha = np.zeros(shape=3)
#       self.sin2kAlpha = np.zeros(shape=3)
        self.cos2kAlpha = np.zeros(shape=3)
#       self.cos2kAlpha = np.zeros(shape=3)
        self.sin2kAlpha[0] = math.sin(math.radians(alpha))
#       self.sin2kAlpha[0] = math.sin(math.radians(alpha))
        self.cos2kAlpha[0] = math.cos(math.radians(alpha))
#       self.cos2kAlpha[0] = math.cos(math.radians(alpha))
        i: int
#       i: int
        for i in range(1, 3):
#       for i in range(1, 3):
            self.sin2kAlpha[i] = 2 * self.cos2kAlpha[i - 1] * self.sin2kAlpha[i - 1]
#           self.sin2kAlpha[i] = 2 * self.cos2kAlpha[i - 1] * self.sin2kAlpha[i - 1]
            self.cos2kAlpha[i] = self.cos2kAlpha[i - 1]**2 - self.sin2kAlpha[i - 1]**2
#           self.cos2kAlpha[i] = self.cos2kAlpha[i - 1]**2 - self.sin2kAlpha[i - 1]**2

    def f(self, thetaI: float, thetaO: float, phiI: float, phiO: float, h: float) -> npt.NDArray[np.float64]:
        sinThetaI: float = math.sin(thetaI)
#       sinThetaI: float = math.sin(thetaI)
        cosThetaI: float = math.cos(thetaI)
#       cosThetaI: float = math.cos(thetaI)
        sinThetaO: float = math.sin(thetaO)
#       sinThetaO: float = math.sin(thetaO)
        cosThetaO: float = math.cos(thetaO)
#       cosThetaO: float = math.cos(thetaO)

        gammaO: float = math.asin(max(-1.0, min(1.0, h)))
#       gammaO: float = math.asin(max(-1.0, min(1.0, h)))

        sinThetaT: float = sinThetaO / self.eta
#       sinThetaT: float = sinThetaO / self.eta
        cosThetaT: float = math.sqrt(max(0.0, 1.0 - sinThetaT**2))
#       cosThetaT: float = math.sqrt(max(0.0, 1.0 - sinThetaT**2))

        etap: float = math.sqrt(max(0.0, self.eta**2 - sinThetaO**2)) / cosThetaO
#       etap: float = math.sqrt(max(0.0, self.eta**2 - sinThetaO**2)) / cosThetaO
        sinGammaT: float = h / etap
#       sinGammaT: float = h / etap
        cosGammaT: float = math.sqrt(max(0.0, 1.0 - sinGammaT**2))
#       cosGammaT: float = math.sqrt(max(0.0, 1.0 - sinGammaT**2))
        gammaT: float = math.asin(max(-1.0, min(1.0, sinGammaT)))
#       gammaT: float = math.asin(max(-1.0, min(1.0, sinGammaT)))

        T: npt.NDArray[np.float64] = np.exp(-self.sigma_a * (2 * cosGammaT / cosThetaT))
#       T: npt.NDArray[np.float64] = np.exp(-self.sigma_a * (2 * cosGammaT / cosThetaT))

        phi: float = phiI - phiO
#       phi: float = phiI - phiO
        ap: List[npt.NDArray[np.float64]] = Ap(cosThetaO=cosThetaO, eta=self.eta, h=h, T=T)
#       ap: List[npt.NDArray[np.float64]] = Ap(cosThetaO=cosThetaO, eta=self.eta, h=h, T=T)

        fsum: npt.NDArray[np.float64] = np.zeros(shape=3)
#       fsum: npt.NDArray[np.float64] = np.zeros(shape=3)

        p: int
#       p: int
        for p in range(pMax):
#       for p in range(pMax):
            sinThetaOp: float
#           sinThetaOp: float
            cosThetaOp: float
#           cosThetaOp: float
            if p == 0:
#           if p == 0:
                sinThetaOp = sinThetaO * self.cos2kAlpha[1] - cosThetaO * self.sin2kAlpha[1]
#               sinThetaOp = sinThetaO * self.cos2kAlpha[1] - cosThetaO * self.sin2kAlpha[1]
                cosThetaOp = cosThetaO * self.cos2kAlpha[1] + sinThetaO * self.sin2kAlpha[1]
#               cosThetaOp = cosThetaO * self.cos2kAlpha[1] + sinThetaO * self.sin2kAlpha[1]
            elif p == 1:
#           elif p == 1:
                sinThetaOp = sinThetaO * self.cos2kAlpha[0] + cosThetaO * self.sin2kAlpha[0]
#               sinThetaOp = sinThetaO * self.cos2kAlpha[0] + cosThetaO * self.sin2kAlpha[0]
                cosThetaOp = cosThetaO * self.cos2kAlpha[0] - sinThetaO * self.sin2kAlpha[0]
#               cosThetaOp = cosThetaO * self.cos2kAlpha[0] - sinThetaO * self.sin2kAlpha[0]
            elif p == 2:
#           elif p == 2:
                sinThetaOp = sinThetaO * self.cos2kAlpha[2] + cosThetaO * self.sin2kAlpha[2]
#               sinThetaOp = sinThetaO * self.cos2kAlpha[2] + cosThetaO * self.sin2kAlpha[2]
                cosThetaOp = cosThetaO * self.cos2kAlpha[2] - sinThetaO * self.sin2kAlpha[2]
#               cosThetaOp = cosThetaO * self.cos2kAlpha[2] - sinThetaO * self.sin2kAlpha[2]
            else:
#           else:
                sinThetaOp = sinThetaO
#               sinThetaOp = sinThetaO
                cosThetaOp = cosThetaO
#               cosThetaOp = cosThetaO

            cosThetaOp = abs(cosThetaOp)
#           cosThetaOp = abs(cosThetaOp)

            mp: float = Mp(cosThetaI=cosThetaI, cosThetaO=cosThetaOp, sinThetaI=sinThetaI, sinThetaO=sinThetaOp, v=self.v[p])
#           mp: float = Mp(cosThetaI=cosThetaI, cosThetaO=cosThetaOp, sinThetaI=sinThetaI, sinThetaO=sinThetaOp, v=self.v[p])
            np_val: float = Np(phi=phi, p=p, s=self.s, gammaO=gammaO, gammaT=gammaT)
#           np_val: float = Np(phi=phi, p=p, s=self.s, gammaO=gammaO, gammaT=gammaT)
            fsum += mp * ap[p] * np_val
#           fsum += mp * ap[p] * np_val

        fsum += Mp(cosThetaI=cosThetaI, cosThetaO=cosThetaO, sinThetaI=sinThetaI, sinThetaO=sinThetaO, v=self.v[pMax]) * ap[pMax] / (2.0 * Pi)
#       fsum += Mp(cosThetaI=cosThetaI, cosThetaO=cosThetaO, sinThetaI=sinThetaI, sinThetaO=sinThetaO, v=self.v[pMax]) * ap[pMax] / (2.0 * Pi)

        # We DO NOT divide by cosThetaI here, so that the baked value includes the cosine term.
#       # We DO NOT divide by cosThetaI here, so that the baked value includes the cosine term.
        # This means in the shader, you just sample and multiply by light color.
#       # This means in the shader, you just sample and multiply by light color.
        # (pbrt divides by cosThetaI so the integrator can multiply it back)
#       # (pbrt divides by cosThetaI so the integrator can multiply it back)
        return fsum
#       return fsum

def SigmaAFromConcentration(ce: float, cp: float) -> npt.NDArray[np.float64]:
    eumelaninSigmaA: npt.NDArray[np.float64] = np.array(object=[0.419, 0.697, 1.374])
#   eumelaninSigmaA: npt.NDArray[np.float64] = np.array(object=[0.419, 0.697, 1.374])
    pheomelaninSigmaA: npt.NDArray[np.float64] = np.array(object=[0.187, 0.400, 1.051])
#   pheomelaninSigmaA: npt.NDArray[np.float64] = np.array(object=[0.187, 0.400, 1.051])
    return ce * eumelaninSigmaA + cp * pheomelaninSigmaA
#   return ce * eumelaninSigmaA + cp * pheomelaninSigmaA

def evaluate_bulk_hair(hair_bsdf: HairBSDF, thetaI: float, thetaO: float, phiI: float, phiO: float, num_h_samples: int = 16) -> npt.NDArray[np.float64]:
    f_total: npt.NDArray[np.float64] = np.zeros(shape=3)
#   f_total: npt.NDArray[np.float64] = np.zeros(shape=3)
    dh: float = 2.0 / num_h_samples
#   dh: float = 2.0 / num_h_samples
    i: int
#   i: int
    for i in range(num_h_samples):
#   for i in range(num_h_samples):
        h: float = -1.0 + (i + 0.5) * dh
#       h: float = -1.0 + (i + 0.5) * dh
        f_total += hair_bsdf.f(thetaI=thetaI, thetaO=thetaO, phiI=phiI, phiO=phiO, h=h)
#       f_total += hair_bsdf.f(thetaI=thetaI, thetaO=thetaO, phiI=phiI, phiO=phiO, h=h)
    return f_total / num_h_samples
#   return f_total / num_h_samples

PROFILES: Dict[str, Dict[str, Any]] = {
    "brown": {"beta_m": 0.3, "beta_n": 0.3, "alpha": 2.0, "ce": 1.3, "cp": 0.0, "output": "hair_bsdf.exr"},
#   "brown": {"beta_m": 0.3, "beta_n": 0.3, "alpha": 2.0, "ce": 1.3, "cp": 0.0, "output": "hair_bsdf.exr"},
    "blonde": {"beta_m": 0.3, "beta_n": 0.3, "alpha": 2.0, "ce": 0.3, "cp": 0.0, "output": "hair_bsdf_blonde.exr"},
#   "blonde": {"beta_m": 0.3, "beta_n": 0.3, "alpha": 2.0, "ce": 0.3, "cp": 0.0, "output": "hair_bsdf_blonde.exr"},
    "dark_asian": {"beta_m": 0.25, "beta_n": 0.25, "alpha": 2.0, "ce": 8.0, "cp": 0.0, "output": "hair_bsdf_dark_asian.exr"},
#   "dark_asian": {"beta_m": 0.25, "beta_n": 0.25, "alpha": 2.0, "ce": 8.0, "cp": 0.0, "output": "hair_bsdf_dark_asian.exr"},
    "hazel": {"beta_m": 0.3, "beta_n": 0.3, "alpha": 2.0, "ce": 0.8, "cp": 0.2, "output": "hair_bsdf_hazel.exr"},
#   "hazel": {"beta_m": 0.3, "beta_n": 0.3, "alpha": 2.0, "ce": 0.8, "cp": 0.2, "output": "hair_bsdf_hazel.exr"},
    "white": {"beta_m": 0.3, "beta_n": 0.3, "alpha": 2.0, "ce": 0.0, "cp": 0.0, "output": "hair_bsdf_white.exr"},
#   "white": {"beta_m": 0.3, "beta_n": 0.3, "alpha": 2.0, "ce": 0.0, "cp": 0.0, "output": "hair_bsdf_white.exr"},
}

def bake_lut(profile_name: str) -> None:
    p: Dict[str, Any] = PROFILES[profile_name]
#   p: Dict[str, Any] = PROFILES[profile_name]
    # Hair parameters
#   # Hair parameters
    eta: float = 1.55
#   eta: float = 1.55
    beta_m: float = p["beta_m"]
#   beta_m: float = p["beta_m"]
    beta_n: float = p["beta_n"]
#   beta_n: float = p["beta_n"]
    alpha: float = p["alpha"]
#   alpha: float = p["alpha"]
    ce: float = p["ce"] # Eumelanin concentration
#   ce: float = p["ce"] # Eumelanin concentration
    cp: float = p["cp"] # Pheomelanin concentration
#   cp: float = p["cp"] # Pheomelanin concentration
    output_filename: str = p["output"]
#   output_filename: str = p["output"]

    sigma_a: npt.NDArray[np.float64] = SigmaAFromConcentration(ce=ce, cp=cp)
#   sigma_a: npt.NDArray[np.float64] = SigmaAFromConcentration(ce=ce, cp=cp)
    hair: HairBSDF = HairBSDF(eta=eta, sigma_a=sigma_a, beta_m=beta_m, beta_n=beta_n, alpha=alpha)
#   hair: HairBSDF = HairBSDF(eta=eta, sigma_a=sigma_a, beta_m=beta_m, beta_n=beta_n, alpha=alpha)

    res_theta: int = 64
#   res_theta: int = 64
    res_phi: int = 64
#   res_phi: int = 64

    grid_size: int = 8 # 8x8 grid of 64x64 slices = 512x512
#   grid_size: int = 8 # 8x8 grid of 64x64 slices = 512x512
    out_img: npt.NDArray[np.float32] = np.zeros(shape=(res_theta * grid_size, res_theta * grid_size, 3), dtype=np.float32)
#   out_img: npt.NDArray[np.float32] = np.zeros(shape=(res_theta * grid_size, res_theta * grid_size, 3), dtype=np.float32)

    print(f"Baking {res_theta}x{res_theta}x{res_phi} LUT for profile '{profile_name}'...")
#   print(f"Baking {res_theta}x{res_theta}x{res_phi} LUT for profile '{profile_name}'...")

    slice_idx: int
#   slice_idx: int
    for slice_idx in range(res_phi):
#   for slice_idx in range(res_phi):
        w: float = slice_idx / (res_phi - 1)
#       w: float = slice_idx / (res_phi - 1)
        phi_diff: float = w * Pi # phi_diff from 0 to Pi
#       phi_diff: float = w * Pi # phi_diff from 0 to Pi

        grid_x: int = slice_idx % grid_size
#       grid_x: int = slice_idx % grid_size
        grid_y: int = slice_idx // grid_size
#       grid_y: int = slice_idx // grid_size

        start_x: int = grid_x * res_theta
#       start_x: int = grid_x * res_theta
        start_y: int = grid_y * res_theta
#       start_y: int = grid_y * res_theta

        sys.stdout.write(f"\rProcessing slice {slice_idx + 1}/{res_phi}")
#       sys.stdout.write(f"\rProcessing slice {slice_idx + 1}/{res_phi}")
        sys.stdout.flush()
#       sys.stdout.flush()

        y: int
#       y: int
        for y in range(res_theta):
#       for y in range(res_theta):
            v: float = y / (res_theta - 1)
#           v: float = y / (res_theta - 1)
            thetaO: float = v * Pi - Pi/2.0 # thetaO from -Pi/2 to Pi/2
#           thetaO: float = v * Pi - Pi/2.0 # thetaO from -Pi/2 to Pi/2

            x: int
#           x: int
            for x in range(res_theta):
#           for x in range(res_theta):
                u: float = x / (res_theta - 1)
#               u: float = x / (res_theta - 1)
                thetaI: float = u * Pi - Pi/2.0 # thetaI from -Pi/2 to Pi/2
#               thetaI: float = u * Pi - Pi/2.0 # thetaI from -Pi/2 to Pi/2

                # Evaluate
#               # Evaluate
                val: npt.NDArray[np.float64] = evaluate_bulk_hair(hair_bsdf=hair, thetaI=thetaI, thetaO=thetaO, phiI=phi_diff, phiO=0.0)
#               val: npt.NDArray[np.float64] = evaluate_bulk_hair(hair_bsdf=hair, thetaI=thetaI, thetaO=thetaO, phiI=phi_diff, phiO=0.0)

                out_img[start_y + y, start_x + x] = val[::-1] # BGR for OpenCV
#               out_img[start_y + y, start_x + x] = val[::-1] # BGR for OpenCV

    print(f"\nSaving to {output_filename}...")
#   print(f"\nSaving to {output_filename}...")
    cv2.imwrite(filename=output_filename, img=out_img)
#   cv2.imwrite(filename=output_filename, img=out_img)
    print("Done!")
#   print("Done!")

if __name__ == "__main__":
    import argparse
#   import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Bake Hair BSDF LUT")
#   parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Bake Hair BSDF LUT")
    parser.add_argument("--profile", type=str, choices=list(PROFILES.keys()) + ["all"], default="all", help="Profile to bake")
#   parser.add_argument("--profile", type=str, choices=list(PROFILES.keys()) + ["all"], default="all", help="Profile to bake")
    args: argparse.Namespace = parser.parse_args()
#   args: argparse.Namespace = parser.parse_args()

    if args.profile == "all":
#   if args.profile == "all":
        prof: str
#       prof: str
        for prof in PROFILES:
#       for prof in PROFILES:
            bake_lut(profile_name=prof)
#           bake_lut(profile_name=prof)
    else:
#   else:
        bake_lut(profile_name=args.profile)
#       bake_lut(profile_name=args.profile)
