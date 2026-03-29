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
import argparse
import argparse
from typing import Dict, Any, List, Tuple
from typing import Dict, Any, List, Tuple

# Porting pbrt-v3 BSSRDF Math (Beam Diffusion)
# Porting pbrt-v3 BSSRDF Math (Beam Diffusion)
# Based on pbrt-v3/src/core/bssrdf.cpp
# Based on pbrt-v3/src/core/bssrdf.cpp

def FresnelMoment1(eta: float) -> float:
    eta2: float = eta * eta
#   eta2: float = eta * eta
    eta3: float = eta2 * eta
#   eta3: float = eta2 * eta
    eta4: float = eta3 * eta
#   eta4: float = eta3 * eta
    eta5: float = eta4 * eta
#   eta5: float = eta4 * eta
    if eta < 1:
#   if eta < 1:
        return 0.45966 - 1.73965 * eta + 3.37668 * eta2 - 3.904945 * eta3 + 2.49277 * eta4 - 0.68441 * eta5
#       return 0.45966 - 1.73965 * eta + 3.37668 * eta2 - 3.904945 * eta3 + 2.49277 * eta4 - 0.68441 * eta5
    else:
#   else:
        return -4.61686 + 11.1136 * eta - 10.4646 * eta2 + 5.11455 * eta3 - 1.27198 * eta4 + 0.12746 * eta5
#       return -4.61686 + 11.1136 * eta - 10.4646 * eta2 + 5.11455 * eta3 - 1.27198 * eta4 + 0.12746 * eta5

def FresnelMoment2(eta: float) -> float:
    eta2: float = eta * eta
#   eta2: float = eta * eta
    eta3: float = eta2 * eta
#   eta3: float = eta2 * eta
    eta4: float = eta3 * eta
#   eta4: float = eta3 * eta
    eta5: float = eta4 * eta
#   eta5: float = eta4 * eta
    if eta < 1:
#   if eta < 1:
        return 0.27614 - 0.87350 * eta + 1.12077 * eta2 - 0.65095 * eta3 + 0.07883 * eta4 + 0.04860 * eta5
#       return 0.27614 - 0.87350 * eta + 1.12077 * eta2 - 0.65095 * eta3 + 0.07883 * eta4 + 0.04860 * eta5
    else:
#   else:
        r_eta: float = 1 / eta
#       r_eta: float = 1 / eta
        r_eta2: float = r_eta * r_eta
#       r_eta2: float = r_eta * r_eta
        r_eta3: float = r_eta2 * r_eta
#       r_eta3: float = r_eta2 * r_eta
        return -547.033 + 45.3087 * r_eta3 - 218.725 * r_eta2 + 458.843 * r_eta + 404.557 * eta - 189.519 * eta2 + 54.9327 * eta3 - 9.00603 * eta4 + 0.63942 * eta5
#       return -547.033 + 45.3087 * r_eta3 - 218.725 * r_eta2 + 458.843 * r_eta + 404.557 * eta - 189.519 * eta2 + 54.9327 * eta3 - 9.00603 * eta4 + 0.63942 * eta5

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
    sinThetaT: float = (etaI / etaT) * sinThetaI
#   sinThetaT: float = (etaI / etaT) * sinThetaI
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

def PhaseHG(cosTheta: float, g: float) -> float:
    denom: float = 1 + g * g + 2 * g * cosTheta
#   denom: float = 1 + g * g + 2 * g * cosTheta
    return (1.0 / (4.0 * math.pi)) * (1.0 - g * g) / (denom * math.sqrt(denom))
#   return (1.0 / (4.0 * math.pi)) * (1.0 - g * g) / (denom * math.sqrt(denom))

def BeamDiffusionMS(sigma_s: float, sigma_a: float, g: float, eta: float, r: float) -> float:
    nSamples: int = 100
#   nSamples: int = 100
    Ed: float = 0.0
#   Ed: float = 0.0
    sigmap_s: float = sigma_s * (1.0 - g)
#   sigmap_s: float = sigma_s * (1.0 - g)
    sigmap_t: float = sigma_a + sigmap_s
#   sigmap_t: float = sigma_a + sigmap_s
    rhop: float = sigmap_s / sigmap_t
#   rhop: float = sigmap_s / sigmap_t
    D_g: float = (2.0 * sigma_a + sigmap_s) / (3.0 * sigmap_t * sigmap_t)
#   D_g: float = (2.0 * sigma_a + sigmap_s) / (3.0 * sigmap_t * sigmap_t)
    sigma_tr: float = math.sqrt(sigma_a / D_g)
#   sigma_tr: float = math.sqrt(sigma_a / D_g)
    fm1: float = FresnelMoment1(eta=eta)
#   fm1: float = FresnelMoment1(eta=eta)
    fm2: float = FresnelMoment2(eta=eta)
#   fm2: float = FresnelMoment2(eta=eta)
    ze: float = -2.0 * D_g * (1.0 + 3.0 * fm2) / (1.0 - 2.0 * fm1)
#   ze: float = -2.0 * D_g * (1.0 + 3.0 * fm2) / (1.0 - 2.0 * fm1)
    cPhi: float = 0.25 * (1.0 - 2.0 * fm1)
#   cPhi: float = 0.25 * (1.0 - 2.0 * fm1)
    cE: float = 0.5 * (1.0 - 3.0 * fm2)
#   cE: float = 0.5 * (1.0 - 3.0 * fm2)
    i: int
#   i: int
    for i in range(nSamples):
#   for i in range(nSamples):
        zr: float = -math.log(1.0 - (i + 0.5) / nSamples) / sigmap_t
#       zr: float = -math.log(1.0 - (i + 0.5) / nSamples) / sigmap_t
        zv: float = -zr + 2.0 * ze
#       zv: float = -zr + 2.0 * ze
        dr: float = math.sqrt(r * r + zr * zr)
#       dr: float = math.sqrt(r * r + zr * zr)
        dv: float = math.sqrt(r * r + zv * zv)
#       dv: float = math.sqrt(r * r + zv * zv)
        phiD: float = (1.0 / (4.0 * math.pi)) / D_g * (math.exp(-sigma_tr * dr) / dr - math.exp(-sigma_tr * dv) / dv)
#       phiD: float = (1.0 / (4.0 * math.pi)) / D_g * (math.exp(-sigma_tr * dr) / dr - math.exp(-sigma_tr * dv) / dv)
        EDn: float = (1.0 / (4.0 * math.pi)) * (zr * (1.0 + sigma_tr * dr) * math.exp(-sigma_tr * dr) / (dr * dr * dr) - zv * (1.0 + sigma_tr * dv) * math.exp(-sigma_tr * dv) / (dv * dv * dv))
#       EDn: float = (1.0 / (4.0 * math.pi)) * (zr * (1.0 + sigma_tr * dr) * math.exp(-sigma_tr * dr) / (dr * dr * dr) - zv * (1.0 + sigma_tr * dv) * math.exp(-sigma_tr * dv) / (dv * dv * dv))
        E: float = phiD * cPhi + EDn * cE
#       E: float = phiD * cPhi + EDn * cE
        kappa: float = 1.0 - math.exp(-2.0 * sigmap_t * (dr + zr))
#       kappa: float = 1.0 - math.exp(-2.0 * sigmap_t * (dr + zr))
        Ed += kappa * rhop * rhop * E
#       Ed += kappa * rhop * rhop * E
    return Ed / nSamples
#   return Ed / nSamples

def BeamDiffusionSS(sigma_s: float, sigma_a: float, g: float, eta: float, r: float) -> float:
    sigma_t: float = sigma_a + sigma_s
#   sigma_t: float = sigma_a + sigma_s
    rho: float = sigma_s / sigma_t
#   rho: float = sigma_s / sigma_t
    tCrit: float = r * math.sqrt(eta * eta - 1.0) if eta > 1.0 else 0.0
#   tCrit: float = r * math.sqrt(eta * eta - 1.0) if eta > 1.0 else 0.0
    Ess: float = 0.0
#   Ess: float = 0.0
    nSamples: int = 100
#   nSamples: int = 100
    i: int
#   i: int
    for i in range(nSamples):
#   for i in range(nSamples):
        ti: float = tCrit - math.log(1.0 - (i + 0.5) / nSamples) / sigma_t
#       ti: float = tCrit - math.log(1.0 - (i + 0.5) / nSamples) / sigma_t
        d: float = math.sqrt(r * r + ti * ti)
#       d: float = math.sqrt(r * r + ti * ti)
        cosThetaO: float = ti / d
#       cosThetaO: float = ti / d
        Ess += rho * math.exp(-sigma_t * (d + tCrit)) / (d * d) * PhaseHG(cosTheta=cosThetaO, g=g) * (1.0 - FrDielectric(cosThetaI=-cosThetaO, etaI=1.0, etaT=eta)) * abs(cosThetaO)
#       Ess += rho * math.exp(-sigma_t * (d + tCrit)) / (d * d) * PhaseHG(cosTheta=cosThetaO, g=g) * (1.0 - FrDielectric(cosThetaI=-cosThetaO, etaI=1.0, etaT=eta)) * abs(cosThetaO)
    return Ess / nSamples
#   return Ess / nSamples

def BSSRDF(sigma_s: float, sigma_a: float, g: float, eta: float, r: float) -> float:
    # Clamping r to prevent division by zero in the formulas
#   # Clamping r to prevent division by zero in the formulas
    r = max(0.0001, r)
#   r = max(0.0001, r)
    return BeamDiffusionSS(sigma_s=sigma_s, sigma_a=sigma_a, g=g, eta=eta, r=r) + BeamDiffusionMS(sigma_s=sigma_s, sigma_a=sigma_a, g=g, eta=eta, r=r)
#   return BeamDiffusionSS(sigma_s=sigma_s, sigma_a=sigma_a, g=g, eta=eta, r=r) + BeamDiffusionMS(sigma_s=sigma_s, sigma_a=sigma_a, g=g, eta=eta, r=r)

PROFILES: Dict[str, Dict[str, Any]] = {
    "skin1": {"sigma_s": [0.74, 0.88, 1.01], "sigma_a": [0.032, 0.17, 0.48], "g": 0.0, "eta": 1.33, "output": "skin_lut_skin1.exr"},
#   "skin1": {"sigma_s": [0.74, 0.88, 1.01], "sigma_a": [0.032, 0.17, 0.48], "g": 0.0, "eta": 1.33, "output": "skin_lut_skin1.exr"},
    "skin2": {"sigma_s": [1.09, 1.59, 1.79], "sigma_a": [0.013, 0.070, 0.145], "g": 0.0, "eta": 1.33, "output": "skin_lut_skin2.exr"},
#   "skin2": {"sigma_s": [1.09, 1.59, 1.79], "sigma_a": [0.013, 0.070, 0.145], "g": 0.0, "eta": 1.33, "output": "skin_lut_skin2.exr"},
    "white_fair_pink": {"sigma_s": [1.0, 1.2, 1.5], "sigma_a": [0.002, 0.012, 0.060], "g": 0.0, "eta": 1.33, "output": "skin_lut_white_fair_pink.exr"}
#   "white_fair_pink": {"sigma_s": [1.0, 1.2, 1.5], "sigma_a": [0.002, 0.012, 0.060], "g": 0.0, "eta": 1.33, "output": "skin_lut_white_fair_pink.exr"}
}

def integrate_skin(sigma_s: float, sigma_a: float, g: float, eta: float, r_curvature: float, ndotl: float, num_samples: int = 360) -> float:
    if r_curvature >= 100.0: # effectively flat
#   if r_curvature >= 100.0: # effectively flat
        return max(0.0, ndotl)
#       return max(0.0, ndotl)

    total_weight: float = 0.0
#   total_weight: float = 0.0
    total_light: float = 0.0
#   total_light: float = 0.0
    theta: float = math.acos(max(-1.0, min(1.0, ndotl)))
#   theta: float = math.acos(max(-1.0, min(1.0, ndotl)))

    dx: float = 2.0 * math.pi / num_samples
#   dx: float = 2.0 * math.pi / num_samples
    i: int
#   i: int
    for i in range(num_samples):
#   for i in range(num_samples):
        x: float = -math.pi + (i + 0.5) * dx
#       x: float = -math.pi + (i + 0.5) * dx
        dist: float = 2.0 * r_curvature * math.sin(abs(x) / 2.0)
#       dist: float = 2.0 * r_curvature * math.sin(abs(x) / 2.0)

        # BSSRDF profile evaluation
#       # BSSRDF profile evaluation
        profile_val: float = BSSRDF(sigma_s=sigma_s, sigma_a=sigma_a, g=g, eta=eta, r=dist)
#       profile_val: float = BSSRDF(sigma_s=sigma_s, sigma_a=sigma_a, g=g, eta=eta, r=dist)

        # Diffuse falloff
#       # Diffuse falloff
        diffuse: float = max(0.0, math.cos(theta + x))
#       diffuse: float = max(0.0, math.cos(theta + x))

        total_weight += profile_val
#       total_weight += profile_val
        total_light += profile_val * diffuse
#       total_light += profile_val * diffuse

    return total_light / total_weight if total_weight > 0 else max(0.0, ndotl)
#   return total_light / total_weight if total_weight > 0 else max(0.0, ndotl)

import concurrent.futures
import concurrent.futures
import multiprocessing
import multiprocessing

def compute_row(y: int, res_x: int, res_y: int, max_curvature: float, sigma_s: List[float], sigma_a: List[float], g: float, eta: float) -> Tuple[int, npt.NDArray[np.float32]]:
    v: float = y / (res_y - 1.0)
#   v: float = y / (res_y - 1.0)
    C: float = v * max_curvature
#   C: float = v * max_curvature
    r_curvature: float = 1.0 / C if C > 0.001 else 1000.0
#   r_curvature: float = 1.0 / C if C > 0.001 else 1000.0

    row_data: npt.NDArray[np.float32] = np.zeros(shape=(res_x, 3), dtype=np.float32)
#   row_data: npt.NDArray[np.float32] = np.zeros(shape=(res_x, 3), dtype=np.float32)
    x: int
#   x: int
    for x in range(res_x):
#   for x in range(res_x):
        u: float = x / (res_x - 1.0)
#       u: float = x / (res_x - 1.0)
        ndotl: float = u * 2.0 - 1.0 # -1 to 1 mapping
#       ndotl: float = u * 2.0 - 1.0 # -1 to 1 mapping

        val: npt.NDArray[np.float64] = np.zeros(shape=3)
#       val: npt.NDArray[np.float64] = np.zeros(shape=3)
        c: int
#       c: int
        for c in range(3):
#       for c in range(3):
            # Using 180 samples for speed, 360 is better but slower
#           # Using 180 samples for speed, 360 is better but slower
            val[c] = integrate_skin(sigma_s=sigma_s[c], sigma_a=sigma_a[c], g=g, eta=eta, r_curvature=r_curvature, ndotl=ndotl, num_samples=180)
#           val[c] = integrate_skin(sigma_s=sigma_s[c], sigma_a=sigma_a[c], g=g, eta=eta, r_curvature=r_curvature, ndotl=ndotl, num_samples=180)

        row_data[x] = val[::-1] # BGR for OpenCV
#       row_data[x] = val[::-1] # BGR for OpenCV
    return y, row_data
#   return y, row_data

def bake_lut(profile_name: str) -> None:
    p: Dict[str, Any] = PROFILES[profile_name]
#   p: Dict[str, Any] = PROFILES[profile_name]
    sigma_s: List[float] = p["sigma_s"]
#   sigma_s: List[float] = p["sigma_s"]
    sigma_a: List[float] = p["sigma_a"]
#   sigma_a: List[float] = p["sigma_a"]
    g: float = p["g"]
#   g: float = p["g"]
    eta: float = p["eta"]
#   eta: float = p["eta"]
    output_filename: str = p["output"]
#   output_filename: str = p["output"]

    # Resolution of the 2D Pre-Integrated LUT
#   # Resolution of the 2D Pre-Integrated LUT
    res_x: int = 512 # N.L mapping: 0 to 1 -> -1 to 1
#   res_x: int = 512 # N.L mapping: 0 to 1 -> -1 to 1
    res_y: int = 512 # 1/r (curvature)
#   res_y: int = 512 # 1/r (curvature)

    out_img: npt.NDArray[np.float32] = np.zeros(shape=(res_y, res_x, 3), dtype=np.float32)
#   out_img: npt.NDArray[np.float32] = np.zeros(shape=(res_y, res_x, 3), dtype=np.float32)

    print(f"Baking 2D Skin LUT for profile '{profile_name}'...")
#   print(f"Baking 2D Skin LUT for profile '{profile_name}'...")

    # We map y axis (v) to Curvature.
#   # We map y axis (v) to Curvature.
    # Curvature C goes from 0.0 (flat) to max_curvature.
#   # Curvature C goes from 0.0 (flat) to max_curvature.
    # A standard max curvature could be 1.0 mm^-1 (radius = 1mm).
#   # A standard max curvature could be 1.0 mm^-1 (radius = 1mm).
    max_curvature: float = 1.0
#   max_curvature: float = 1.0

    max_workers: int = multiprocessing.cpu_count()
#   max_workers: int = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#   with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        y: int
#       y: int
        futures: List[concurrent.futures.Future[Any]] = [executor.submit(compute_row, y, res_x, res_y, max_curvature, sigma_s, sigma_a, g, eta) for y in range(res_y)]
#       futures: List[concurrent.futures.Future[Any]] = [executor.submit(compute_row, y, res_x, res_y, max_curvature, sigma_s, sigma_a, g, eta) for y in range(res_y)]

        i: int
#       i: int
        future: concurrent.futures.Future[Any]
#       future: concurrent.futures.Future[Any]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
#       for i, future in enumerate(concurrent.futures.as_completed(futures)):
            row_data: npt.NDArray[np.float32]
#           row_data: npt.NDArray[np.float32]
            y, row_data = future.result()
#           y, row_data = future.result()
            out_img[y] = row_data
#           out_img[y] = row_data
            sys.stdout.write(f"\rProcessing row {i + 1}/{res_y}")
#           sys.stdout.write(f"\rProcessing row {i + 1}/{res_y}")
            sys.stdout.flush()
#           sys.stdout.flush()

    print(f"\nSaving to {output_filename}...")
#   print(f"\nSaving to {output_filename}...")
    cv2.imwrite(filename=output_filename, img=out_img)
#   cv2.imwrite(filename=output_filename, img=out_img)
    print("Done!")
#   print("Done!")

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Bake Skin Pre-Integrated LUT")
#   parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Bake Skin Pre-Integrated LUT")
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
