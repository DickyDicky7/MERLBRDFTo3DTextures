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
import multiprocessing
import multiprocessing
import concurrent.futures
import concurrent.futures
from typing import Dict, Any, List, Tuple
from typing import Dict, Any, List, Tuple

# --- Core Math and PBRT Port ---
# --- Core Math and PBRT Port ---

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

def GGX_D(m_z: float, alpha: float) -> float:
    if m_z <= 0.0:
#   if m_z <= 0.0:
        return 0.0
#       return 0.0
    alpha2: float = alpha * alpha
#   alpha2: float = alpha * alpha
    denom: float = m_z * m_z * (alpha2 - 1.0) + 1.0
#   denom: float = m_z * m_z * (alpha2 - 1.0) + 1.0
    return alpha2 / (math.pi * denom * denom)
#   return alpha2 / (math.pi * denom * denom)

def GGX_G1(v_z: float, alpha: float) -> float:
    if v_z <= 0.0:
#   if v_z <= 0.0:
        return 0.0
#       return 0.0
    alpha2: float = alpha * alpha
#   alpha2: float = alpha * alpha
    return 2.0 * v_z / (v_z + math.sqrt(alpha2 + (1.0 - alpha2) * v_z * v_z))
#   return 2.0 * v_z / (v_z + math.sqrt(alpha2 + (1.0 - alpha2) * v_z * v_z))

def SampleGGX(u1: float, u2: float, alpha: float) -> Tuple[float, float, float]:
    theta: float = math.atan(alpha * math.sqrt(u1) / math.sqrt(1.0 - u1))
#   theta: float = math.atan(alpha * math.sqrt(u1) / math.sqrt(1.0 - u1))
    phi: float = 2.0 * math.pi * u2
#   phi: float = 2.0 * math.pi * u2
    m_x: float = math.sin(theta) * math.cos(phi)
#   m_x: float = math.sin(theta) * math.cos(phi)
    m_y: float = math.sin(theta) * math.sin(phi)
#   m_y: float = math.sin(theta) * math.sin(phi)
    m_z: float = math.cos(theta)
#   m_z: float = math.cos(theta)
    return m_x, m_y, m_z
#   return m_x, m_y, m_z

# --- Monte Carlo Integration for LUTs ---
# --- Monte Carlo Integration for LUTs ---

def integrate_energy_and_transmission(ndotv: float, alpha: float, eta: float, num_samples: int = 1024) -> Tuple[float, float, float, float]:
    # v is the view direction
#   # v is the view direction
    v_z: float = ndotv
#   v_z: float = ndotv
    v_x: float = math.sqrt(max(0.0, 1.0 - v_z * v_z))
#   v_x: float = math.sqrt(max(0.0, 1.0 - v_z * v_z))
    v_y: float = 0.0
#   v_y: float = 0.0

    sum_Er: float = 0.0
#   sum_Er: float = 0.0
    sum_Et: float = 0.0
#   sum_Et: float = 0.0
    sum_E_split_sum: float = 0.0
#   sum_E_split_sum: float = 0.0

    # For importance sampling, we sample m from the GGX distribution
#   # For importance sampling, we sample m from the GGX distribution
    # and compute the reflected/transmitted directions.
#   # and compute the reflected/transmitted directions.

    # We use a Halton sequence for low-discrepancy sampling (simplified as random here, but fixed seed for stability)
#   # We use a Halton sequence for low-discrepancy sampling (simplified as random here, but fixed seed for stability)
    np.random.seed(seed=int(ndotv * 1000 + alpha * 10000))
#   np.random.seed(seed=int(ndotv * 1000 + alpha * 10000))

    valid_samples: int = 0
#   valid_samples: int = 0
    i: int
#   i: int
    for i in range(num_samples):
#   for i in range(num_samples):
        u1: float = float(np.random.rand())
#       u1: float = float(np.random.rand())
        u2: float = float(np.random.rand())
#       u2: float = float(np.random.rand())

        m_x: float
#       m_x: float
        m_y: float
#       m_y: float
        m_z: float
#       m_z: float
        m_x, m_y, m_z = SampleGGX(u1=u1, u2=u2, alpha=alpha)
#       m_x, m_y, m_z = SampleGGX(u1=u1, u2=u2, alpha=alpha)

        # View dot macro-normal
#       # View dot macro-normal
        v_dot_m: float = v_x * m_x + v_y * m_y + v_z * m_z
#       v_dot_m: float = v_x * m_x + v_y * m_y + v_z * m_z
        if v_dot_m <= 0.0:
#       if v_dot_m <= 0.0:
            continue
#           continue

        # Reflected direction
#       # Reflected direction
        l_r_x: float = 2.0 * v_dot_m * m_x - v_x
#       l_r_x: float = 2.0 * v_dot_m * m_x - v_x
        l_r_y: float = 2.0 * v_dot_m * m_y - v_y
#       l_r_y: float = 2.0 * v_dot_m * m_y - v_y
        l_r_z: float = 2.0 * v_dot_m * m_z - v_z
#       l_r_z: float = 2.0 * v_dot_m * m_z - v_z

        F: float = FrDielectric(cosThetaI=v_dot_m, etaI=1.0, etaT=eta)
#       F: float = FrDielectric(cosThetaI=v_dot_m, etaI=1.0, etaT=eta)
        G: float = GGX_G1(v_z=v_z, alpha=alpha) * GGX_G1(v_z=max(0.001, l_r_z), alpha=alpha)
#       G: float = GGX_G1(v_z=v_z, alpha=alpha) * GGX_G1(v_z=max(0.001, l_r_z), alpha=alpha)

        # Weight for reflection importance sampling: F * G * v_dot_m / (v_z * m_z)
#       # Weight for reflection importance sampling: F * G * v_dot_m / (v_z * m_z)
        if l_r_z > 0.0 and m_z > 0.0:
#       if l_r_z > 0.0 and m_z > 0.0:
            weight_r: float = F * G * v_dot_m / (v_z * m_z + 0.0001)
#           weight_r: float = F * G * v_dot_m / (v_z * m_z + 0.0001)
            sum_Er += weight_r
#           sum_Er += weight_r

        # Transmitted direction (Snell's law)
#       # Transmitted direction (Snell's law)
        # Using relative IOR eta_r = eta_i / eta_t
#       # Using relative IOR eta_r = eta_i / eta_t
        eta_r: float = 1.0 / eta
#       eta_r: float = 1.0 / eta
        c: float = v_dot_m
#       c: float = v_dot_m
        cs2: float = 1.0 - eta_r * eta_r * (1.0 - c * c)
#       cs2: float = 1.0 - eta_r * eta_r * (1.0 - c * c)

        if cs2 >= 0.0:
#       if cs2 >= 0.0:
            # Not total internal reflection
#           # Not total internal reflection
            l_t_x: float = (eta_r * c - math.sqrt(cs2)) * m_x - eta_r * v_x
#           l_t_x: float = (eta_r * c - math.sqrt(cs2)) * m_x - eta_r * v_x
            l_t_y: float = (eta_r * c - math.sqrt(cs2)) * m_y - eta_r * v_y
#           l_t_y: float = (eta_r * c - math.sqrt(cs2)) * m_y - eta_r * v_y
            l_t_z: float = (eta_r * c - math.sqrt(cs2)) * m_z - eta_r * v_z
#           l_t_z: float = (eta_r * c - math.sqrt(cs2)) * m_z - eta_r * v_z

            l_t_z = -l_t_z # point downwards
#           l_t_z = -l_t_z # point downwards
            if l_t_z > 0.0 and m_z > 0.0:
#           if l_t_z > 0.0 and m_z > 0.0:
                l_dot_m: float = - (l_t_x * m_x + l_t_y * m_y - l_t_z * m_z) # l is pointing outwards from surface for dot product
#               l_dot_m: float = - (l_t_x * m_x + l_t_y * m_y - l_t_z * m_z) # l is pointing outwards from surface for dot product
                if l_dot_m > 0.0:
#               if l_dot_m > 0.0:
                    G_t: float = GGX_G1(v_z=v_z, alpha=alpha) * GGX_G1(v_z=l_t_z, alpha=alpha)
#                   G_t: float = GGX_G1(v_z=v_z, alpha=alpha) * GGX_G1(v_z=l_t_z, alpha=alpha)

                    # Weight for transmission importance sampling
#                   # Weight for transmission importance sampling
                    # From Walter et al. 2007: Microfacet Models for Refraction
#                   # From Walter et al. 2007: Microfacet Models for Refraction
                    # The integral of BTDF * cos(theta_l) over the hemisphere
#                   # The integral of BTDF * cos(theta_l) over the hemisphere
                    weight_t: float = (1.0 - F) * G_t * v_dot_m / (v_z * m_z + 0.0001)
#                   weight_t: float = (1.0 - F) * G_t * v_dot_m / (v_z * m_z + 0.0001)
                    sum_Et += weight_t
#                   sum_Et += weight_t
                    sum_E_split_sum += weight_t * math.pow(1.0 - max(0.0, v_z), 5.0)
#                   sum_E_split_sum += weight_t * math.pow(1.0 - max(0.0, v_z), 5.0)

        valid_samples += 1
#       valid_samples += 1

    if valid_samples > 0:
#   if valid_samples > 0:
        return sum_Er / valid_samples, sum_Et / valid_samples, sum_E_split_sum / valid_samples, 1.0
#       return sum_Er / valid_samples, sum_Et / valid_samples, sum_E_split_sum / valid_samples, 1.0
    return 0.0, 0.0, 0.0, 1.0
#   return 0.0, 0.0, 0.0, 1.0

def compute_energy_row(y: int, res_x: int, res_y: int, eta: float) -> Tuple[int, npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    roughness: float = y / (res_y - 1.0)
#   roughness: float = y / (res_y - 1.0)
    alpha: float = max(0.001, roughness * roughness)
#   alpha: float = max(0.001, roughness * roughness)

    row_energy: npt.NDArray[np.float32] = np.zeros(shape=(res_x, 3), dtype=np.float32)
#   row_energy: npt.NDArray[np.float32] = np.zeros(shape=(res_x, 3), dtype=np.float32)
    row_trans: npt.NDArray[np.float32] = np.zeros(shape=(res_x, 3), dtype=np.float32)
#   row_trans: npt.NDArray[np.float32] = np.zeros(shape=(res_x, 3), dtype=np.float32)

    x: int
#   x: int
    for x in range(res_x):
#   for x in range(res_x):
        ndotv: float = max(0.001, x / (res_x - 1.0))
#       ndotv: float = max(0.001, x / (res_x - 1.0))

        Er: float
#       Er: float
        Et: float
#       Et: float
        E_split_sum: float
#       E_split_sum: float
        _: float
#       _: float
        Er, Et, E_split_sum, _ = integrate_energy_and_transmission(ndotv=ndotv, alpha=alpha, eta=eta, num_samples=1024)
#       Er, Et, E_split_sum, _ = integrate_energy_and_transmission(ndotv=ndotv, alpha=alpha, eta=eta, num_samples=1024)

        # Energy Comp LUT: R = Er (Reflection), G = Et (Transmission), B = Total (Er + Et)
#       # Energy Comp LUT: R = Er (Reflection), G = Et (Transmission), B = Total (Er + Et)
        row_energy[x] = np.array(object=[Er, Et, Er + Et], dtype=np.float32)[::-1] # BGR
#       row_energy[x] = np.array(object=[Er, Et, Er + Et], dtype=np.float32)[::-1] # BGR

        # Transmission LUT: R = Split-sum base, G = Split-sum bias, B = Roughness mapping
#       # Transmission LUT: R = Split-sum base, G = Split-sum bias, B = Roughness mapping
        row_trans[x] = np.array(object=[Et, E_split_sum, roughness], dtype=np.float32)[::-1] # BGR
#       row_trans[x] = np.array(object=[Et, E_split_sum, roughness], dtype=np.float32)[::-1] # BGR

    return y, row_energy, row_trans
#   return y, row_energy, row_trans

def bake_energy_and_transmission(eta: float, output_energy: str, output_trans: str) -> None:
    res_x: int = 128
#   res_x: int = 128
    res_y: int = 128
#   res_y: int = 128

    out_energy: npt.NDArray[np.float32] = np.zeros(shape=(res_y, res_x, 3), dtype=np.float32)
#   out_energy: npt.NDArray[np.float32] = np.zeros(shape=(res_y, res_x, 3), dtype=np.float32)
    out_trans: npt.NDArray[np.float32] = np.zeros(shape=(res_y, res_x, 3), dtype=np.float32)
#   out_trans: npt.NDArray[np.float32] = np.zeros(shape=(res_y, res_x, 3), dtype=np.float32)

    print(f"Baking Energy Compensation and Transmission LUTs (eta={eta})...")
#   print(f"Baking Energy Compensation and Transmission LUTs (eta={eta})...")

    max_workers: int = multiprocessing.cpu_count()
#   max_workers: int = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#   with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        y: int
#       y: int
        futures: List[concurrent.futures.Future[Any]] = [executor.submit(compute_energy_row, y, res_x, res_y, eta) for y in range(res_y)]
#       futures: List[concurrent.futures.Future[Any]] = [executor.submit(compute_energy_row, y, res_x, res_y, eta) for y in range(res_y)]

        i: int
#       i: int
        future: concurrent.futures.Future[Any]
#       future: concurrent.futures.Future[Any]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
#       for i, future in enumerate(concurrent.futures.as_completed(futures)):
            row_energy: npt.NDArray[np.float32]
#           row_energy: npt.NDArray[np.float32]
            row_trans: npt.NDArray[np.float32]
#           row_trans: npt.NDArray[np.float32]
            y, row_energy, row_trans = future.result()
#           y, row_energy, row_trans = future.result()
            out_energy[y] = row_energy
#           out_energy[y] = row_energy
            out_trans[y] = row_trans
#           out_trans[y] = row_trans
            sys.stdout.write(f"\rProcessing row {i + 1}/{res_y}")
#           sys.stdout.write(f"\rProcessing row {i + 1}/{res_y}")
            sys.stdout.flush()
#           sys.stdout.flush()

    print(f"\nSaving {output_energy}...")
#   print(f"\nSaving {output_energy}...")
    cv2.imwrite(filename=output_energy, img=out_energy)
#   cv2.imwrite(filename=output_energy, img=out_energy)
    print(f"Saving {output_trans}...")
#   print(f"Saving {output_trans}...")
    cv2.imwrite(filename=output_trans, img=out_trans)
#   cv2.imwrite(filename=output_trans, img=out_trans)

# --- Volumetric Absorption (Beer's Law) ---
# --- Volumetric Absorption (Beer's Law) ---

def bake_absorption(sigma_a: List[float], max_thickness: float, output_abs: str) -> None:
    res_x: int = 256 # Thickness
#   res_x: int = 256 # Thickness

    out_abs: npt.NDArray[np.float32] = np.zeros(shape=(1, res_x, 3), dtype=np.float32)
#   out_abs: npt.NDArray[np.float32] = np.zeros(shape=(1, res_x, 3), dtype=np.float32)

    print(f"Baking Volumetric Absorption LUT (max_thickness={max_thickness})...")
#   print(f"Baking Volumetric Absorption LUT (max_thickness={max_thickness})...")

    x: int
#   x: int
    for x in range(res_x):
#   for x in range(res_x):
        thickness: float = (x / (res_x - 1.0)) * max_thickness
#       thickness: float = (x / (res_x - 1.0)) * max_thickness

        tr_r: float = math.exp(-sigma_a[0] * thickness)
#       tr_r: float = math.exp(-sigma_a[0] * thickness)
        tr_g: float = math.exp(-sigma_a[1] * thickness)
#       tr_g: float = math.exp(-sigma_a[1] * thickness)
        tr_b: float = math.exp(-sigma_a[2] * thickness)
#       tr_b: float = math.exp(-sigma_a[2] * thickness)

        out_abs[0, x] = np.array(object=[tr_r, tr_g, tr_b], dtype=np.float32)[::-1] # BGR
#       out_abs[0, x] = np.array(object=[tr_r, tr_g, tr_b], dtype=np.float32)[::-1] # BGR

    print(f"Saving {output_abs}...")
#   print(f"Saving {output_abs}...")
    cv2.imwrite(filename=output_abs, img=out_abs)
#   cv2.imwrite(filename=output_abs, img=out_abs)

PROFILES: Dict[str, Dict[str, Any]] = {
    "clear_glass": {"eta": 1.5, "sigma_a": [0.01, 0.01, 0.01], "max_thickness": 10.0, "prefix": "glass_lut_clear"},
#   "clear_glass": {"eta": 1.5, "sigma_a": [0.01, 0.01, 0.01], "max_thickness": 10.0, "prefix": "glass_lut_clear"},
    "tinted_blue": {"eta": 1.5, "sigma_a": [0.8, 0.2, 0.05], "max_thickness": 10.0, "prefix": "glass_lut_blue"},
#   "tinted_blue": {"eta": 1.5, "sigma_a": [0.8, 0.2, 0.05], "max_thickness": 10.0, "prefix": "glass_lut_blue"},
    "tinted_green": {"eta": 1.5, "sigma_a": [0.2, 0.05, 0.8], "max_thickness": 10.0, "prefix": "glass_lut_green"}
#   "tinted_green": {"eta": 1.5, "sigma_a": [0.2, 0.05, 0.8], "max_thickness": 10.0, "prefix": "glass_lut_green"}
}

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Bake Glass BSDF LUTs")
#   parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Bake Glass BSDF LUTs")
    parser.add_argument("--profile", type=str, choices=list(PROFILES.keys()) + ["all"], default="all")
#   parser.add_argument("--profile", type=str, choices=list(PROFILES.keys()) + ["all"], default="all")
    args: argparse.Namespace = parser.parse_args()
#   args: argparse.Namespace = parser.parse_args()

    profiles_to_bake: List[str] = list(PROFILES.keys()) if args.profile == "all" else [args.profile]
#   profiles_to_bake: List[str] = list(PROFILES.keys()) if args.profile == "all" else [args.profile]

    prof_name: str
#   prof_name: str
    for prof_name in profiles_to_bake:
#   for prof_name in profiles_to_bake:
        prof: Dict[str, Any] = PROFILES[prof_name]
#       prof: Dict[str, Any] = PROFILES[prof_name]
        prefix: str = prof["prefix"]
#       prefix: str = prof["prefix"]

        # Output filenames
#       # Output filenames
        out_energy: str = f"{prefix}_energy_comp.exr"
#       out_energy: str = f"{prefix}_energy_comp.exr"
        out_trans: str = f"{prefix}_transmission.exr"
#       out_trans: str = f"{prefix}_transmission.exr"
        out_abs: str = f"{prefix}_absorption.exr"
#       out_abs: str = f"{prefix}_absorption.exr"

        bake_energy_and_transmission(eta=prof["eta"], output_energy=out_energy, output_trans=out_trans)
#       bake_energy_and_transmission(eta=prof["eta"], output_energy=out_energy, output_trans=out_trans)
        bake_absorption(sigma_a=prof["sigma_a"], max_thickness=prof["max_thickness"], output_abs=out_abs)
#       bake_absorption(sigma_a=prof["sigma_a"], max_thickness=prof["max_thickness"], output_abs=out_abs)
        print(f"Profile '{prof_name}' complete.\n")
#       print(f"Profile '{prof_name}' complete.\n")

if __name__ == "__main__":
    main()
#   main()
