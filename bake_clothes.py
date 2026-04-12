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
from typing import Dict, Any, List
from typing import Dict, Any, List

# PBRT-v3 Disney Cloth Models
# PBRT-v3 Disney Cloth Models
# Based on pbrt-v3/src/materials/disney.cpp
# Based on pbrt-v3/src/materials/disney.cpp

def SchlickWeight(cosTheta: float) -> float:
    m: float = max(0.0, min(1.0, 1.0 - cosTheta))
#   m: float = max(0.0, min(1.0, 1.0 - cosTheta))
    return (m * m) * (m * m) * m
#   return (m * m) * (m * m) * m

def Lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)
#   return a + t * (b - a)

def evaluate_disney_cloth(ndotl: float, ndotv: float, ldotv: float, roughness: float, sheenWeight: float, subsurfaceWeight: float) -> npt.NDArray[np.float64]:
    if ndotl <= 0.0 or ndotv <= 0.0:
#   if ndotl <= 0.0 or ndotv <= 0.0:
        return np.zeros(shape=3)
#       return np.zeros(shape=3)

    # cosThetaD is L.H
#   # cosThetaD is L.H
    # H = (L + V) / |L + V|
#   # H = (L + V) / |L + V|
    # |L + V| = sqrt(2 + 2 * L.V)
#   # |L + V| = sqrt(2 + 2 * L.V)
    # L.H = (1 + L.V) / sqrt(2 + 2 * L.V)
#   # L.H = (1 + L.V) / sqrt(2 + 2 * L.V)
    cosThetaD: float
#   cosThetaD: float
    if ldotv >= 0.9999:
#   if ldotv >= 0.9999:
        cosThetaD = 1.0
#       cosThetaD = 1.0
    elif ldotv <= -0.9999:
#   elif ldotv <= -0.9999:
        cosThetaD = 0.0
#       cosThetaD = 0.0
    else:
#   else:
        cosThetaD = math.sqrt(max(0.0, 1.0 + ldotv)) / math.sqrt(2.0)
#       cosThetaD = math.sqrt(max(0.0, 1.0 + ldotv)) / math.sqrt(2.0)

    Fo: float = SchlickWeight(cosTheta=ndotv)
#   Fo: float = SchlickWeight(cosTheta=ndotv)
    Fi: float = SchlickWeight(cosTheta=ndotl)
#   Fi: float = SchlickWeight(cosTheta=ndotl)

    # 1. Disney Diffuse
#   # 1. Disney Diffuse
    diffuse: float = (1.0 / math.pi) * (1.0 - Fo / 2.0) * (1.0 - Fi / 2.0)
#   diffuse: float = (1.0 / math.pi) * (1.0 - Fo / 2.0) * (1.0 - Fi / 2.0)

    # 2. Disney Retro
#   # 2. Disney Retro
    Rr: float = 2.0 * roughness * cosThetaD * cosThetaD
#   Rr: float = 2.0 * roughness * cosThetaD * cosThetaD
    retro: float = (1.0 / math.pi) * Rr * (Fo + Fi + Fo * Fi * (Rr - 1.0))
#   retro: float = (1.0 / math.pi) * Rr * (Fo + Fi + Fo * Fi * (Rr - 1.0))

    # 3. Disney Fake Subsurface (Hanrahan-Krueger approx)
#   # 3. Disney Fake Subsurface (Hanrahan-Krueger approx)
    Fss90: float = cosThetaD * cosThetaD * roughness
#   Fss90: float = cosThetaD * cosThetaD * roughness
    Fss: float = Lerp(a=Fo, b=1.0, t=Fss90) * Lerp(a=Fi, b=1.0, t=Fss90)
#   Fss: float = Lerp(a=Fo, b=1.0, t=Fss90) * Lerp(a=Fi, b=1.0, t=Fss90)
    ss: float = 1.25 * (Fss * (1.0 / max(1e-4, ndotl + ndotv) - 0.5) + 0.5)
#   ss: float = 1.25 * (Fss * (1.0 / max(1e-4, ndotl + ndotv) - 0.5) + 0.5)
    fakess: float = (1.0 / math.pi) * ss
#   fakess: float = (1.0 / math.pi) * ss

    # 4. Disney Sheen
#   # 4. Disney Sheen
    sheen: float = SchlickWeight(cosTheta=cosThetaD)
#   sheen: float = SchlickWeight(cosTheta=cosThetaD)

    # Combine base lobes (Diffuse, Retro, Fake Subsurface) into the R channel
#   # Combine base lobes (Diffuse, Retro, Fake Subsurface) into the R channel
    # Combine Sheen into the G channel
#   # Combine Sheen into the G channel
    # This allows the shader to multiply the R channel by BaseColor and the G channel by SheenColor
#   # This allows the shader to multiply the R channel by BaseColor and the G channel by SheenColor
    val: float = 0.0
#   val: float = 0.0
    if subsurfaceWeight < 1.0:
#   if subsurfaceWeight < 1.0:
        val += (1.0 - subsurfaceWeight) * diffuse
#       val += (1.0 - subsurfaceWeight) * diffuse
    if subsurfaceWeight > 0.0:
#   if subsurfaceWeight > 0.0:
        val += subsurfaceWeight * fakess
#       val += subsurfaceWeight * fakess
    if roughness > 0.0:
#   if roughness > 0.0:
        val += retro
#       val += retro

    return np.array(object=[val, sheenWeight * sheen, 0.0])
#   return np.array(object=[val, sheenWeight * sheen, 0.0])

PROFILES: Dict[str, Dict[str, Any]] = {
    "cotton": {"roughness": 0.8, "sheen": 0.2, "subsurface": 0.0, "output": "clothes_bsdf_cotton.exr"},
#   "cotton": {"roughness": 0.8, "sheen": 0.2, "subsurface": 0.0, "output": "clothes_bsdf_cotton.exr"},
    "silk": {"roughness": 0.2, "sheen": 0.8, "subsurface": 0.0, "output": "clothes_bsdf_silk.exr"},
#   "silk": {"roughness": 0.2, "sheen": 0.8, "subsurface": 0.0, "output": "clothes_bsdf_silk.exr"},
    "velvet": {"roughness": 1.0, "sheen": 1.0, "subsurface": 0.2, "output": "clothes_bsdf_velvet.exr"},
#   "velvet": {"roughness": 1.0, "sheen": 1.0, "subsurface": 0.2, "output": "clothes_bsdf_velvet.exr"},
    "denim": {"roughness": 0.9, "sheen": 0.1, "subsurface": 0.0, "output": "clothes_bsdf_denim.exr"},
#   "denim": {"roughness": 0.9, "sheen": 0.1, "subsurface": 0.0, "output": "clothes_bsdf_denim.exr"},
}

def bake_lut(profile_name: str) -> None:
    p: Dict[str, Any] = PROFILES[profile_name]
#   p: Dict[str, Any] = PROFILES[profile_name]
    roughness: float = p["roughness"]
#   roughness: float = p["roughness"]
    sheenWeight: float = p["sheen"]
#   sheenWeight: float = p["sheen"]
    subsurfaceWeight: float = p["subsurface"]
#   subsurfaceWeight: float = p["subsurface"]
    output_filename: str = p["output"]
#   output_filename: str = p["output"]

    res_u: int = 64 # N.L
#   res_u: int = 64 # N.L
    res_v: int = 64 # N.V
#   res_v: int = 64 # N.V
    res_w: int = 64 # L.V
#   res_w: int = 64 # L.V

    grid_size: int = 8 # 8x8 slices = 64
#   grid_size: int = 8 # 8x8 slices = 64
    out_img: npt.NDArray[np.float32] = np.zeros(shape=(res_v * grid_size, res_u * grid_size, 3), dtype=np.float32)
#   out_img: npt.NDArray[np.float32] = np.zeros(shape=(res_v * grid_size, res_u * grid_size, 3), dtype=np.float32)

    print(f"Baking {res_u}x{res_v}x{res_w} Clothes LUT for profile '{profile_name}'...")
#   print(f"Baking {res_u}x{res_v}x{res_w} Clothes LUT for profile '{profile_name}'...")

    slice_idx: int
#   slice_idx: int
    for slice_idx in range(res_w):
#   for slice_idx in range(res_w):
        w: float = slice_idx / (res_w - 1.0)
#       w: float = slice_idx / (res_w - 1.0)
        ldotv: float = w * 2.0 - 1.0 # Map 0..1 to -1..1
#       ldotv: float = w * 2.0 - 1.0 # Map 0..1 to -1..1

        grid_x: int = slice_idx % grid_size
#       grid_x: int = slice_idx % grid_size
        grid_y: int = slice_idx // grid_size
#       grid_y: int = slice_idx // grid_size
        start_x: int = grid_x * res_u
#       start_x: int = grid_x * res_u
        start_y: int = grid_y * res_v
#       start_y: int = grid_y * res_v

        sys.stdout.write(f"\rProcessing slice {slice_idx + 1}/{res_w}")
#       sys.stdout.write(f"\rProcessing slice {slice_idx + 1}/{res_w}")
        sys.stdout.flush()
#       sys.stdout.flush()

        y: int
#       y: int
        for y in range(res_v):
#       for y in range(res_v):
            v_coord: float = y / (res_v - 1.0)
#           v_coord: float = y / (res_v - 1.0)
            ndotv: float = v_coord # 0..1
#           ndotv: float = v_coord # 0..1

            x: int
#           x: int
            for x in range(res_u):
#           for x in range(res_u):
                u_coord: float = x / (res_u - 1.0)
#               u_coord: float = x / (res_u - 1.0)
                ndotl: float = u_coord # 0..1
#               ndotl: float = u_coord # 0..1

                # Check if N.L, N.V, L.V forms a valid spherical triangle
#               # Check if N.L, N.V, L.V forms a valid spherical triangle
                # The bounds of L.V given N.L and N.V are:
#               # The bounds of L.V given N.L and N.V are:
                # L.V = N.L * N.V + sin(theta_L) * sin(theta_V) * cos(phi)
#               # L.V = N.L * N.V + sin(theta_L) * sin(theta_V) * cos(phi)
                sinThetaL: float = math.sqrt(max(0.0, 1.0 - ndotl * ndotl))
#               sinThetaL: float = math.sqrt(max(0.0, 1.0 - ndotl * ndotl))
                sinThetaV: float = math.sqrt(max(0.0, 1.0 - ndotv * ndotv))
#               sinThetaV: float = math.sqrt(max(0.0, 1.0 - ndotv * ndotv))
                min_ldotv: float = ndotl * ndotv - sinThetaL * sinThetaV
#               min_ldotv: float = ndotl * ndotv - sinThetaL * sinThetaV
                max_ldotv: float = ndotl * ndotv + sinThetaL * sinThetaV
#               max_ldotv: float = ndotl * ndotv + sinThetaL * sinThetaV

                valid_ldotv: float = max(min_ldotv, min(max_ldotv, ldotv))
#               valid_ldotv: float = max(min_ldotv, min(max_ldotv, ldotv))

                val: npt.NDArray[np.float64] = evaluate_disney_cloth(ndotl=ndotl, ndotv=ndotv, ldotv=valid_ldotv, roughness=roughness, sheenWeight=sheenWeight, subsurfaceWeight=subsurfaceWeight)
#               val: npt.NDArray[np.float64] = evaluate_disney_cloth(ndotl=ndotl, ndotv=ndotv, ldotv=valid_ldotv, roughness=roughness, sheenWeight=sheenWeight, subsurfaceWeight=subsurfaceWeight)
                out_img[start_y + y, start_x + x] = val[::-1] # OpenCV expects BGR
#               out_img[start_y + y, start_x + x] = val[::-1] # OpenCV expects BGR

    print(f"\nSaving to {output_filename}...")
#   print(f"\nSaving to {output_filename}...")
    cv2.imwrite(filename=output_filename, img=out_img)
#   cv2.imwrite(filename=output_filename, img=out_img)
    print("Done!")
#   print("Done!")

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Bake Clothes BSDF LUT")
#   parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Bake Clothes BSDF LUT")
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
