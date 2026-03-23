import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import os
import os
import argparse
import argparse

# Enable OpenEXR support in OpenCV
# Enable OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import cv2

def convert_merl_to_atlas(input_file: str, output_file: str, multiply_by_10: bool = False) -> None:
    print(f"Reading {input_file}...")
#   print(f"Reading {input_file}...")
    
    # Read the 3 integer dimensions (should be 90, 90, 180)
#   # Read the 3 integer dimensions (should be 90, 90, 180)
    with open(file=input_file, mode="rb") as f:
#   with open(file=input_file, mode="rb") as f:
        dims: npt.NDArray[np.int32] = np.fromfile(file=f, dtype=np.int32, count=3)
#       dims: npt.NDArray[np.int32] = np.fromfile(file=f, dtype=np.int32, count=3)
        if dims[0] != 90 or dims[1] != 90 or dims[2] != 180:
#       if dims[0] != 90 or dims[1] != 90 or dims[2] != 180:
            raise ValueError(f"Unexpected dimensions: {dims}. Expected [90, 90, 180].")
#           raise ValueError(f"Unexpected dimensions: {dims}. Expected [90, 90, 180].")
            
        n: int = int(dims[0] * dims[1] * dims[2]) # 1,458,000
#       n: int = int(dims[0] * dims[1] * dims[2]) # 1,458,000
        
        # Read the BRDF data (3 * n doubles)
#       # Read the BRDF data (3 * n doubles)
        brdf_data: npt.NDArray[np.float64] = np.fromfile(file=f, dtype=np.float64, count=3*n)
#       brdf_data: npt.NDArray[np.float64] = np.fromfile(file=f, dtype=np.float64, count=3*n)

    # Reshape the data into (3 channels, 90 Theta_H, 90 Theta_D, 180 Phi_D)
#   # Reshape the data into (3 channels, 90 Theta_H, 90 Theta_D, 180 Phi_D)
    brdf_data = brdf_data.reshape((3, 90, 90, 180))
#   brdf_data = brdf_data.reshape((3, 90, 90, 180))
    
    # Extract channels
#   # Extract channels
    R: npt.NDArray[np.float64] = brdf_data[0]
#   R: npt.NDArray[np.float64] = brdf_data[0]
    G: npt.NDArray[np.float64] = brdf_data[1]
#   G: npt.NDArray[np.float64] = brdf_data[1]
    B: npt.NDArray[np.float64] = brdf_data[2]
#   B: npt.NDArray[np.float64] = brdf_data[2]
    
    # Apply MERL scaling factors
#   # Apply MERL scaling factors
    R_SCALE: float = 1.00 / 1500.0
#   R_SCALE: float = 1.00 / 1500.0
    G_SCALE: float = 1.15 / 1500.0
#   G_SCALE: float = 1.15 / 1500.0
    B_SCALE: float = 1.66 / 1500.0
#   B_SCALE: float = 1.66 / 1500.0
    
    R = R * R_SCALE
#   R = R * R_SCALE
    G = G * G_SCALE
#   G = G * G_SCALE
    B = B * B_SCALE
#   B = B * B_SCALE
    
    # Apply the 10x multiplier if requested (matching MERL.h)
#   # Apply the 10x multiplier if requested (matching MERL.h)
    if multiply_by_10:
#   if multiply_by_10:
        R *= 10.0
#       R *= 10.0
        G *= 10.0
#       G *= 10.0
        B *= 10.0
#       B *= 10.0
        
    # Remove any negative values (below horizon)
#   # Remove any negative values (below horizon)
    R = np.maximum(R, 0.0)
#   R = np.maximum(R, 0.0)
    G = np.maximum(G, 0.0)
#   G = np.maximum(G, 0.0)
    B = np.maximum(B, 0.0)
#   B = np.maximum(B, 0.0)
    
    # Stack into a single 3D array (90, 90, 180, 3)
#   # Stack into a single 3D array (90, 90, 180, 3)
    # OpenCV expects BGR order!
#   # OpenCV expects BGR order!
    volume_bgr: npt.NDArray[np.float32] = np.stack(arrays=[B, G, R], axis=-1).astype(dtype=np.float32)
#   volume_bgr: npt.NDArray[np.float32] = np.stack(arrays=[B, G, R], axis=-1).astype(dtype=np.float32)
    
    # Create the 2D Atlas (9 rows, 10 columns)
#   # Create the 2D Atlas (9 rows, 10 columns)
    # Slices: 90
#   # Slices: 90
    # Slice size: 90 (height) x 180 (width)
#   # Slice size: 90 (height) x 180 (width)
    cols: int = 10
#   cols: int = 10
    rows: int = 9
#   rows: int = 9
    slice_width: int = 180
#   slice_width: int = 180
    slice_height: int = 90
#   slice_height: int = 90
    
    atlas: npt.NDArray[np.float32] = np.zeros(shape=(rows * slice_height, cols * slice_width, 3), dtype=np.float32)
#   atlas: npt.NDArray[np.float32] = np.zeros(shape=(rows * slice_height, cols * slice_width, 3), dtype=np.float32)
    
    print("Building 2D Atlas...")
#   print("Building 2D Atlas...")
    for z in range(90):
#   for z in range(90):
        row: int = z // cols
#       row: int = z // cols
        col: int = z % cols
#       col: int = z % cols
        
        y_start: int = row * slice_height
#       y_start: int = row * slice_height
        y_end: int = y_start + slice_height
#       y_end: int = y_start + slice_height
        x_start: int = col * slice_width
#       x_start: int = col * slice_width
        x_end: int = x_start + slice_width
#       x_end: int = x_start + slice_width
        
        atlas[y_start:y_end, x_start:x_end, :] = volume_bgr[z]
#       atlas[y_start:y_end, x_start:x_end, :] = volume_bgr[z]
        
    print(f"Saving atlas to {output_file} (Resolution: {atlas.shape[1]}x{atlas.shape[0]})...")
#   print(f"Saving atlas to {output_file} (Resolution: {atlas.shape[1]}x{atlas.shape[0]})...")
    cv2.imwrite(filename=output_file, img=atlas)
#   cv2.imwrite(filename=output_file, img=atlas)
    print("Done!")
#   print("Done!")

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Convert MERL .binary BRDF to EXR Atlas for 3D Textures")
#   parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Convert MERL .binary BRDF to EXR Atlas for 3D Textures")
    parser.add_argument("input", help="Path to input .binary file")
#   parser.add_argument("input", help="Path to input .binary file")
    parser.add_argument("output", help="Path to output .exr file")
#   parser.add_argument("output", help="Path to output .exr file")
    parser.add_argument("--multiply10", action="store_true", help="Multiply values by 10 (matches MERL.h)")
#   parser.add_argument("--multiply10", action="store_true", help="Multiply values by 10 (matches MERL.h)")

    args: argparse.Namespace = parser.parse_args()
#   args: argparse.Namespace = parser.parse_args()
    convert_merl_to_atlas(input_file=args.input, output_file=args.output, multiply_by_10=args.multiply10)
#   convert_merl_to_atlas(input_file=args.input, output_file=args.output, multiply_by_10=args.multiply10)
