import os
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import cv2
import sys
import sys
import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
from typing import Optional, Any
from typing import Optional, Any

def main() -> None:
    if len(sys.argv) < 2:
#   if len(sys.argv) < 2:
        print("Usage: python view_exr.py <path_to_exr>")
#       print("Usage: python view_exr.py <path_to_exr>")
        sys.exit(1)
#       sys.exit(1)

    exr_path: str = sys.argv[1]
#   exr_path: str = sys.argv[1]

    # Read the EXR file. IMREAD_ANYCOLOR and IMREAD_ANYDEPTH are required for HDR/EXR.
#   # Read the EXR file. IMREAD_ANYCOLOR and IMREAD_ANYDEPTH are required for HDR/EXR.
    print(f"Loading {exr_path}...")
#   print(f"Loading {exr_path}...")
    img: Optional[npt.NDArray[Any]] = cv2.imread(filename=exr_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#   img: Optional[npt.NDArray[Any]] = cv2.imread(filename=exr_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    if img is None:
#   if img is None:
        print(f"Error: Could not read {exr_path}. Ensure it is a valid EXR file and OpenCV is installed with EXR support.")
#       print(f"Error: Could not read {exr_path}. Ensure it is a valid EXR file and OpenCV is installed with EXR support.")
        sys.exit(1)
#       sys.exit(1)

    # EXR images are often in linear color space and can have values > 1.0.
#   # EXR images are often in linear color space and can have values > 1.0.
    # We apply a simple Reinhard tone mapping to bring values into the [0, 1] range for display.
#   # We apply a simple Reinhard tone mapping to bring values into the [0, 1] range for display.
    # You can comment this out and use np.clip(img, 0.0, 1.0) if you just want to clip highlights.
#   # You can comment this out and use np.clip(img, 0.0, 1.0) if you just want to clip highlights.
    tonemapped: npt.NDArray[Any] = img / (1.0 + img)
#   tonemapped: npt.NDArray[Any] = img / (1.0 + img)

    # Gamma correction (approximate sRGB) for proper display on standard monitors
#   # Gamma correction (approximate sRGB) for proper display on standard monitors
    gamma: float = 2.2
#   gamma: float = 2.2
    tonemapped = np.power(np.maximum(tonemapped, 0), 1.0 / gamma)
#   tonemapped = np.power(np.maximum(tonemapped, 0), 1.0 / gamma)

    cv2.namedWindow(winname=f"EXR Viewer - {exr_path}", flags=cv2.WINDOW_NORMAL)
#   cv2.namedWindow(winname=f"EXR Viewer - {exr_path}", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname=f"EXR Viewer - {exr_path}", mat=tonemapped)
#   cv2.imshow(winname=f"EXR Viewer - {exr_path}", mat=tonemapped)
    
    print("Press any key in the image window to close...")
#   print("Press any key in the image window to close...")
    cv2.waitKey(delay=0)
#   cv2.waitKey(delay=0)
    cv2.destroyAllWindows()
#   cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#   main()
