# Image Panorama Stitching

This project implements image panorama stitching using SIFT (Scale-Invariant Feature Transform) keypoints, homography estimation, and Laplacian pyramid blending. The program processes a set of input images, aligns them using computed transformations, and blends them to create a seamless panorama.

---

## Features

- **SIFT Keypoint Detection**: Detects and matches keypoints between overlapping images.
- **Homography Estimation**: Computes homography matrices using RANSAC or a custom implementation to align images.
- **Image Warping**: Transforms images using computed homographies.
- **Laplacian Pyramid Blending**: Blends overlapping regions for seamless transitions.
- **Support for Multiple Image Sets**: Processes predefined scenes with varying numbers of images.

---

## Requirements

To run the project, you need the following dependencies:

- Python 3.7+
- OpenCV
- NumPy

You can install the dependencies using pip:

```bash
pip install opencv-python numpy
```

---

## Directory Structure

The project expects the following directory structure:

```
Dataset/
├── scene1/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
├── scene2/
│   ├── ...
├── outputs/
│   ├── scene1/
│   │   ├── self/
│   │   ├── inbuilt/
│   ├── scene2/
```

- **`sceneX/`**: Contains input images for stitching.
- **`outputs/`**: Contains the resulting panoramas in separate subfolders:
  - **`self/`**: Results using the custom homography implementation.
  - **`inbuilt/`**: Results using OpenCV's `cv2.findHomography`.

---

## Usage

1. **Prepare Input Images**:
   - Place your images in a subdirectory under `Dataset/` (e.g., `Dataset/scene1/`).

2. **Run the Script**:
   - Update the `imageSet` variable in the script to match the scene number (e.g., `imageSet = 1`).
   - Execute the script:
     ```bash
     python panorama_stitching.py
     ```

3. **View Outputs**:
   - The resulting warped and blended images will be saved in `Dataset/outputs/sceneX/self/` and `Dataset/outputs/sceneX/inbuilt/`.

---

## Key Functions

### `SIFT(img1, img2, nFeaturesReturn=30)`
Detects SIFT keypoints, computes descriptors, and matches keypoints between two images.

### `Homography(matches)`
Estimates the homography matrix using a custom RANSAC-like approach.

### `transformation(img, H, dst, offset=(0, 0))`
Applies the computed homography to warp an image.

### `blend_images(img1, img2, depth=6)`
Blends two images using Laplacian pyramid blending for smooth transitions.

---

## Example Output

After running the script, the final panorama will be saved as `FINALBLENDED.png` in the corresponding output folder.

---

## Notes

- Ensure all input images have sufficient overlap for accurate keypoint matching.
- Modify the `shape`, `offset`, and `threshold` variables as needed to suit your dataset.

---

## Future Improvements

- Support for dynamic offset calculation based on input image dimensions.
- Parallel processing for faster keypoint detection and homography computation.
- Enhanced blending for better handling of exposure differences.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
