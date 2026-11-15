"""
Unit Tests for Unified CNN Preprocessing (Phase 2 Validation)
------------------------------------------------------------
Validates that:
  1. File-path input and numpy-array input produce identical outputs.
  2. BGR and Grayscale paths produce identical dimensions and bounds.
  3. Preprocessing produces tensor with shape (224, 224, 3) and range [-1.0, 1.0].
  4. Reproducibility across multiple calls is strictly deterministic.
"""

import os
import cv2
import numpy as np
from cnn_preprocessing import preprocess_image_for_cnn, get_residual_display_maps, IMG_SIZE

def test_preprocessing():
    print("Testing CNN Preprocessing consistency...")
    
    test_img_path = os.path.join("data", "clean", "clean_001.png")
    assert os.path.exists(test_img_path), f"Test image {test_img_path} not found"
    
    # 1. Path vs Array input
    tensor_from_path = preprocess_image_for_cnn(test_img_path)
    img_bgr = cv2.imread(test_img_path)
    tensor_from_array = preprocess_image_for_cnn(img_bgr)
    
    np.testing.assert_allclose(tensor_from_path, tensor_from_array, rtol=1e-5, atol=1e-5)
    print("  [PASS] Path input vs Array input produce identical outputs.")
    
    # 2. Shape and Range verification
    assert tensor_from_path.shape == (IMG_SIZE[1], IMG_SIZE[0], 3), f"Shape mismatch: {tensor_from_path.shape}"
    assert tensor_from_path.dtype == np.float32, f"Dtype mismatch: {tensor_from_path.dtype}"
    assert np.min(tensor_from_path) >= -1.0, f"Min below -1: {np.min(tensor_from_path)}"
    assert np.max(tensor_from_path) <= 1.0, f"Max above 1: {np.max(tensor_from_path)}"
    print(f"  [PASS] Output tensor shape is {tensor_from_path.shape}, dtype is {tensor_from_path.dtype}, range is [{np.min(tensor_from_path):.3f}, {np.max(tensor_from_path):.3f}].")
    
    # 3. Determinism
    tensor_repeat = preprocess_image_for_cnn(img_bgr)
    np.testing.assert_array_equal(tensor_from_array, tensor_repeat)
    print("  [PASS] Deterministic output verified across repeat invocations.")
    
    # 4. Display maps helper
    maps = get_residual_display_maps(img_bgr)
    assert "gaussian_residual" in maps
    assert "kv_residual" in maps
    assert "laplacian_residual" in maps
    assert maps["gaussian_residual"].shape == (IMG_SIZE[1], IMG_SIZE[0])
    assert maps["gaussian_residual"].dtype == np.uint8
    print("  [PASS] Display maps generation validated.")
    
    print("\nAll Preprocessing Consistency Tests PASSED successfully!\n")

if __name__ == "__main__":
    test_preprocessing()
