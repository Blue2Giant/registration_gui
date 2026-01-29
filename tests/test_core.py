
import unittest
import numpy as np
import cv2
import sys
import os
from pathlib import Path
import tempfile

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.affine_ransac import estimate_affine_3x3_ransac
from app.core.transform_ransac import estimate_transform_3x3_ransac
from app.core.transform_estimator import estimate_transform_3x3
from app.core.visualization import draw_matches_side_by_side, checkerboard_fusion
from app.core.folder_pairs import parse_pairs_txt

class TestCore(unittest.TestCase):
    def test_affine_ransac_identity(self):
        # Create synthetic points: Identity transform
        p1 = np.random.rand(10, 2) * 100
        p2 = p1.copy() # Identity
        
        # Add a clear outlier
        p1 = np.vstack([p1, [0, 0]])
        p2 = np.vstack([p2, [100, 100]]) # Huge shift
        
        res = estimate_affine_3x3_ransac(p1.astype(np.float32), p2.astype(np.float32), thresh_px=3.0)
        
        self.assertTrue(res.inlier_mask[-1] == 0, "Last point should be outlier")
        self.assertTrue(np.allclose(res.H_3x3, np.eye(3), atol=1e-5), "Should recover identity")

    def test_affine_ransac_translation(self):
        # Translation by (10, 20)
        p1 = np.random.rand(20, 2) * 100
        p2 = p1 + np.array([10, 20])
        
        res = estimate_affine_3x3_ransac(p1.astype(np.float32), p2.astype(np.float32), thresh_px=1.0)
        
        expected = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        self.assertTrue(np.allclose(res.H_3x3, expected, atol=0.1))
        self.assertLess(res.rmse, 0.1)

    def test_homography_ransac_identity(self):
        p1 = np.random.rand(30, 2) * 100
        p2 = p1.copy()
        res = estimate_transform_3x3_ransac(p1.astype(np.float32), p2.astype(np.float32), "homography", thresh_px=2.0)
        self.assertTrue(np.allclose(res.H_3x3, np.eye(3), atol=1e-2))

    def test_fsc_affine_identity(self):
        p1 = np.random.rand(30, 2) * 100
        p2 = p1.copy()
        res = estimate_transform_3x3(p1.astype(np.float32), p2.astype(np.float32), "fsc-affine", thresh_px=2.0)
        self.assertTrue(np.allclose(res.H_3x3, np.eye(3), atol=1e-2))

    def test_visualization_smoke(self):
        # Smoke test for visualization functions
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        p1 = np.array([[10, 10], [50, 50]], dtype=np.float32)
        p2 = np.array([[15, 15], [55, 55]], dtype=np.float32)
        mask = np.array([1, 0], dtype=np.uint8)
        
        # Draw matches
        vis = draw_matches_side_by_side(img1, img2, p1, p2, mask)
        self.assertEqual(vis.shape, (100, 200, 3))
        
        # Checkerboard
        H = np.eye(3)
        fusion = checkerboard_fusion(img1, img2, H, tile_px=10)
        self.assertEqual(fusion.shape, (100, 100, 3))

    def test_parse_pairs_txt(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            img1 = td_p / "a.jpg"
            img2 = td_p / "b.png"
            img1.write_bytes(b"fake")
            img2.write_bytes(b"fake")
            txt = td_p / "pairs.txt"
            txt.write_text(f"{img1},{img2}\n", encoding="utf-8")

            pairs = parse_pairs_txt(str(txt))
            self.assertEqual(len(pairs), 1)
            self.assertTrue(pairs[0].fixed_path.endswith("a.jpg"))
            self.assertTrue(pairs[0].moving_path.endswith("b.png"))

if __name__ == '__main__':
    unittest.main()
