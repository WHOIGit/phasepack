import numpy as np
import pytest
from phasepack.tools import lowpassfilter, rayleighmode, perfft2


class TestTools:
    """Test suite for phasepack.tools functions."""
    
    def test_lowpassfilter_basic(self):
        """Test basic lowpassfilter functionality."""
        # Test with square filter
        size = (32, 32)
        cutoff = 0.25
        n = 2
        
        result = lowpassfilter(size, cutoff, n)
        
        assert isinstance(result, np.ndarray), "Should return numpy array"
        assert result.shape == size, f"Shape should be {size}, got {result.shape}"
        assert np.all(result >= 0), "Filter values should be >= 0"
        assert np.all(result <= 1), "Filter values should be <= 1"
        assert np.all(np.isfinite(result)), "Filter values should be finite"
    
    def test_lowpassfilter_validation(self):
        """Test lowpassfilter input validation."""
        size = (32, 32)
        n = 2
        
        # Test invalid cutoff values
        with pytest.raises(Exception):
            lowpassfilter(size, -0.1, n)  # cutoff < 0
        
        with pytest.raises(Exception):
            lowpassfilter(size, 0.6, n)   # cutoff > 0.5
        
        # Test invalid n (should be integer)
        with pytest.raises(Exception):
            lowpassfilter(size, 0.25, 2.5)  # non-integer n
    
    
    def test_rayleighmode_basic(self):
        """Test basic rayleighmode functionality."""
        # Generate Rayleigh-distributed data
        np.random.seed(42)  # For reproducibility
        sigma = 2.0
        data = np.random.rayleigh(sigma, 1000)
        
        mode = rayleighmode(data)
        
        assert isinstance(mode, (float, np.floating)), "Should return scalar"
        assert mode > 0, "Mode should be positive"
        assert np.isfinite(mode), "Mode should be finite"
        
        # Mode should be approximately sigma for Rayleigh distribution
        assert abs(mode - sigma) < 1.0, f"Mode {mode} should be close to {sigma}"
    
    def test_rayleighmode_with_bins(self):
        """Test rayleighmode with different number of bins."""
        np.random.seed(42)
        data = np.random.rayleigh(2.0, 1000)
        
        mode1 = rayleighmode(data, nbins=25)
        mode2 = rayleighmode(data, nbins=100)
        
        # Results should be similar but not necessarily identical
        assert abs(mode1 - mode2) < 1.0, "Different bin counts should give similar results"
    
    def test_perfft2_basic(self):
        """Test basic perfft2 functionality."""
        # Create test image
        test_image = np.random.rand(32, 32)
        
        # Test default behavior (returns S only)
        result = perfft2(test_image, compute_P=False, compute_spatial=False)
        
        assert isinstance(result, np.ndarray), "Should return numpy array"
        assert result.shape == test_image.shape, "Shape should match input"
        assert result.dtype == np.complex128, "Should return complex array"
    
    def test_perfft2_with_P(self):
        """Test perfft2 with P computation."""
        test_image = np.random.rand(16, 16)
        
        result = perfft2(test_image, compute_P=True, compute_spatial=False)
        
        assert isinstance(result, tuple), "Should return tuple when compute_P=True"
        assert len(result) == 2, "Should return (S, P)"
        
        S, P = result
        assert S.shape == test_image.shape, "S shape should match input"
        assert P.shape == test_image.shape, "P shape should match input"
        assert S.dtype == np.complex128, "S should be complex"
        assert P.dtype == np.complex128, "P should be complex"
    
    def test_perfft2_with_spatial(self):
        """Test perfft2 with spatial domain computation."""
        test_image = np.random.rand(16, 16)
        
        result = perfft2(test_image, compute_P=True, compute_spatial=True)
        
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 4, "Should return (S, P, s, p)"
        
        S, P, s, p = result
        assert s.shape == test_image.shape, "s shape should match input"
        assert p.shape == test_image.shape, "p shape should match input"
        assert s.dtype in [np.float32, np.float64], "s should be real"
        assert p.dtype in [np.float32, np.float64], "p should be real"
        
        # Verify decomposition: im ≈ s + p
        reconstructed = s + p
        np.testing.assert_array_almost_equal(test_image, reconstructed, decimal=10)
    
    def test_perfft2_data_types(self):
        """Test perfft2 with different data types."""
        # Test with integer input
        int_image = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        result = perfft2(int_image, compute_P=False, compute_spatial=False)
        
        assert isinstance(result, np.ndarray), "Should handle integer input"
        assert np.all(np.isfinite(result)), "Result should be finite"