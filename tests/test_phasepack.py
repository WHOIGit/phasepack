import numpy as np
import pytest
import phasepack


class TestPhasepack:
    """Test suite for phasepack functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test image
        self.test_image = np.random.rand(32, 32).astype(np.float64)
        
        # Create an image with known features
        x, y = np.meshgrid(np.linspace(0, 2*np.pi, 32), np.linspace(0, 2*np.pi, 32))
        self.synthetic_image = np.sin(x) * np.cos(y)
    
    def test_import_all_functions(self):
        """Test that all expected functions are available."""
        expected_functions = ['phasecong', 'phasecongmono', 'phasesym', 'phasesymmono']
        
        for func_name in expected_functions:
            assert hasattr(phasepack, func_name), f"Function {func_name} not found"
            assert callable(getattr(phasepack, func_name)), f"Function {func_name} not callable"
    
    def test_phasecong_basic(self):
        """Test basic phasecong functionality."""
        result = phasepack.phasecong(self.test_image)
        
        # Should return a tuple/list of results
        assert isinstance(result, (tuple, list)), "phasecong should return tuple/list"
        assert len(result) > 0, "phasecong should return non-empty result"
        
        # First result should be an array with same shape as input
        pc = result[0]
        assert isinstance(pc, np.ndarray), "First result should be numpy array"
        assert pc.shape == self.test_image.shape, "Output shape should match input"
    
    def test_phasecongmono_basic(self):
        """Test basic phasecongmono functionality."""
        result = phasepack.phasecongmono(self.test_image)
        
        # Should return a tuple/list of results
        assert isinstance(result, (tuple, list)), "phasecongmono should return tuple/list"
        assert len(result) > 0, "phasecongmono should return non-empty result"
        
        # First result should be an array with same shape as input
        pc = result[0]
        assert isinstance(pc, np.ndarray), "First result should be numpy array"
        assert pc.shape == self.test_image.shape, "Output shape should match input"
    
    def test_phasesym_basic(self):
        """Test basic phasesym functionality."""
        result = phasepack.phasesym(self.test_image)
        
        # Should return a tuple/list of results
        assert isinstance(result, (tuple, list)), "phasesym should return tuple/list"
        assert len(result) > 0, "phasesym should return non-empty result"
        
        # First result should be an array with same shape as input
        ps = result[0]
        assert isinstance(ps, np.ndarray), "First result should be numpy array"
        assert ps.shape == self.test_image.shape, "Output shape should match input"
    
    def test_phasesymmono_basic(self):
        """Test basic phasesymmono functionality."""
        result = phasepack.phasesymmono(self.test_image)
        
        # Should return a tuple/list of results
        assert isinstance(result, (tuple, list)), "phasesymmono should return tuple/list"
        assert len(result) > 0, "phasesymmono should return non-empty result"
        
        # First result should be an array with same shape as input
        ps = result[0]
        assert isinstance(ps, np.ndarray), "First result should be numpy array"
        assert ps.shape == self.test_image.shape, "Output shape should match input"
    
    
    def test_synthetic_image(self):
        """Test functions work with synthetic image."""
        # All functions should work without error on synthetic image
        pc_result = phasepack.phasecong(self.synthetic_image)
        pcm_result = phasepack.phasecongmono(self.synthetic_image)
        ps_result = phasepack.phasesym(self.synthetic_image)
        psm_result = phasepack.phasesymmono(self.synthetic_image)
        
        # All should return valid results
        for result in [pc_result, pcm_result, ps_result, psm_result]:
            assert isinstance(result, (tuple, list))
            assert len(result) > 0
            assert isinstance(result[0], np.ndarray)
    
    def test_output_range(self):
        """Test that outputs are in reasonable ranges."""
        result = phasepack.phasecongmono(self.test_image)
        pc = result[0]
        
        # Phase congruency should be between 0 and 1
        assert np.all(pc >= 0), "Phase congruency values should be >= 0"
        assert np.all(pc <= 1), "Phase congruency values should be <= 1"
        assert np.all(np.isfinite(pc)), "Phase congruency values should be finite"