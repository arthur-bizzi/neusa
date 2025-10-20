from scipy.interpolate import RegularGridInterpolator
import numpy as np

def create_u_interpolator(t_points, x_points, u_grid, method='linear', fill_value=None):
    """
    Create an interpolator for u(t,x) given a regular grid of values.
    
    Parameters:
    -----------
    t_points : array_like, 1D
        Time points in ascending order
    x_points : array_like, 1D
        Spatial points in ascending order
    u_grid : array_like, 2D
        Values of u at grid points, shape should be (len(t_points), len(x_points))
    method : str
        Interpolation method: 'linear', 'nearest' or 'cubic'
    fill_value : float or None
        Value to use for points outside the grid
        
    Returns:
    --------
    callable
        Function to interpolate u at arbitrary (t,x) points
    """
    # Ensure grid points are in ascending order
    if not (np.all(np.diff(t_points) > 0) and np.all(np.diff(x_points) > 0)):
        raise ValueError("t_points and x_points must be in ascending order.")
    
    # Create interpolator
    interp = RegularGridInterpolator(
        (t_points, x_points), 
        u_grid, 
        method=method, 
        bounds_error=False, 
        fill_value=fill_value
    )
    
    # Return function that accepts (t,x) points
    def interpolate(points):
        """
        Interpolate u at given (t,x) points
        
        Parameters:
        -----------
        points : array_like, shape (..., 2)
            Points (t,x) at which to interpolate
            
        Returns:
        --------
        array_like
            Interpolated values of u
        """
        return interp(points)
    
    return interpolate