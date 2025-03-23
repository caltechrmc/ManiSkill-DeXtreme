from enum import Enum

SMALL_CUBE_SIZE = 0.05
LARGE_CUBE_SIZE = 0.09

SMALL_CYLINDER_DIAMETER = 0.06
MEDIUM_CYLINDER_DIAMETER = 0.08
LARGE_CYLINDER_DIAMETER = 0.1
CYLINDER_HEIGHT = 0.08

class RGMCCubeSize(Enum):
    SMALL = 0
    LARGE = 1

class RGMCCylinderSize(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    
def get_cube_size(size: RGMCCubeSize) -> int:
    """Get the side length of a competition cube size (in m)."""
    match size:
        case RGMCCubeSize.SMALL:
            return SMALL_CUBE_SIZE
        case RGMCCubeSize.LARGE:
            return LARGE_CUBE_SIZE
        
def get_cylinder_diameter(size: RGMCCylinderSize) -> int:
    """Get the diameter of a competition cylinder size (in m)."""
    match size:
        case RGMCCylinderSize.SMALL:
            return SMALL_CYLINDER_DIAMETER
        case RGMCCylinderSize.MEDIUM:
            return MEDIUM_CYLINDER_DIAMETER
        case RGMCCylinderSize.LARGE:
            return LARGE_CYLINDER_DIAMETER
        
def get_cylinder_height(size: RGMCCylinderSize) -> int:
    """Get the height of a competition cylinder size (in m)."""
    return CYLINDER_HEIGHT