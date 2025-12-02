from scipy.ndimage import label, find_objects
from skimage.segmentation import find_boundaries


def get_volume(mask, voxel_volume=1):
    return np.sum(mask > 0) * voxel_volume

def get_island_count(mask):
    labeled, num = label(mask)
    return num

def get_area(mask, pixel_area=1.0):
    """Compute total area (nonzero pixels × pixel area)."""
    return np.sum(mask > 0) * pixel_area

