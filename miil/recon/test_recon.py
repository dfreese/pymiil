import miil
import numpy as np

def test_kwarg_raise():
    wrong_keyword_caught = False
    try:
        model = miil.recon.BreastPETSystemMatrix([1, 2, 3], wrong_keyword=True)
    except AttributeError:
        wrong_keyword_caught = True
    assert(wrong_keyword_caught)

def test_forward_project():
    image_shape = (324, 128, 68)
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=0.5, tor_width=-1)
    lors = np.array((855655487,), dtype=np.int64) # LOR straight across FOV
    val = model._forward_project(np.ones(image_shape, np.float32), lors)
    assert((val == image_shape[1]).all())

    val = model.forward_project(np.ones(image_shape, np.float32), lors)
    assert((val == image_shape[1]).all())

def test_forward_project_with_crystal_weights():
    image_shape = (324, 128, 68)
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=0.5, tor_width=-1)
    lors = np.array((855655487,), dtype=np.int64) # LOR straight across FOV
    ref = 2.0
    crystal_weights = ref * np.ones(miil.no_crystals())
    val = model._forward_project(np.ones(image_shape, np.float32), lors,
                                crystal_weights=crystal_weights)
    assert((val == (ref ** 2) * image_shape[1]).all())

    val = model.forward_project(np.ones(image_shape, np.float32), lors,
                                crystal_weights=crystal_weights)
    assert((val == (ref ** 2) * image_shape[1]).all())

def test_forward_project_with_slor_weights():
    image_shape = (324, 128, 68)
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=0.5, tor_width=-1)
    lors = np.array((855655487,), dtype=np.int64) # LOR straight across FOV
    ref = 2.0
    slor_weights = ref * np.ones(miil.no_slors())
    val = model._forward_project(np.ones(image_shape, np.float32), lors,
                                slor_weights=slor_weights)
    assert((val == ref * image_shape[1]).all())

    val = model.forward_project(np.ones(image_shape, np.float32), lors,
                                slor_weights=slor_weights)
    assert((val == ref * image_shape[1]).all())

def test_forward_project_to_crystals():
    image_shape = (324, 128, 68)
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=0.5, tor_width=-1)
    lors = np.array((855655487,), dtype=np.int64) # LOR straight across FOV
    ref = 2.0
    slor_weights = ref * np.ones(miil.no_slors())
    val = model.forward_project(np.ones(image_shape, np.float32), lors,
                                slor_weights=slor_weights,
                                return_val=False, return_crystals=True)

    expected = miil.get_crystal_distribution(lors, model.system_shape) * \
        ref * image_shape[1]
    assert((val == expected).all())

def test_forward_project_to_slors():
    image_shape = (324, 128, 68)
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=0.5, tor_width=-1)
    lors = np.array((855655487,), dtype=np.int64) # LOR straight across FOV
    ref = 2.0
    slor_weights = ref * np.ones(miil.no_slors())
    val = model.forward_project(np.ones(image_shape, np.float32), lors,
                                slor_weights=slor_weights,
                                return_val=False, return_slors=True)

    expected = miil.lor_to_slor_bins(lors, model.system_shape) * \
        ref * image_shape[1]
    assert((val == expected).all())

def test_forward_project_multiple_subsets():
    image_shape = (324, 128, 68)
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=0.5, tor_width=-1)
    lors = np.array(2 * (855655487,), dtype=np.int64) # LOR straight across FOV

    val = model.forward_project(np.ones(image_shape, np.float32),
                                lors, subset_size=1)
    assert((val == image_shape[1]).all())
    assert(val.size == 2)

def test_back_project():
    image_shape = (324, 128, 68)
    voxel_size_mm = 0.5
    sigma = voxel_size_mm
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=voxel_size_mm,
                                         sigma=sigma,
                                         tor_width=1)
    lors = np.array((855655487,), dtype=np.int64) # LOR straight across FOV
    val = model._back_project(lors)

    x_mm = voxel_size_mm * (np.arange(0, image_shape[0]) -
                            image_shape[0] / 2 + 0.5)
    z_mm = voxel_size_mm * (np.arange(0, image_shape[2]) -
                            image_shape[2] / 2 + 0.5)

    pos0, pos1 = miil.get_lor_positions(lors)
    ref = np.ones(image_shape, dtype=np.float32)
    d2 = ((pos0[0,0] - x_mm[:, None, None]) / voxel_size_mm) ** 2 + \
         ((pos0[0,2] - z_mm[None, None, :]) / voxel_size_mm) ** 2
    d2_exp = np.exp(-d2 * sigma)
    d2_exp[d2 > 1.0] = 0
    ref *= d2_exp
    assert(((val == 0) == (ref == 0)).all())
    assert((np.abs(val - ref) < 1e-5).all())

    val = model.back_project(lors)
    assert(((val == 0) == (ref == 0)).all())
    assert((np.abs(val - ref) < 1e-5).all())

def test_back_project_with_crystal_weights():
    image_shape = (324, 128, 68)
    voxel_size_mm = 0.5
    sigma = voxel_size_mm
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=voxel_size_mm,
                                         sigma=sigma,
                                         tor_width=1)
    weight_val = 2.0
    crystal_weights = weight_val * np.ones(miil.no_crystals())
    lors = np.array((855655487,), dtype=np.int64) # LOR straight across FOV
    val = model._back_project(lors, crystal_weights=crystal_weights)

    x_mm = voxel_size_mm * (np.arange(0, image_shape[0]) -
                            image_shape[0] / 2 + 0.5)
    z_mm = voxel_size_mm * (np.arange(0, image_shape[2]) -
                            image_shape[2] / 2 + 0.5)

    pos0, pos1 = miil.get_lor_positions(lors)
    ref = np.ones(image_shape, dtype=np.float32)
    d2 = ((pos0[0,0] - x_mm[:, None, None]) / voxel_size_mm) ** 2 + \
         ((pos0[0,2] - z_mm[None, None, :]) / voxel_size_mm) ** 2
    d2_exp = np.exp(-d2 * sigma)
    d2_exp[d2 > 1.0] = 0
    ref *= d2_exp
    ref *= weight_val ** 2
    assert(((val == 0) == (ref == 0)).all())
    assert((np.abs(val - ref) < 1e-5).all())

    val = model.back_project(lors, crystal_weights=crystal_weights)
    assert(((val == 0) == (ref == 0)).all())
    assert((np.abs(val - ref) < 1e-5).all())

def test_back_project_with_slor_weights():
    image_shape = (324, 128, 68)
    voxel_size_mm = 0.5
    sigma = voxel_size_mm
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=voxel_size_mm,
                                         sigma=sigma,
                                         tor_width=1)
    weight_val = 2.0
    slor_weights = weight_val * np.ones(miil.no_slors())
    lors = np.array((855655487,), dtype=np.int64) # LOR straight across FOV
    val = model._back_project(lors, slor_weights=slor_weights)

    x_mm = voxel_size_mm * (np.arange(0, image_shape[0]) -
                            image_shape[0] / 2 + 0.5)
    z_mm = voxel_size_mm * (np.arange(0, image_shape[2]) -
                            image_shape[2] / 2 + 0.5)

    pos0, pos1 = miil.get_lor_positions(lors)
    ref = np.ones(image_shape, dtype=np.float32)
    d2 = ((pos0[0,0] - x_mm[:, None, None]) / voxel_size_mm) ** 2 + \
         ((pos0[0,2] - z_mm[None, None, :]) / voxel_size_mm) ** 2
    d2_exp = np.exp(-d2 * sigma)
    d2_exp[d2 > 1.0] = 0
    ref *= d2_exp
    ref *= weight_val
    assert(((val == 0) == (ref == 0)).all())
    assert((np.abs(val - ref) < 1e-5).all())

    val = model.back_project(lors, slor_weights=slor_weights)
    assert(((val == 0) == (ref == 0)).all())
    assert((np.abs(val - ref) < 1e-5).all())

def test_back_project_multiple_subsets():
    image_shape = (324, 128, 68)
    voxel_size_mm = 0.5
    sigma = voxel_size_mm
    model = miil.recon.BreastPETSystemMatrix(image_shape,
                                         voxel_size_mm=voxel_size_mm,
                                         sigma=sigma,
                                         tor_width=1)
    lors = np.array(2 * (855655487,), dtype=np.int64) # LOR straight across FOV

    x_mm = voxel_size_mm * (np.arange(0, image_shape[0]) -
                            image_shape[0] / 2 + 0.5)
    z_mm = voxel_size_mm * (np.arange(0, image_shape[2]) -
                            image_shape[2] / 2 + 0.5)

    pos0, pos1 = miil.get_lor_positions(lors)
    ref = np.ones(image_shape, dtype=np.float32)
    d2 = ((pos0[0,0] - x_mm[:, None, None]) / voxel_size_mm) ** 2 + \
         ((pos0[0,2] - z_mm[None, None, :]) / voxel_size_mm) ** 2
    d2_exp = np.exp(-d2 * sigma)
    d2_exp[d2 > 1.0] = 0
    ref *= d2_exp
    # Double the reference for the two of the same lors
    ref *= 2

    # split the two lors into two subsets.
    val = model.back_project(lors, subset_size=1)
    assert(((val == 0) == (ref == 0)).all())
    assert((np.abs(val - ref) < 1e-5).all())
