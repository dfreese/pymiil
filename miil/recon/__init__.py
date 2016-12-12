#!/usr/bin/env python

import numpy as np
import miil

try:
    import projection

    class BreastPETSystemMatrix():
        def __init__(self, image_size, system_shape=None, position_params=None,
                     recon_param=None, **kwargs):
            self.system_shape = system_shape
            self.no_crystals = miil.no_crystals(self.system_shape)
            self.crystal_pos = miil.get_position_global_crystal(
                xrange(self.no_crystals))
            self.image_size = image_size

            if position_params is None:
                relevant_kwargs = miil.PositionParams.__init__.im_func.func_code.co_varnames[1:]
                relevant_dict = {k:kwargs[k] for k in kwargs if k in relevant_kwargs}
                self.position_params = miil.PositionParams(**relevant_dict)
                kwargs = {k:kwargs[k] for k in kwargs if k not in relevant_kwargs}
            else:
                if not isinstance(position_params, miil.PositionParams):
                    raise TypeError('position_params is not PositionParams class')
                self.position_params = position_params

            if recon_param is None:
                relevant_kwargs = projection.ReconParam.__init__.im_func.func_code.co_varnames[1:]
                relevant_dict = {k:kwargs[k] for k in kwargs if k in relevant_kwargs}
                recon_param = projection.ReconParam(**relevant_dict)
                kwargs = {k:kwargs[k] for k in kwargs if k not in relevant_kwargs}
            else:
                if not isinstance(recon_param, projection.ReconParam):
                    raise TypeError('recon_param is not ReconParam class')

            # We're passing kwargs down to their relevant classes, and then
            # remove them from kwargs.  If there's any left, then we need to
            # throw an error, so misspelled keywords don't pass silently.
            if len(kwargs) > 0:
                raise AttributeError('unused kwargs: %s' %
                                     ', '.join(kwargs.keys()))

            self.projection = projection.Projection(image_size,
                                                    recon_param=recon_param)

        def _forward_project(self, image, lors, weight=None,
                             crystal_weights=None, slor_weights=None):
            lors = np.asarray(lors, dtype=np.int64)
            if weight is None:
                weight = np.ones(lors.size, dtype=np.float32)
            else:
                weight = np.asarray(weight, dtype=np.float32)
            c0, c1 = miil.get_crystals_from_lor(lors, self.system_shape)
            if crystal_weights is not None:
                crystal_weights = np.asarray(crystal_weights)
                weight *= crystal_weights[c0] * crystal_weights[c1]
            if slor_weights is not None:
                slor_weights = np.asarray(slor_weights)
                slor_idx = miil.lor_to_slor(lors)
                weight *= slor_weights[slor_idx]
            lines = projection.Lines(c0, c1, self.crystal_pos, weight=weight)
            return self.projection.forward_project(lines, image)

        def forward_project(self, image, lors=None, weight=None,
                            crystal_weights=None, slor_weights=None,
                            subset_size=40000000, return_val=True,
                            return_slors=False, return_crystals=False):
            if not return_val and not return_crystals and not return_slors:
                raise ValueError('No output selected')
            if lors is None:
                no_lors = miil.no_lors(self.system_shape)
            else:
                lors = np.asarray(lors, dtype=np.int64)
                no_lors = lors.size
            no_crystals = miil.no_crystals(self.system_shape)
            no_slors = miil.no_slors(self.system_shape)

            if return_val:
                lor_val = np.zeros(no_lors, dtype=np.float32)
            if return_crystals:
                crystal_val = np.zeros(no_crystals)
            if return_slors:
                slor_val = np.zeros(no_slors)

            if weight is not None:
                weight = np.asarray(weight)
                if weight.size != no_lors:
                    raise RuntimeError('''Number of weights is not equal to
                                       the number of LORs''')
            start_idxs = np.arange(0, no_lors, subset_size)
            end_idxs = np.minimum(
                np.arange(0, no_lors, subset_size) + subset_size, no_lors)
            for start_idx, end_idx in zip(start_idxs, end_idxs):
                if lors is None:
                    local_lors = np.arange(start_idx, end_idx)
                else:
                    local_lors = lors[start_idx:end_idx]

                if weight is None:
                    local_weight = None
                else:
                    local_weight = weight[start_idx:end_idx]

                val = self._forward_project(
                    image, local_lors, weight=local_weight,
                    crystal_weights=crystal_weights, slor_weights=slor_weights)
                if return_val:
                    lor_val[start_idx:end_idx] = val.copy()
                if return_slors:
                    slor_idx = miil.lor_to_slor(local_lors, self.system_shape)
                    slor_val += np.bincount(slor_idx, weights=val,
                                            minlength=no_slors)
                if return_crystals:
                    c0, c1 = miil.get_crystals_from_lor(local_lors,
                                                        self.system_shape)
                    crystal_val += np.bincount(c0, weights=val,
                                               minlength=no_crystals)
                    crystal_val += np.bincount(c1, weights=val,
                                               minlength=no_crystals)
            out = []
            if return_val:
                out.append(lor_val)
            if return_slors:
                out.append(slor_val)
            if return_crystals:
                out.append(crystal_val)

            if len(out) > 1:
                return out
            else:
                return out[0]

        def _back_project(self, lors, value=None,
                          crystal_weights=None, slor_weights=None):
            lors = np.asarray(lors, dtype=np.int64)
            if value is None:
                value = np.ones(lors.size)
            else:
                value = np.asarray(value)
            c0, c1 = miil.get_crystals_from_lor(lors, self.system_shape)
            if crystal_weights is not None:
                crystal_weights = np.asarray(crystal_weights)
                value *= crystal_weights[c0] * crystal_weights[c1]
            if slor_weights is not None:
                slor_idx = miil.lor_to_slor(lors)
                value *= slor_weights[slor_idx]
            lines = projection.Lines(c0, c1, self.crystal_pos, weight=value)
            return self.projection.back_project(lines)

        def back_project(self, lors=None, value=None,
                         crystal_weights=None, slor_weights=None,
                         subset_size=40000000):
            if lors is None:
                no_lors = miil.no_lors(self.system_shape)
            else:
                lors = np.asarray(lors, dtype=np.int64)
                no_lors = lors.size

            if value is not None:
                value = np.asarray(value)
                if value.size != no_lors:
                    raise RuntimeError('''Number of values is not equal to
                                       the number of LORs''')
            if crystal_weights is not None:
                crystal_weights = np.asarray(crystal_weights, dtype=np.float64)
                if crystal_weights.size != miil.no_crystals(self.system_shape):
                    raise RuntimeError('''Number of crystal weights is not
                                       equal to the number of crystals''')

            if slor_weights is not None:
                slor_weights = np.asarray(slor_weights, dtype=np.float64)
                if slor_weights.size != miil.no_slors(self.system_shape):
                    raise RuntimeError('''Number of slor weights is not equal
                                       to the number of slors''')
            image = np.zeros(self.image_size, dtype=np.float32)
            start_idxs = np.arange(0, no_lors, subset_size)
            end_idxs = np.minimum(
                np.arange(0, no_lors, subset_size) + subset_size, no_lors)
            for start_idx, end_idx in zip(start_idxs, end_idxs):
                if lors is None:
                    local_lors = np.arange(start_idx, end_idx)
                else:
                    local_lors = lors[start_idx:end_idx]

                if value is None:
                    local_value = None
                else:
                    local_value = value[start_idx:end_idx]

                local_image = self._back_project(
                    local_lors, value=local_value,
                    crystal_weights=crystal_weights, slor_weights=slor_weights)
                image += local_image
            return image

except ImportError:
    class BreastPETSystemMatrix():
        def __init__(self, image_size, system_shape=None, position_params=None,
                     recon_param=None, **kwargs):
            raise NotImplementedError('Need the projection module')
