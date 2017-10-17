import numpy as np
import miil
import unittest

def test_pcfm_pcdrm_roundtrip():
    module = np.arange(miil.no_modules())
    fin = module // miil.no_modules_per_fin()
    cartridge = fin // miil.no_fins_per_cartridge()
    panel = cartridge // miil.no_cartridges_per_panel()
    cartridge %= miil.no_cartridges_per_panel()
    fin %= miil.no_fins_per_cartridge()
    module %= miil.no_modules_per_fin()

    daq, rena, rena_local_module = miil.pcfm_to_pcdrm(panel, cartridge, fin, module)
    fin_copy, module_copy = miil.pcdrm_to_pcfm(panel, cartridge, daq, rena, rena_local_module)


    assert(module_copy.size == module.size), 'module array size should be unchanged'
    assert((module_copy == module).all()), 'module round trip failed'

    assert(fin_copy.size == fin.size), 'fin array size should be unchanged'
    assert((fin_copy == fin).all()), 'fin round trip failed'
