from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import numpy as np
from os.path import join
import os
from invoke.context import Context
import unittest

import templates
import ennclave_inference as ennclave
import config as cfg


def common(backend: str):
    target_dir = join(cfg.get_ennclave_home(), 'backend', 'generated')

    preamble_backend = backend
    if backend == 'sgx':
        preamble_backend = 'sgx_enclave'

    with open(join(target_dir, f'{backend}_forward.cpp'), 'w+') as forward_file:
        forward_file.write(templates.preamble.render(backend=preamble_backend))
        forward_file.write(
            f"print_out(\"Hello, this is backend {backend}\\n\");")
        forward_file.write(templates.postamble)

    with open(join(target_dir, 'parameters.bin'), 'w') as parameter_file:
        pass

    with open(join(target_dir, 'sgx_config.xml'), 'w') as config_file:
        config_file.write("""     
<EnclaveConfiguration>
  <ProdID>0</ProdID>
  <ISVSVN>0</ISVSVN>
  <StackMaxSize>0x40000</StackMaxSize>
  <HeapInitSize>0x7e00000</HeapInitSize>
  <HeapMaxSize>0x7e00000</HeapMaxSize>
  <TCSNum>10</TCSNum>
  <TCSPolicy>1</TCSPolicy>
  <!-- Recommend changing 'DisableDebug' to 1 to make the sgx undebuggable for sgx release -->
  <DisableDebug>0</DisableDebug>
  <MiscSelect>0</MiscSelect>
  <MiscMask>0xFFFFFFFF</MiscMask>
</EnclaveConfiguration>""")

    context = Context()
    with context.cd(cfg.get_ennclave_home()):
        context.run('mkdir -p build')
        with context.cd('build'):
            # context.run('cmake ..')
            context.run(f'make backend_{backend}')

    if backend == 'native':
        ennclave.native_forward(b'', 0, 0)
    else:
        ennclave.sgx_forward(b'', 0, 0)


# noinspection PyMethodMayBeStatic
class BasicTests(unittest.TestCase):
    def test_native(self):
        common('native')

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def test_sgx(self):
        common('sgx')
