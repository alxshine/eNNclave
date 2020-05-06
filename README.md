# eNNclave

## Setting up a testing environment

Building SGX and running SGX enclaves requires the SGX driver, Platform Service Module, and SGX SDK.
Our test machines use the Linux versions for Ubuntu Server 18.04.
For convenience a setup script is provided [here](https://github.com/alxshine/sgx-installer).

The python requirements can be found in [requirements.txt](requirements.txt).
For testing without GPU support we provide an alternate python package list in [requirements_cpu.txt](requirements_cpu.txt)

## Obtaining the datasets

TODO

## Training a model

TODO

## Extracting the enclave

TODO: reword

The script called `build_enclave.py` is used to generate the weight files and the C functions.
It takes two parameters: the original model file, and the number of layers to extract into an enclave.
The extracted layers will be replaced by an `EnclaveLayer`, which wraps the generated enclave in a manner compatible with the TensorFlow API.
From the original layers that were not extracted and the new `EnclaveLayer` it builds a new model and saves it.

## Compiling the enclave

TODO: reword

Building the enclave (or native) code happens in the `lib` directory, so move the generated files there.

The decision which version to build is decided based on the `MODE` environment variable.
All directories contain Makefiles, so running `make` in the project root will build all necessary subdirectories.


## Setting up `LD_LIBRARY_PATH`

The enclave model needs to be able to find the shared libraries that were previously compiled.
To provide the location of the libraries, please run this command from the project root:

    source setup_ld_path.sh

## Evaluating models

TODO