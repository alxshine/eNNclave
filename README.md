# eNNclave

This is the eNNclave framework for automatically generating partial black-boxes for machine learning (ML) models.
It takes a trained ML model as input and generates code to then be compiled into a trusted enclave for trusted execution environments.
Currently only TensorFlow and Intel's SGX are supported, but more trusted processor implementations and ML frameworks will be added in the future.
This framework is under active development.

The original paper for it is [eNNclave: Offline Inference with Model Confidentiality](https://dl.acm.org/doi/10.1145/3411508.3421376), and it was authored by [Alexander Schlögl](https://informationsecurity.uibk.ac.at/people/alexander-schloegl/) and [Rainer Böhme](https://informationsecurity.uibk.ac.at/people/rainer-boehme/).
Our experiments presented in the paper are extracted to a separate [repository](https://github.com/alxshine/ennclave-experiments).
For project specific questions, problems and comments feel free to open an issue or send a mail to [Alexander Schlögl](mailto:alexander.schloegl@uibk.ac.at).

## Setup

This section lists the required tools, libraries and SDKs required for compiling and running the project.
More detailed explanations may be added in the future if necessary.

### Prerequisites

The core libraries are written in C++ and build files are generated using `cmake`, with a minimum version of 3.10.
I have observed problems during compilation with `clang`, so `gcc` is recommended.
To use the interoperability with TensorFlow you need `python`.

### SGX Driver & SDK

For Linux the SGX driver and SDK can be found [here](https://github.com/intel/linux-sgx).
There are precompiled releases for common distributions [here](https://github.com/intel/linux-sgx/releases).

Alternatively, I provide a [Dockerfile](Dockerfile) to run everything in a docker container.
I am currently not 100% confident in the stability of that, but improvements are underway.

### Setting environment variables

Due to the architecture of this framework the code generation process needs to know where the root of the eNNclave framework is (i.e. the directory that this README is in).
This should be stored in the `ENNCLAVE_HOME` environment variable.

## Compiling the project

You can compile the code wherever you want, the resulting libraries are stored in `$ENNCLAVE_HOME/lib`.
However, the [experiment repository](https://github.com/alxshine/ennclave-experiments) expects the build directory to be `$ENNCLAVE_HOME/build`.

## Installing the python frontend

To call the generated enclaves from python I provide an interoperability layer.
This has to be installed in your desired python environment.
If you have `pip` available navigate to the [frontend directory](frontend/python) and run
```shell
pip install .
```
This will automatically compile the C file and install it in your python environment.

If you want to make changes to the frontend install with the `-e` parameter, but keep in mind the C code is not automatically compiled.

## Performing inference

I will write a small tutorial with code ASAP, I promise.
For the time being you can take a look at [our basic python test](frontend/python/tests/test_basics.py), or the [experiment repository](https://github.com/alxshine/ennclave-experiments)

## Anything else

If there are any problems, please open an issue.
If you want to collaborate for a project, please send me an email.

Contributions are appreciated, feel free to open a pull request :)

## Disclaimer

THIS IS RESEARCH WORK!
I give no guarantee that it works, is stable, or does what you expect it to.

## DO NOT USE FOR PRODUCTION!