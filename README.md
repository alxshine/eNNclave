
# Table of Contents

1.  [NN SGX](#orgf54201d)
    1.  [People](#orgcc8537d)
    2.  [Project Outline](#org6324bd4)
        1.  [First prototype <code>[1/5]</code>](#orgbc5cb97)
        2.  [Improvements for paper <code>[0/3]</code>](#org1850bbc)
        3.  [Future Work <code>[0/1]</code>](#org35a3772)
    3.  [Unforeseen events](#org076fc3d)
        1.  [Figure out how many bits the enclave uses for floats](#orgaa03e6f)
2.  [Project Diary](#orga7b97fa)


<a id="orgf54201d"></a>

# NN SGX

Running the dense part of CNNs inside the trusted enclave to reduce leakage and protect against model stealing.
We hope to make this as robust against model stealing as online oracles.


<a id="orgcc8537d"></a>

## People

RBO, CPA, ASC


<a id="org6324bd4"></a>

## Project Outline


<a id="orgbc5cb97"></a>

### First prototype <code>[1/5]</code>

1.  DONE Create working environment for Testing and Development

2.  TODO Extract Weights from Tensorflow Tensors

3.  TODO Choose test network for first prototype

4.  TODO Do naive matrix multiplication inside the SGX enclave

5.  TODO Feed result back into tensorflow, or calculate softmax


<a id="org1850bbc"></a>

### Improvements for paper <code>[0/3]</code>

1.  TODO Test on actual hardware

2.  TODO Automate memory layout inside SGX

    We might have to do some memory magic because otherwise we might run out of memory inside the SGX.
    The first prototype can do this explicitly for the chosen network, but for publishing we should do this automatically.


<a id="org35a3772"></a>

### Future Work <code>[0/1]</code>

1.  TODO Integrate with the Tensorflow API

    Rainer said it would be nice to integrate the SGX with the tensorflow API


<a id="org076fc3d"></a>

## Unforeseen events


<a id="orgaa03e6f"></a>

### TODO Figure out how many bits the enclave uses for floats

This could cause some weird results and incompatibilities.
It's also not perfectly clear if the enclave even supports floats.


<a id="orga7b97fa"></a>

# Project Diary

<span class="timestamp-wrapper"><span class="timestamp">[2019-08-14 Wed]</span></span>
The project unfortunately only runs on Ubuntu 18.04 and some other OSes, and not on my own machine&#x2026;
I'm currently developing inside a VM.

The project currently consists of a minimal Makefile for a single enclave and app that runs in simulation mode.
Adding functions to the enclave requires their definition in the header file, and implementation, as for regular C code.
Additionally, it requires addition to the `trusted` block in [Enclave/Enclave.edl](Enclave/Enclave.edl), with publicly accessible functions having the `public` keyword added to them.

In the app, the functions are then called with a slightly different signature.
The return value of enclave functions is always an sgx<sub>status</sub>, and the `global_eid` is added to the parameters.
Return values of the function are set by the SGX wrapper through pointers, C-style.

