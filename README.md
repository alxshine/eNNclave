
# Table of Contents

1.  [NN SGX](#org79de442)
    1.  [People](#org01062e0)
    2.  [Project Outline](#org8c488e9)
        1.  [First prototype <code>[10/12]</code>](#org485c92d)
        2.  [Improvements for paper <code>[0/3]</code>](#orgae288de)
        3.  [Future Work <code>[0/1]</code>](#orgba05680)
    3.  [Unforeseen events](#org0c27216)
        1.  [Figure out how many bits the enclave uses for floats](#org92a9b70)


<a id="org79de442"></a>

# NN SGX

Running the dense part of CNNs inside the trusted enclave to reduce leakage and protect against model stealing.
We hope to make this as robust against model stealing as online oracles.


<a id="org01062e0"></a>

## People

RBO, CPA, ASC


<a id="org8c488e9"></a>

## Project Outline


<a id="org485c92d"></a>

### First prototype <code>[10/12]</code>

1.  DONE Create working environment for Testing and Development

    <span class="timestamp-wrapper"><span class="timestamp">[2019-08-14 Wed]</span></span>
    The project unfortunately only runs on Ubuntu 18.04 and some other OSes, and not on my own machine&#x2026;
    I'm currently developing inside a VM.
    
    The project currently consists of a minimal Makefile for a single enclave and app that runs in simulation mode.
    Adding functions to the enclave requires their definition in the header file, and implementation, as for regular C code.
    Additionally, it requires addition to the `trusted` block in [Enclave/Enclave.edl](Enclave/Enclave.edl), with publicly accessible functions having the `public` keyword added to them.
    
    In the app, the functions are then called with a slightly different signature.
    The return value of enclave functions is always an sgx<sub>status</sub>, and the `global_eid` is added to the parameters.
    Return values of the function are set by the SGX wrapper through pointers, C-style.

2.  DONE Extract Weights from neural net

    I can store the weights of a model in HDF5 format, and then load that using their C++ API.
    Then I could either load the weights from inside the enclave (which gives us no benefit at all), or hardcode them.
    
    I need the activation functions anyway, so hardcoding is probably the way to go.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-03 Tue 09:40]</span></span>
    FUCK TENSORFLOW
    The C++ API requires building from source, and that requires bazel, and then everything together is a massive house of cards where the cards randomly self ignite, and then the example doesn't compile and nothing works.
    That seems like an unreasonable amount of super annoying work, which I don't want to do.
    
    So, instead I will try to create the NN in python, and create cython bindings for my C++ code.
    This probably means I will have to write a wrapper app for the enclave code, and maybe later make calls to the enclave directly.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-04 Wed 14:24]</span></span>
    Does it have to be Tensorflow?
    PyTorch seems to do the same thing but nicer, because it gives us more granular access to the underlying matrices.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-16 Mon 14:39]</span></span>
    I moved the code for executing the dense layers to their own function in the `Net` class.
    Now I can load the weights from the state<sub>dict</sub> and then do the computation using custom built functions for the matrix multiplication.
    This should serve as a good starting point for moving that first to C and then to the SGX enclave.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-16 Mon 15:30]</span></span>
    Finished moving the dense NN calculation to plain numpy.
    It's actually fairly simple, I just need to remember to transpose the weight matrix.
    Here's the source code for it:
    
        def dense(self, x):
            state = self.state_dict()
            # breakpoint()
            w1 = state['fc1.weight'].detach().numpy().T
            b1 = state['fc1.bias'].detach().numpy()
            x = x.detach().numpy()
            tmp = np.matmul(x, w1) + b1
            x = np.maximum(tmp, 0)
        
            w2 = state['fc2.weight'].detach().numpy().T
            b2 = state['fc2.bias'].detach().numpy()
            tmp = np.matmul(x, w2) + b2
            x = np.maximum(tmp, 0)
            return torch.Tensor(x)

3.  DONE Understand biases in Pytorch NNS

    I need to understand how biases are used in pytorch, so I can correctly model that behaviour.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-16 Mon 16:50]</span></span>
    They are simply added after the input is multiplied with the weight.
    Then the sum of the multiplication result and the biases is run through the nonlinearity.

4.  DONE Write a script that generates C arrays from pytorch weight matrices

    Because the weights should be hardcoded into the enclave (this avoids decryption during loading for now), I need a script to extract weights from a `.pt` file and generate C array definitions from it.
    
    See [this script](./python/torch/gen_headers.py).

5.  DONE Choose test network for first prototype

    Just use MNIST CNN, I have working code for torch

6.  DONE Write naive matrix multiplication code in C <code>[3/3]</code>

    Just simple stuff, including testing
    
    -   [X] matrix matrix multiplication
    -   [X] matrix matrix addition
    -   [X] nonlinearity

7.  DONE Compare results of torch, numpy and my C code

    There seem to be some differences in results that can't entirely be blamed on rounding errors.
    The differences are currently in the range of e-7, which is more than numpy uses for its `np.allclose` function (e-8).
    I will build a small network and then compare all three methods of computation, to see what is happening and where it starts to diverge.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-19 Thu 10:26]</span></span>
    Finished evaluation, see [test<sub>correctness.py</sub>](test_correctness.py).
    Even though the differences are around e-6, they seem to be normal distributed, and the max/min increases with increased matrix size (which strengthens my belief in the normal distribution).
    I don't think it is a problem, and if it becomes one I already have a test script set up for evaluation.

8.  DONE Move the fully connected layers to naive matrix multiplication in C

    In the forward function of the network, instead of invoking `nn.Linear` I can call a C function.
    This means that pytorch doesn't know how to backpropagate, but it doesn't need to anymore.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-18 Wed 17:10]</span></span>
    The numpy and C variants do give slightly different results than the pure pytorch variant (in the e-7 range).
    Jakob thinks this is more than just rounding errors, so I should check that out.
    See [this TODO](#orgd276bad)

9.  DONE Combine C functions into one `dense` function

    This function can then be moved to the enclave, otherwise it leaks intermediate values

10. DONE Add compile<sub>commands</sub>

    Added a single recursive Makefile, which can be wrapped using bear

11. TODO Do naive matrix multiplication inside the SGX enclave

    Move the aforementioned code function to the enclave.
    With a well enough defined interface this shouldn't be too much work

12. TODO Feed result back into pytorch, or calculate softmax

    As we want this to be an oracle, I should execute the "softmax" (just taking the maximum)
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-19 Thu 14:21]</span></span>
    matutil<sub>dense</sub>() already only returns the label


<a id="orgae288de"></a>

### Improvements for paper <code>[0/3]</code>

1.  TODO Test on actual hardware

2.  TODO Automate memory layout inside SGX

    We might have to do some memory magic because otherwise we might run out of memory inside the SGX.
    The first prototype can do this explicitly for the chosen network, but for publishing we should do this automatically.
    
    Alternative we could also generate a fixed function from sparse matrix multiplication.
    This would mean that we go through the output cell by cell, calculating all immediate steps in a row.
    Using this would throw away any shared results, and be much slower.
    However, this could help us avoid memory issues.


<a id="orgba05680"></a>

### Future Work <code>[0/1]</code>

1.  TODO Integrate with the framework API

    Rainer said it would be nice to integrate the SGX with the tensorflow API (or pytorch, whatever)


<a id="org0c27216"></a>

## Unforeseen events


<a id="org92a9b70"></a>

### TODO Figure out how many bits the enclave uses for floats

This could cause some weird results and incompatibilities.
It's also not perfectly clear if the enclave even supports floats.

