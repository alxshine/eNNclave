
# Table of Contents

1.  [NN SGX](#orgbb94a2a)
    1.  [People](#orga4739c5)
    2.  [Project Outline](#orgd397ea5)
        1.  [First prototype <code>[14/14]</code>](#org48016d5)
        2.  [Improvements for paper <code>[3/9]</code>](#org028f7c4)
        3.  [Future Work <code>[0/2]</code>](#org9606181)
    3.  [Unforeseen events](#orgb94c082)
        1.  [Figure out how many bits the enclave uses for floats](#orgce9c5db)
        2.  [Find out what is needed for the foreshadow mitigation](#org3ac2247):sgx:
        3.  [Test if our processors are vulnerable to foreshadow](#org1905900):sgx:


<a id="orgbb94a2a"></a>

# NN SGX

Running the dense part of CNNs inside the trusted enclave to reduce leakage and protect against model stealing.
We hope to make this as robust against model stealing as online oracles.


<a id="orga4739c5"></a>

## People

RBO, CPA, ASC


<a id="orgd397ea5"></a>

## Project Outline


<a id="org48016d5"></a>

### First prototype <code>[14/14]</code>

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

3.  DONE Understand biases in Pytorch NNs

    I need to understand how biases are used in pytorch, so I can correctly model that behaviour.
    
         <span class="timestamp-wrapper"><span class="timestamp">[2019-09-16 Mon 16:50]</span></span>
    They are simply added after the input is multiplied with the weight.
    Then the sum of the multiplication result and the biases is run through the nonlinearity.
    
         <span class="timestamp-wrapper"><span class="timestamp">[2019-09-25 Wed 10:54]</span></span>
    The reason why I was so confused by biases in Pytorch is the fact that multiple inputs in a batch are `hstacked` together into a single matrix.
    This means the computation the tensor does internally is more than simple matrix multiplication.
    In order to do this correctly manually, one has to iterate over every column and compute the matrix multiplication for that column alone.
    Then the dimensions of the bias vector also fit again.

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
    See [this TODO](#orga6b8650)

9.  DONE Combine C functions into one `dense` function

    This function can then be moved to the enclave, otherwise it leaks intermediate values

10. DONE Add compile<sub>commands</sub>

    Added a single recursive Makefile, which can be wrapped using bear

11. DONE Check the details of parameter passing into enclaves

    Simply passing input pointers to the matrices doesn't work, so I will need to test this with some smaller examples
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-23 Mon 11:33]</span></span>
    Adding a [in, count=s] parameter for every array tells the SGX autogenerated wrapper how large the array should be, and it is then copied to enclave memory

12. DONE Do naive matrix multiplication inside the SGX enclave

    Move the aforementioned code function to the enclave.
    With a well enough defined interface this shouldn't be too much work

13. DONE Feed result back into pytorch, or calculate softmax

    As we want this to be an oracle, I should execute the "softmax" (just taking the maximum)
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-19 Thu 14:21]</span></span>
    matutil<sub>dense</sub>() already only returns the label

14. DONE Find a way to actually call the Enclave from PyTorch

    Currently the enclave is not called from pytorch
    The wrapper is called correctly, and then no error happens, but the enclave is not called.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-23 Mon 14:22]</span></span>
    I had not initialized the enclave, and did not really output any errors in the enclave wrapper.
    So yeah, this one's on me.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-23 Mon 14:40]</span></span>
    Added some error handling to pymatutil.


<a id="org028f7c4"></a>

### Improvements for paper <code>[3/9]</code>

1.  DONE Find out how model architectures are stored

    We want to be able to specify a cut similar to current SOTA architecture visualization tools.
    
    1.  PyTorch
    
        PyTorch uses the python classes for model definition, usually only storing the state<sub>dict</sub>.
        However, it is also possible to store the entire **module** where the model is defined, using Python's [pickle](https://docs.python.org/3/library/pickle.html) module.
        Internally this does the same thing as keeping the file containing the python class and loading the state<sub>dict</sub>, as it stores that file as well.
        This means it is dependent on the containing folder structure, which might lead to some very weird errors.
        
        PyTorch has no innate visualization tool, it instead is directed more at a programmer's view of things, as it allows for debugging the actual code of the `forward` function.
        What one can do is export the model to [ONNX](#org240de07) format, and then visualize it using something like [VisualDL](https://github.com/PaddlePaddle/VisualDL).
        
        For our separation we could create some macro, wrapper, whatever that is then easily plugged into the definition of `forward`.
    
    2.  Tensorflow
    
        For Tensorflow, everything is in the session.
        The graph that describes the model, the state of layers, optimizers etc.
        The problem is that this way I'm not sure if my code would even work, as this would mean rebuilding everything so it works with Tensorflow.
        
        Tensorflow visualizes its graphs using Tensorboard, and gets the info for that from the log directory.
        A link to this is [here](https://www.tensorflow.org/guide/graph_viz).
        I'm not sure how easy we can use Tensorboard for input, as Tensorflow in general is hard to edit.
    
    3.  Open Neural Network Exchange - <a id="org240de07"></a>
    
        [ONNX](https://github.com/onnx/onnx) is meant to be a framework independent way of specifying models, complete with data types and operators.
        There are ways to **export** to ONNX format from most common tools, but not many have a way of **importing**.
        This makes it very difficult to use for our purposes, as my code would then still be framework dependent, just using an independent way to specify the model.
        
        ONNX does have its own runtime, and I could try and move dense parts of that to the SGX enclave, but that would make comparison harder for the (I assume) not super broadly adopted ONNX runtime.
    
    4.  Keras
    
        Keras tries to be a human readable extension sitting on top of Tensorflow, Theano etc.
        The layers of Keras (and therefore any custom layers) can be written in pure python, and I think they can then also call C functions.
        
        Storing models in Keras saves the architecture as well as the weights.
        Anything that also contains weights is stored as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file.
        It's also possible to store the architecture alone as a json file.
        
        Keras can do rudimentary visualization by using pydot, which uses [pydot](https://github.com/pydot/pydot) to interface with [graphviz](http://www.graphviz.org/).
        The function for this is `plot_model` in `keras.utils`.
        Alternatively Keras offers a `model.summary()` function which prints a summary of all the layers.
        
        For these reasons, **Keras** seems to be the best choice, offering the versatility and customizability we need along with nice tooling for training, storing and loading, as well as visualization.

2.  DONE Design a splitting functionality for Keras <code>[3/3]</code>

    RBO wants to have a good way of specifying split position and visualization of it before we try and run the code.
    The best way to do this is writing a Keras layer container (similar to Sequential).
    This way we have control over how the underlying layers function.
    
    1.  DONE Fix plotting for composed Keras models
    
        Keras models can be composed from each other.
        There is a concat function, but one model can actually just contain another model.
        For both the `model.summary()` function and the `plot` function however, the contained model is not summarized recursively.
        It is just listed as `sequential_2`.
        This will be the first step.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-09-25 Wed 15:55]</span></span>
        `plot_model` has a parameter for expanding submodules, called `expand_nested`.
        Setting this to true gives output like in [this image](python/mnist_cnn.png).
    
    2.  DONE Finish test code for MNIST in Keras
    
        In order to evaluate I need fully working code for MNIST
    
    3.  DONE Write extension to Keras Sequential <code>[5/5]</code>
    
        We want to have a container similar to Keras's `Sequential` model that can also generate C code for the contained layers.
        It should be possible to iterate over all contained layers and store the `output_shape` along the way.
        Then I also need to store the `input_shape` of the first layer and the weights and biases of all layers.
        Getting the activation functions could be a bit trickier.
        Here it is probably best to either call the C functions like the Keras names, or introduce a mapping.
        I'm not sure right now if I can actually get the name of the activation function or not, which could increase difficulty.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-09-25 Wed 18:32]</span></span>
        When executing a Keras Sequential (e.g. for `predict()`) at the execution at one point starts calling backend functions.
        This is something I can't really trace, and try and emulate this with my extension to `Sequential`.
        Due to this it's smarter to build a function that takes a sequential model and builds the C code from that.
        This could then be wrapped in a custom Keras layer that calls the C code.
        I would first however have to test whether calling C code from a Keras layer works at all.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-09-26 Thu 10:52]</span></span>
        I took a debugging dive into the `model.predict()` function.
        It's fairly easy to follow until Keras hands off execution to the Tensorflow backend.
        The model is collapsed into a single function, which makes extensive use of the TF session model (ugh&#x2026;).
        This function is then wrapped into a Keras `Function` object, which is also a TF `GraphExecutionFunction` object when using the TF backend.
        
        1.  DONE Test control flow for custom Keras layers
        
            <span class="timestamp-wrapper"><span class="timestamp">[2019-09-26 Thu 11:54]</span></span>
            Keras only sits on top of whatever backend it uses.
            This means that any custom Keras layer really only generates TF tensors.
            The contents of their [`call()`](https://keras.io/layers/writing-your-own-keras-layers/) function are really only called once, where they are converted to TF Graphexecution functions.
            Also, the `x` input is a backend tensor, so there is not really any way to call my C function.
            For that I would need the actual values, which they don't have at that point, because they are only placeholders&#x2026;
            
            The real problem is the inherent lazyness of TF 1.14.
            **HOWEVER**, there is a way to add eager functions to it, by wrapping them with [`tf.py_func`](https://www.tensorflow.org/api_docs/python/tf/py_function).
            This is a tensor wrapping a regular C function.
            With this I can wrap my `pymatutil` function in a regular python function, taking numpy arrays as input.
            
            Keras doesn't really have an equivalent.
            I'm currently not sure whether it's smarter to switch to TF in general or just use the TF function in Keras.
            I will check how easy it is to get weights out of TF NNs.
            
            With TF 2.0 (which is currently available as a beta) eager execution is the default, so this would make things a lot easier.
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-09-26 Thu 16:00]</span></span>
            So I switched to TF 2.0.0 rc2, because the eagerness seems to be exactly what I want.
            Unfortunately, during training it still switches to lazy execution, I guess for performance.
            However, once everything is done the environment is in eager mode again.
            I guess they never meant eager execution to work during full blown execution of the network.
            **HOWEVER**, this still helps me to write my code, as I can use the Keras bundled with TF, and use [`tf.py_func`](https://www.tensorflow.org/api_docs/python/tf/py_function) to call my code.
            Will test tomorrow.
        
        2.  DONE Use `tf.py_func` to call C code for testing
        
            It should probably be a simple operation like multiplying with 2.
        
        3.  DONE Test calling C Code from a custom Keras layer
        
            Does not make sense, see [this point](#org1bca657), lazyness is super annoying.
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-09-27 Fri 09:56]</span></span>
            Actually, with TF as backend this still works (see [this code](python/tf_poc.py)).
            The input to `Layer.call()` is a TF tensor in that case, and I can use that as an input to a `tf.py_function`.
            This `tf.py_function` can then call my interop code.
        
        4.  DONE Extract weights, biases, etc.
        
            Currently just for Keras, will be done by extending `Sequential`
        
        5.  DONE Generate `dense()`
        
            Generate the calling code, currently without any template engine.
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-10-01 Tue 09:58]</span></span>
            The `Enclave` extension of `Sequential` has a `generate_dense()` function that will build the dense function out of matutil functions.
            This is done via format strings, which is not the best solution, but it works for now.

3.  DONE Build custom Keras layer for C code execution

    After the C code is generated, the custom Sequential model should either call that instead of the underlying layers, or replace itself with a new layer.
    That new layer would then either call the Enclave or native C code.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-10-01 Tue 09:59]</span></span>
    This is done by creating a custom Layer that builds a `tf.py_function`, which calls the python function calling my C interop code.

4.  TODO Test on actual hardware <code>[16/21]</code>     :sgx:

    We want to test this on some actual hardware.
    For a representative evaluation we should use one consumer grade, and one server grade CPU.
    According to [this forum answer](https://software.intel.com/en-us/forums/intel-software-guard-extensions-intel-sgx/topic/606636) Intel SGX was introduced with Skylake generation CPUs, so any 6xxx processor will be fine.
    
    1.  DONE Write an e-mail to Manuel
    
    2.  DONE Set up the machine
    
        I have a [setup script](setup_sgx_machine.sh) for Ubuntu 19.04, which I tested on VMs.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-04 Fri 12:00]</span></span>
        The hardware is set up, it only needs a setup of Ubuntu, and network access.
    
    3.  DONE Plan evaluation
    
        1.  DONE Coordinate with RBO how exactly we evaluate     :sgx:
        
            I work out the exact benchmarking details with him.
            He said he wanted to test different cutting points, so I should also add Convolutional layers to the enclave.
            
            Convolutional layers in the front.
            Ideally more than two dense layers in the end.
            We can look at ImageNet structures, which should have more dense layers.
    
    4.  CANCELED Get VGG Face Dataset up and running
    
        This is large enough to merit its own project, and it's currently underway
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-29 Tue 11:55]</span></span>
        We are currently at 229 &#x2026; images out of over 2 6.. &#x2026;, so slightly under 10%.
        We're getting there.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-29 Tue 16:31]</span></span>
        We're still doing VGG face for its own sake, but it's not good enough (with all the missing images) for our current use-case.
    
    5.  DONE Prepare evaluation using TF built-in datasets
    
        TF has a built in, pretrained version of VGG16.
        It also has multiple built-in datasets which can be used for retraining the dense layers.
        This can be used as a preparation for the evaluation.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-15 Tue 11:41]</span></span>
        The tensorflow<sub>datasets</sub> library is a bit weird to use, so for now I'm sticking to a dataset that's downloaded from the tensorflow servers.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-15 Tue 11:48]</span></span>
        Added evaluation scripts for both versions of vgg<sub>flowers</sub>.
        Now only the timing needs to be set up.
    
    6.  DONE Think about good ways to measure inference time
    
        We have one-time setups that take place, which need to be measured.
        There is also additional communication between the GPU and CPU that needs to take place.
        
        We should measure:
        
        -   single inference time, which should include enclave setup
        -   time for a batch predict, so we have a better understanding of how the enclave performs over larger datasets
        -   check if the entire model fits in the 128MB that we have
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-18 Fri 23:10]</span></span>
        This new approach to using TF dataset genartors requires some switchup to the method.
        I will change the evaluation script so it provides a method you can call, providing it the model and the generators.
        It will give me a fixed timing framework for the benchmarking project.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-21 Mon 15:43]</span></span>
        Dataset generators are done.
    
    7.  DONE Add timing code to evaluation script
    
        Probably just a simple timestamp before and after.
        It's probably best to use system time, as using only active CPU time would miss GPU time.
    
    8.  DONE Convert trained VGG flowers model to enclave
    
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-15 Tue 15:35]</span></span>
        The original model with two layers of 4096 neurons is too large to reasonably convert using my current toolchain.
        They are also too large to fit in the enclave without paging.
        I reduced the number of neurons in these layers to 2048 and requeued the training, it's currently enqueued.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-15 Tue 17:14]</span></span>
        I'm currently running into the problem that the weight matrices are just too large.
        I split the first weight matrix into its own file, and tried to compile only that, but the compilation took up all 16GB of RAM available on my machine.
        The next step would be to look into building ELF files by hand, or simply switching to a more powerful machine.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-16 Wed 09:31]</span></span>
        I'm currently looking into what the problem with the state is.
        My intuition says that it's even before the actual compilation.
        To verify this, I'm using clang with the `-emit-ast` option.
        
        If this goes through, then it could be the constant table that's being built during compilation.
        I might be able to turn this off during compilation, but I'm not sure.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-16 Wed 15:28]</span></span>
        I changed the output so it generates binary files containing the weight matrices.
        The code now compiles without issues.
        However, it's still too large for the enclave
    
    9.  DONE Train VGG flowers model
    
        This is a necessary first step before training the VGG face model
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-21 Mon 09:23]</span></span>
        I have a trained model using a GlobalAveragePooling2D layer instead of flattening.
        This greatly reduces the number of trainable weights, which is good because I don't have enough data for the larger number anyway.
        For the much larger VGG face dataset this might be different.
    
    10. DONE Install CUDA and tf-gpu on SGX machine
    
        For a valid benchmark we need to compare it to the GPU version of tensorflow.
    
    11. CANCELED Train a smaller VGG face model with 10 classes
    
        This should make it reasonably close to the VGG flowers model, but still relevant enough for security so we can use it.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-31 Thu 15:23]</span></span>
        The dataset has changed, see [here](#org61d3572)
    
    12. DONE Resolve the problem of too large networks
    
        The network as it is doesn't fit in the enclave.
        I should first look at the `GlobalAveragingLayer` (or similar) in TF.
        This might give us the much needed size reduction.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-16 Wed 15:50]</span></span>
        I just checked the included VGG16 model, whose structure looks like the following:
        
            from tensorflow.keras.applications import VGG16
            VGG16().summary()
        
        As you can see, there are 25088 inputs to the dense part of the network.
        This does require some additional reduction before I can use it.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-21 Mon 09:25]</span></span>
        Rainer suggested adding an additional dense layer before the enclave to reduce dimensionality.
        This might work, but it also reduces the number of neurons hidden from the network host.
        We can probably argue that the shape of the enclave network is irrelevant, only the number of hidden neurons is important.
        Then we can treat the shape of the additional dense layer as a "security parameter" (which it technically isn't but that's ok)
        
        1.  DONE try an `GlobalAveragePooling2D` in vgg<sub>flowers</sub> code
        
            As a hopefully better measure I will try to add a global average pooling layer before the flatten.
            This will greatly reduce dimensionality, and seems to be a SOTA technique.
            Hopefully tests from the flower dataset generalize to the face dataset.
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-10-21 Mon 09:28]</span></span>
            This does work well on the flowers dataset.
            More tests with the face dataset coming.
    
    13. DONE Get MS face dataset
    
        There is an academic torrent for the [MS-Celeb-1M](http://academictorrents.com/details/9e67eb7cc23c9417f39778a8e06cca5e26196a97) dataset.
        While this isn't VGG face it is a face recognition database by Microsoft, so benchmarking on it seems fine.
        We can use this in case the VGG face download doesn't work out fast enough.
    
    14. CANCELED Extract images from MS face dataset
    
        The images in the MS dataset are contained as base64 blobs (I think).
        Some assembly required&#x2026;
        
        1.  DONE Get more disk space for the sgx machine
        
            Downloading the celebrity dataset requires more disk space than the test machine currently has.
            I will talk to Manuel about an additional HDD.
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-10-21 Mon 12:08]</span></span>
            Manuel gave me an external 3TB hard drive.
            An internal one is already ordered, it will be here this week.
        
        2.  DONE Check the disk space on the cluster
        
            I'm not sure if I could move the entire dataset to the cluster, which would mean I can't train there.
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-10-21 Mon 09:34]</span></span>
            The main disk has 457G of space, so moving the large dataset there is probably not prudent.
            There is an HDD raid, but I don't have permissions to access that at the moment.
            I should ask the admins about it.
        
        3.  CANCELED Ask about getting access to the HDD Raid on the cluster
        
            This would be the best way IMO to use large datasets on the cluster.
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-10-21 Mon 09:58]</span></span>
            The resized dataset is not very large anymore, so this isn't really necessary at the moment
    
    15. DONE Talk to Rainer about the reduced class subset of VGG face     :sgx:
    
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-29 Tue 08:56]</span></span>
        
        We currently have 222 classes (221 of which should downloaded as much as possible)
        This seems like a good starting point for a classifier
    
    16. DONE Get ANY dataset working
    
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-29 Tue 17:32]</span></span>
        [This](https://megapixels.cc/datasets/msceleb/) is a pretty damning report about the MS Face dataset.
        We could still use it, but we should be aware of the things happening around it.
        It also links to [this workshop](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop/), where ther is a download link for a reduced and cleaned up dataset extracted from it.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-29 Tue 17:52]</span></span>
        The labeled faces in the wild dataset is contained in `tensorflow_datasets` and seems to work.
        There is a git repo [here](https://github.com/davidsandberg/facenet) which implements the model from [this paper](https://arxiv.org/pdf/1503.03832.pdf).
        However, the author of the repo doesn't seem to be affiliated with the authors of the paper.
        LFW in general at least seems like a promising direction.
        
        <a id="org61d3572"></a>
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-30 Wed 17:08]</span></span>
        I started training a classifier on LFW
    
    17. TODO Get NN above 80% on a 100 class image recognition task
    
        The images should be 224x224 (or larger), and the convolutional part should be based on VGG (or ResNet).
        I'm currently working the 15 to 115 most common classes in the LFW dataset, the model is still running.
        My best result to date is 65%.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-11-05 Tue 16:15]</span></span>
        I just fixed a bug in the dataset generation that breaks everything.
        I'll let the currently running instance finish, but will schedule a new VGG and ResNet run.
        Hopefully there will be better results tomorrow.
        
        1.  VGG-LFW
        
            I pick the 100 most common classes, dropping the most common class.
            The most common class is dropped to somewhat reduce variation in the prior probabilities (the most common class has over 500 samples, the next has 100).
            
            1.  "Baseline"
            
                The current baseline has 4096 hidden neurons, and is trained for 100 epochs.
                It reached 61% once, but I since then changed some dataset stuff, so I'm not sure if that is correct anymore.
                
                1.  WATING Retrain baseline model
                
                    I changed some dataset generation code, I should rerun the baseline to still have an accurate and current baseline.
                    
                    <span class="timestamp-wrapper"><span class="timestamp">[2019-11-06 Wed 12:56]</span></span>
                    Just started, it's job 11908.
            
            2.  Larger variants
            
                I tried increasing the number of hidden neurons and increasing the number of training epochs.
                Both variants slightly increased the performance (larger 62%, longer 62%, all combined 65%).
                However, that doesn't seem like the right thing to do, because my results already look like they're overfitting too much.
            
            3.  Smaller variants
            
                I will also try running smaller variants, one with only 50 training epochs, and one with only 512 hidden neurons.
                We'll see how that goes.
                
                1.  DONE Train 50 epoch model
                
                    It's job 11909.
                    This is the log output:
                    
                        Hypeparameters:
                        num_epochs: 50
                        hidden_neurons: 4096
                        training set size: 2822
                        test set size: 313
                        
                        Epoch 1/50
                        40/40 [==============================] - 144s 4s/step - loss: 4.5281 - acc: 0.0570 - val_loss: 4.3582 - val_acc: 0.0641
                        Epoch 2/50
                        40/40 [==============================] - 148s 4s/step - loss: 4.2580 - acc: 0.0859 - val_loss: 4.1780 - val_acc: 0.1070
                        Epoch 3/50
                        40/40 [==============================] - 149s 4s/step - loss: 4.0062 - acc: 0.1203 - val_loss: 3.8714 - val_acc: 0.0992
                        Epoch 4/50
                        40/40 [==============================] - 149s 4s/step - loss: 3.6143 - acc: 0.1773 - val_loss: 3.4942 - val_acc: 0.1609
                        Epoch 5/50
                        40/40 [==============================] - 150s 4s/step - loss: 3.2635 - acc: 0.2164 - val_loss: 3.2345 - val_acc: 0.2250
                        Epoch 6/50
                        40/40 [==============================] - 150s 4s/step - loss: 2.9261 - acc: 0.2773 - val_loss: 2.8594 - val_acc: 0.2711
                        Epoch 7/50
                        40/40 [==============================] - 150s 4s/step - loss: 2.6633 - acc: 0.3094 - val_loss: 2.6242 - val_acc: 0.3313
                        Epoch 8/50
                        40/40 [==============================] - 150s 4s/step - loss: 2.4841 - acc: 0.3539 - val_loss: 2.6124 - val_acc: 0.3320
                        Epoch 9/50
                        40/40 [==============================] - 151s 4s/step - loss: 2.3263 - acc: 0.3844 - val_loss: 2.4848 - val_acc: 0.3469
                        Epoch 10/50
                        40/40 [==============================] - 151s 4s/step - loss: 2.1146 - acc: 0.4156 - val_loss: 2.3971 - val_acc: 0.3602
                        Epoch 11/50
                        40/40 [==============================] - 150s 4s/step - loss: 2.0584 - acc: 0.4453 - val_loss: 2.3637 - val_acc: 0.3633
                        Epoch 12/50
                        40/40 [==============================] - 150s 4s/step - loss: 1.8657 - acc: 0.4766 - val_loss: 2.2422 - val_acc: 0.3562
                        Epoch 13/50
                        40/40 [==============================] - 151s 4s/step - loss: 1.8343 - acc: 0.4961 - val_loss: 2.2244 - val_acc: 0.4187
                        Epoch 14/50
                        40/40 [==============================] - 151s 4s/step - loss: 1.7056 - acc: 0.5188 - val_loss: 2.0923 - val_acc: 0.4570
                        Epoch 15/50
                        40/40 [==============================] - 150s 4s/step - loss: 1.6859 - acc: 0.5211 - val_loss: 2.0829 - val_acc: 0.4398
                        Epoch 16/50
                        40/40 [==============================] - 150s 4s/step - loss: 1.5415 - acc: 0.5594 - val_loss: 2.1350 - val_acc: 0.4437
                        Epoch 17/50
                        40/40 [==============================] - 150s 4s/step - loss: 1.4506 - acc: 0.5813 - val_loss: 2.1626 - val_acc: 0.4258
                        Epoch 18/50
                        40/40 [==============================] - 151s 4s/step - loss: 1.4247 - acc: 0.5961 - val_loss: 2.1133 - val_acc: 0.4453
                        Epoch 19/50
                        40/40 [==============================] - 151s 4s/step - loss: 1.3201 - acc: 0.6086 - val_loss: 1.9491 - val_acc: 0.4547
                        Epoch 20/50
                        40/40 [==============================] - 150s 4s/step - loss: 1.2787 - acc: 0.6336 - val_loss: 1.8885 - val_acc: 0.4844
                        Epoch 21/50
                        40/40 [==============================] - 150s 4s/step - loss: 1.1745 - acc: 0.6562 - val_loss: 1.8885 - val_acc: 0.4906
                        Epoch 22/50
                        40/40 [==============================] - 150s 4s/step - loss: 1.1080 - acc: 0.6695 - val_loss: 2.1113 - val_acc: 0.4289
                        Epoch 23/50
                        40/40 [==============================] - 150s 4s/step - loss: 1.0706 - acc: 0.6922 - val_loss: 2.1356 - val_acc: 0.4555
                        Epoch 24/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.9893 - acc: 0.7039 - val_loss: 1.8050 - val_acc: 0.5078
                        Epoch 25/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.9377 - acc: 0.7312 - val_loss: 1.8422 - val_acc: 0.5281
                        Epoch 26/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.8181 - acc: 0.7578 - val_loss: 1.8654 - val_acc: 0.4914
                        Epoch 27/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.8668 - acc: 0.7352 - val_loss: 1.8883 - val_acc: 0.4805
                        Epoch 28/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.7349 - acc: 0.7820 - val_loss: 1.8898 - val_acc: 0.5492
                        Epoch 29/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.7134 - acc: 0.7922 - val_loss: 1.7778 - val_acc: 0.5297
                        Epoch 30/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.6433 - acc: 0.8062 - val_loss: 1.8707 - val_acc: 0.5250
                        Epoch 31/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.6419 - acc: 0.8180 - val_loss: 1.8195 - val_acc: 0.5273
                        Epoch 32/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.5811 - acc: 0.8273 - val_loss: 1.7483 - val_acc: 0.5508
                        Epoch 33/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.6008 - acc: 0.8180 - val_loss: 1.8984 - val_acc: 0.5188
                        Epoch 34/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.5169 - acc: 0.8484 - val_loss: 2.0376 - val_acc: 0.5078
                        Epoch 35/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.5090 - acc: 0.8469 - val_loss: 2.0583 - val_acc: 0.5086
                        Epoch 36/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.4752 - acc: 0.8578 - val_loss: 2.1739 - val_acc: 0.5148
                        Epoch 37/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.4264 - acc: 0.8805 - val_loss: 1.9400 - val_acc: 0.5469
                        Epoch 38/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.4108 - acc: 0.8836 - val_loss: 1.8909 - val_acc: 0.5414
                        Epoch 39/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.3942 - acc: 0.8797 - val_loss: 2.1462 - val_acc: 0.5172
                        Epoch 40/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.3634 - acc: 0.9000 - val_loss: 1.8605 - val_acc: 0.5367
                        Epoch 41/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.3340 - acc: 0.9148 - val_loss: 2.0350 - val_acc: 0.5461
                        Epoch 42/50
                        40/40 [==============================] - 151s 4s/step - loss: 0.3156 - acc: 0.9187 - val_loss: 2.1340 - val_acc: 0.5320
                        Epoch 43/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.3115 - acc: 0.9109 - val_loss: 2.2825 - val_acc: 0.5016
                        Epoch 44/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.2939 - acc: 0.9219 - val_loss: 1.8770 - val_acc: 0.5539
                        Epoch 45/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.2446 - acc: 0.9383 - val_loss: 2.4037 - val_acc: 0.5242
                        Epoch 46/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.2757 - acc: 0.9211 - val_loss: 2.1594 - val_acc: 0.5500
                        Epoch 47/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.2746 - acc: 0.9289 - val_loss: 2.2189 - val_acc: 0.5375
                        Epoch 48/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.2381 - acc: 0.9266 - val_loss: 2.0340 - val_acc: 0.5664
                        Epoch 49/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.2680 - acc: 0.9195 - val_loss: 1.8832 - val_acc: 0.5969
                        Epoch 50/50
                        40/40 [==============================] - 150s 4s/step - loss: 0.2057 - acc: 0.9469 - val_loss: 2.0120 - val_acc: 0.5773
                        40/40 [==============================] - 74s 2s/step - loss: 1.9711 - acc: 0.5797
                        loss: 1.97
                        accuracy: 0.58
                    
                    It reaches 58% accuracy for the test set, 94% for the training set.
                
                2.  TODO Train 512 neuron model
                
                    It's job 11911.
                    
                    <span class="timestamp-wrapper"><span class="timestamp">[2019-11-06 Wed 13:11]</span></span>
                    The third job that I schedule keeps on failing, I'll just have to wait until the 50 epoch or baseline model are done.
                    
                    <span class="timestamp-wrapper"><span class="timestamp">[2019-11-06 Wed 15:40]</span></span>
                    Just started it again, job 11917 on the cluster.
    
    18. TODO Get Ember dataset working
    
        We now have a working image dataset for benchmarking.
        Ideally we get a working second media type that also uses CNNs working, for better results.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-11-05 Tue 16:01]</span></span>
        I started on working with the Ember dataset.
        The features are extracted from the raw data, I just don't know that they mean yet&#x2026;
    
    19. TODO Train classifier on Benchmark Data
    
        Currently one of the benchmark sets is the 100 most common classes in labelled faces in the wild (LFW).
        We also need a second dataset, maybe something on malware
        
        1.  TODO Test different vgg<sub>lfw</sub> variants     :sgx:
        
            <span class="timestamp-wrapper"><span class="timestamp">[2019-11-04 Mon 08:35]</span></span>
            
            The baseline model reaches 61% validation accuracy after 100 epochs.
            See if this can be improved via:
            
            -   increasing training epochs: 62%
            -   dense on flattened: 63%
            -   globalmean with more dense neurons: 62%
            -   more dense neurons on flattened:
    
    20. TODO Measure performances for multiple network types
    
        enclave vs. GPU for multiple cutoff points
        Test with enclave and with native C code
        
        MNIST, one more type of data
        
        1.  DONE get VGG16 model
        
            There is one included within `keras.applications`.
            I think we should use it, with standard TF datasets.
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-10-15 Tue 08:53]</span></span>
            We are probably using that network, with the pre-trained weights.
            However the end goal will be to evaluate it on the VGG face datset, when that is available to us.
    
    21. TODO Plan privacy evaluation
    
        more on this later

5.  TODO Add capability for convolutional layers     :sgx:

6.  TODO Test the splitting with several popular architectures     :sgx:

    We should test the splitting with several common architectures.
    RBO also said something about splitting according to some metrics (e.g. GPU memory/utilization), which I'm not too sure about how he meant it.
    We could try and split before the first dense layer, I should try and see if there are other layer types used for actual classification.

7.  TODO Automate memory layout inside SGX     :sgx:

    We might have to do some memory magic because otherwise we might run out of memory inside the SGX.
    The first prototype can do this explicitly for the chosen network, but for publishing we should do this automatically.
    
    Alternative we could also generate a fixed function from sparse matrix multiplication.
    This would mean that we go through the output cell by cell, calculating all immediate steps in a row.
    Using this would throw away any shared results, and be much slower.
    However, this could help us avoid memory issues.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-09-25 Wed 10:56]</span></span>
    This has to wait until we are testing on actual hardware.
    For the simulation mode we are currently running, memory doesn't seem to be an issue.
    I'm currently specifying intermediate matrices after every `fc` layer of the network (see [the enclave code](lib/sgx/Enclave/Enclave.cpp)), without any issues.
    
    This either means we are not as memory constrained as we thought, or the simulation simply has a larger amount of memory than the actual enclave would have.
    
    <span class="timestamp-wrapper"><span class="timestamp">[2019-10-08 Tue 16:09]</span></span>
    The enclave has 100MB of memory.
    This might require some trickery for full models.
    I could try and use only two temporary buffers, to reduce memory requirements.

8.  TODO Anwendungsfälle auflisten     :sgx:

    <span class="timestamp-wrapper"><span class="timestamp">[2019-10-04 Fri 11:36]</span></span>
    
    1.  Model stealing
    
        1.  DONE Find some papers on model stealing     :sgx:
    
    2.  Offline ML-as-a-service <a id="orgc255e46"></a>
    
        Making content of `dense()` and network weights independent of launch code would allow MLAAS providers to sign an execution environment once, and then give signed and encrypted weights and architectures to their customers.
        This would require decrypting the models and dynamically generating the `dense()` function.
        
        By doing so we can move MLAAS from the cloud to offline, allowing for easier use, at least on desktop devices.
        While this doesn't necessarily work for mobile devices, it would allow users of MLAAS services to run the models themselves.
        
        The resulting classification could even be signed together with the input, thus verifying that the classification came from a certain model.
        **This would add accountability to every single classification.**
        
        Maybe we can spin an additional paper from this.
    
    3.  Oblivious distributed machine learning
    
        Along the lines of [this](#orgc255e46) and distributed SGD, we can do the same thing the other way round.
        Companies could send signed and encrypted inputs, together with model updates to people offering their compute power.
        The people then train their local model inside the enclave, so they can't steal the model.
        The code in the enclave then signs and encrypts the weight updates and sends them back to the company.
        
        The local devices would all use the same signature if they are provided with the same enclave code, so they wouldn't be identifiable from their updates alone.
        This can be cited from [Shokri ML privacy](related_work/shokri15privacy.pdf), the learning in general would be similar to [Downpour](related_work/dean2012large.pdf).
        
        This would make an **awesome** paper.
    
    4.  Privacy -> Set Membership
    
        I should do the math where the total attack accuracy for only returning the label comes from in the set membership paper.
        Other than that we still have the advantage of making an offline model as robust as an online oracle.
        
        1.  TODO Leakage confinement
        
            Find out if the leakage information is constrained to the dense layers.
            Figure -> dense weights before and after retraining
            
            <span class="timestamp-wrapper"><span class="timestamp">[2019-10-31 Thu 15:29]</span></span>
            I mean it's contained in the dense layers if you keep it there.
            What would be interesting is how much the convolutional layers change if you don't fix them.
            Here a delta per individual weight would be interesting.
            
            A comparison of accuracies on the same dataset with and without freezing the convolutional layers would also be beneficial.
            This relates to [transfer learning](#org479f675).
        
        2.  TODO Run on Cecilias face classifier
        
            Retraining, then try and see where the differences are during retraining.
            Evaluate set membership.
    
    5.  TODO consider possible new attacks on partial black boxes
    
        An attacker might use the early convolutional layers to reduce the search space for adversarial examples.
        This needs to be in the final paper.
    
    6.  Lastverteilung zw. Edge, Cloud
    
        1.  TODO Find some papers on load balancing for distributed heterogenous architectures     :sgx:
        
            Or talk to DPS groups

9.  Read related work <code>[5/6]</code>     :sgx:

    <span class="timestamp-wrapper"><span class="timestamp">[2019-10-03 Thu 10:15]</span></span>
    
    1.  DONE Data privacy:
    
        One is [on privacy preserving deep learning (leakage)](file:///home/alex/Projects/nn-sgx/related_work/shokri15privacy.pdf), which provides a way to perform SGD distributed over multiple nodes.
        By not sending all state updates to the state server the participating parties retain privacy of their dataset.
        This is cool in the sense that it solves a problem that we could also try and solve with our method (which would give us [the ohmirenko paper](related_work/ohrimenko16enclave.pdf) exactly).
        Other than that there's not too much in common.
        
        The other Shokri paper is [on membership attacks](file:///home/alex/Projects/nn-sgx/related_work/shokri17membership.pdf).
        However, in my opininion this doesn't have too much to do with our case.
        The case for k=1 label, however is weird.
        Just guessing gives the correct accuracy, but precision and recall are a bit weird.
        I should talk to Rainer about this
        
        1.  DONE Talk to Rainer about the k=1 label case for set membership attacks
    
    2.  DONE Enclave Security     :sgx:
    
        [This paper](related_work/kaptchuk2019state.pdf) by Kaptchuk et al. uses ledgers to add state to the enclave, which guarantees that every state is only presented to it once.
        
        We could use this mechanism to implement rate limiting on our models.
        This would mean they aren't truly offline anymore, but I guess that's the security/usability tradeoff we have to face.
    
    3.  DONE NNs in TEEs
    
        Dan Boneh released [SLALOM](related_work/tramer19slalom.pdf) with Florian Tramer, talking about something very similar to what we're doing.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-15 Tue 10:50]</span></span>
        They do talk about something similar to what we're doing, but they're doing it very differently.
        Instead of running parts of the NN in the enclave, they run it on the GPU and verify its outputs in the enclave.
        Also they apply some encryption mechanism to the input data to keep it private from the model owner.
        
        The notion of verifying outputs that come from the GPU is very cool, and we should look into something similar for our project.
        Encrypting the input might also be a posibility for future work.
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-26 Sat 19:10]</span></span>
        In 2016 Ohrimenko et al. released a paper on [training different ML flavors in the enclave](related_work/ohrimenko16enclave.pdf).
        Their focus is defending against power analysis attacks to keep the training data private from some third party datacenter.
        Not entirely related to our method.
        
        Also what they do requires the entire training process to happen in the enclave, which is a lot slower than training on GPUs, which we could do.
    
    4.  DONE Model stealing
    
        A good paper on model stealing seems to be [this one](related_work/tramer16stealing.pdf) by Tramer et al., which Daniel recommended to me.
        The query budget they set for their attack is `100*k`, where `k` is the number of parameters.
        The dense part of our model has 18903.045 parameters, so that wouldn't be feasible.
    
    5.  TODO Find something on accountability for NNs
    
        <span class="timestamp-wrapper"><span class="timestamp">[2019-10-26 Sat 17:58]</span></span>
        What I found is more about accountability for [automatically generated textual reports](http://merrill.umd.edu/wp-content/uploads/2016/01/p56-diakopoulos.pdf).
        A [blog post](https://officialblogofunio.com/2019/04/29/a-short-introduction-to-accountability-in-machine-learning-algorithms-under-the-gdpr/
        ) (that is heavily focused on law) talks about accountability of data processing under GDPR.
        One could think of a use case where a model is trained, tested for differential privacy and after passing that testing it is approved and signed.
        The legislator could then co-sign the code running in the enclave, like a certificate.
        Whoever runs the model then can provide proof that the model they are running is differentially private, if that model signs it's outputs.
    
    6.  DONE Layer freezing in transfer learning
    
        Cecilia pointed me towards transfer learning.
        There should be some papers on this especially in the context of layer freezing.
        I should talk to her next week
        
        <span class="timestamp-wrapper"><span class="timestamp">[2019-11-04 Mon 09:24]</span></span>
        Cecialia sent me an email with resources, I should check them out.
        
        1.  TODO Talk to Cecilia about Layer Freezing
        
            After me reading some literature we might be able to have a good discussion.


<a id="org9606181"></a>

### Future Work <code>[0/2]</code>

1.  TODO Integrate with the framework API     :sgx:

    Rainer said it would be nice to integrate the SGX with the tensorflow API (or pytorch, whatever)

2.  TODO Examine required query counts for transfer and full model stealing attacks

    Transfer attacks require much less precise replications of the target model.
    It would be interesting to see how the query numbers differ for actual model stealing and transfer attacks.


<a id="orgb94c082"></a>

## Unforeseen events


<a id="orgce9c5db"></a>

### DONE Figure out how many bits the enclave uses for floats

This could cause some weird results and incompatibilities.
It's also not perfectly clear if the enclave even supports floats.

<span class="timestamp-wrapper"><span class="timestamp">[2019-10-04 Fri 10:10]</span></span>
The enclave supports floats.
In all my tests there have been rounding errors (larger than the `numpy` default of 10<sup>-8</sup>), but the results are identical.
I think this is good enough.


<a id="org3ac2247"></a>

### DONE Find out what is needed for the foreshadow mitigation     :sgx:

  <span class="timestamp-wrapper"><span class="timestamp">[2019-10-04 Fri 10:02]</span></span>
I should probably read [the paper](https://foreshadowattack.eu/foreshadow.pdf).
According to [the attack's website](https://foreshadowattack.eu/) (what a world we live in), under the headline "Are there mitigations against Foreshadow?", mitigations require software and microcode updates.
I would assume the software updates are for changing keys (which could have been leaked by vulnerable systems) and resealing enclave code.

The microcode is for patching the vulnerability.
Intel released a [security advisory](https://www.intel.com/content/www/us/en/security-center/advisory/intel-sa-00161.html) on foreshadow, as well as a summary of microcode updates.
This means that having currenty microcode updates on our benchmark machines would make them resistant against foreshadow.
At least as resistant as we can currently get.


<a id="org1905900"></a>

### TODO Test if our processors are vulnerable to foreshadow     :sgx:

<span class="timestamp-wrapper"><span class="timestamp">[2019-10-04 Fri 11:46]</span></span>

If they are, don't update, we want to measure performance impact of that 
Updating is probably `irreversible`, so we should find out if they are vulnerable before installing any microcode packages.

