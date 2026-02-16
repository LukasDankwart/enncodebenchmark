# ====== Params for Counterfactuals ==========

TIME_LIMIT = 240
MIPFOCUS = 2
ITERATIONS = 20
MNIST_LB = 0.0
MNIST_UB = 1.0
CLASSIFIERS_LB = 0.0
CLASSIFIERS_UB = 1.0

# ===== Params for Build ======
BUILD_ITERATIONS = 20



# ====== PATHS ==========

# FC MNIST
FC_MNIST_PATH = "models/mnist/fc_mnist.onnx"
FC_MNIST_PATH_BOUNDS = "models/mnist/fc_mnist_with_bounds.onnx"
FC_MNIST_SHAPE = (1, 784)

# CONV MNIST
CONV_MNIST_PATH = "models/mnist/conv_net.onnx"
CONV_MNIST_PATH_BOUNDS = "models/mnist/conv_net_with_bounds.onnx"
CONV_MNIST_SHAPE = (1, 28, 28)

# nn4sys
NN4SYS_PATH = "models/nn4sys/lindex_deep.onnx"
NN4SYS_PATH_BOUNDS = "models/nn4sys/lindex_deep_with_bounds.onnx"
NN4SYS_SHAPE = (1, 1)


# tllverifybench
TLLVERIFYBENCH_PATH = "models/tllverifybench/tllBench_n=2_N=M=16_m=1_instance_1_0.onnx"
TLLVERIFYBENCH_PATH_BOUNDS = "models/tllverifybench/tllBench_n=2_N=M=16_m=1_instance_1_0_with_bounds.onnx"
TLLVERIFYBENCH_SHAPE = (1, 2)


