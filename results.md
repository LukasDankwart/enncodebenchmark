# Experimental setup 
eNNCode vs. OMLT (v1.2.2 vs Gurobi Machine Learning (v1.5.5)

Please note that we had a different pseudonym for eNNcode, called 'ONNXGurobi' during this experimental phase.
We have not yet adjusted this aspect in this setup, for example due to the path specifications. Therefore, we ask that 
ONNXGurobi scripts and results be interpreted accordingly as those of our eNNcode library.

## Evaluation 

Our scripts for the evaluation can be found in "ctf_{baseline}_{network}" respectively.
The four networks used in our evaluation, which is displayed in our paper, are:
- models/mnist/conv_net.onnx
- models/mnist/fc_mnist.onnx
- models/datasets/trained/classifier_medium_concrete.onnx
- models/datasets/trained/classifier_medium_wine.onnx


The results are stored in "results/final/". 


## Compatibility on VNN-Comp 23 networks

Following networks were only checked for compatibility comparison between OMLT and eNNcode in the early experimental phase.

## 1. Acasxu

- ONNXGurobi: see Results/building_onnxgurobi_acasxu.txt
- OMLT: ValueError: Unhandled node type sub

## 2. cctsdb_yolo

- ONNXGurobi: NotImplementedError: Parser for operator 'Slice' is not supported.
- OMLT: KeyError: ''

## 3. cgan

- ONNXGurobi: NotImplementedError: Parser for operator 'ConvTranspose' is not supported.
- OMLT: ValueError: Unhandled node type BatchNormalization

## 4. collins_rul_cnn

- ONNXGurobi: see results/bulding_onnxgurobi_collins_rul_cnn.txt
- OMLT: ValueError: Unhandled node type Dropout

## 5. collins_yolo_robustness

- ONNXGurobi: NotImplementedError: Parser for operator 'LeakyRelu' is not supported.
- OMLT: has non-zero pads ([2, 2, 2, 2]). This is not supported.

## 6. dist_shift

- ONNXGurobi: NotImplementedError: Parser for operator 'Sigmoid' is not supported.
- OMLT: KeyError: 'onnx::Reshape_13'

## 7. metaroom (modified with batch dim)

- ONNXGurobi: see results/building_onnxgurobi_metaroom.txt
- OMLT:  has non-zero pads ([1, 1, 1, 1]). This is not supported.

## 8. ml4acopf

- ONNXGurobi: NotImplementedError: Parser for operator 'Slice' is not supported.
- OMLT: ValueError: Unhandled node type Slice

## 9. nn4sys (both compatible)

- ONNXGurobi: see results/building_onnxgurobi_nn4sys.txt
- OMLT: see results/building_omlt_nn4sys.txt

## 10. tllverifybench (both compatible)

- ONNXGurobi: see results/building_onnxgurobi_tllverifybench.txt
- OMLT: see results/building_omlt_tllverifybench.txt

## 11. traffic_signs_recognition

- ONNXGurobi: NotImplementedError: Parser for operator 'Transpose' is not supported.
- OMLT: ValueError: Unhandled node type Transpose

## 12. vit:

- ONNXGurobi: NotImplementedError: Parser for operator 'Shape' is not supported.
- OMLT: KeyError: '/0/Concat_output_0'

## 13. yolo:

- ONNXGurobi: NotImplementedError: Parser for operator 'Pad' is not supported.
- OMLT: ValueError: Input/output size ([1, 3, 52, 52]) must have one more dimension than initialized kernel shape ([3, 3]).

