#%%
import torch
import onnx
from onnx.backend.test.case.node import expect
from onnx import AttributeProto as ap
from onnx import TensorProto as tp
import numpy as np
import onnxruntime as ort
import onnx_graphsurgeon as OGS

from onnx import helper, shape_inference
import copy

import os
import sys

from google.protobuf.json_format import MessageToDict

# Reference: https://github.com/NVIDIA/TensorRT/blob/master/tools/onnx-graphsurgeon/examples/08_replacing_a_subgraph/replace.py
#%%
def create_test_maxpool2dx3_subgraph():
    tensor_shape = [1,512,13,13]
    output_tensor_shape = copy.deepcopy(tensor_shape)
    output_tensor_shape[1] *= 3
    
    a = OGS.Variable("a", shape=tensor_shape, dtype=np.float32)
    
    output_0 = OGS.Variable("output_0", shape=tensor_shape, dtype=np.float32)
    output_1 = OGS.Variable("output_1", shape=tensor_shape, dtype=np.float32)
    output_2 = OGS.Variable("output_2", shape=tensor_shape, dtype=np.float32)
    concat_output = OGS.Variable("concat_output", shape=output_tensor_shape, dtype=np.float32)
    
    maxpool_node_0 = OGS.Node(
                op = "MaxPool",
                inputs=[a],
                outputs=[output_0],
                attrs={
                        "kernel_shape": [5, 5],
                        "strides": [1, 1],
                        "pads": [2, 2, 2, 2]
                       },
                name="MaxPool_0"
            )

    maxpool_node_1 = OGS.Node(
                op = "MaxPool",
                inputs=[a],
                outputs=[output_1],
                attrs={
                        "kernel_shape": [9, 9],
                        "strides": [1, 1],
                        "pads": [4, 4, 4, 4]
                       },
                name="MaxPool_1"
            )

    maxpool_node_2 = OGS.Node(
                op = "MaxPool",
                inputs=[a],
                outputs=[output_2],
                attrs={
                        "kernel_shape": [13, 13],
                        "strides": [1, 1],
                        "pads": [6, 6, 6, 6]
                       },
                name="MaxPool_2"
            )
    
    concat_node = OGS.Node(
                op="Concat",
                inputs=[output_0, output_1, output_2],
                outputs=[concat_output],
                attrs={
                    "axis": 1 # Concatenate along the channel dimension
                },
                name="Concat"
            )
        
    OGS_graph = OGS.Graph(nodes=[maxpool_node_0, maxpool_node_1, maxpool_node_2, concat_node], inputs=[a], outputs=[concat_output])
    onnx_graph = OGS.export_onnx(OGS_graph)
    onnx_graph_shape_inferred = shape_inference.infer_shapes(onnx_graph)
    
    return onnx_graph_shape_inferred
    
#%%
if __name__=="__main__":
    model_path = "./onnx_graph/maxpool2dx3_subgraph.onnx"
    maxpool_subgraph_onnx = create_test_maxpool2dx3_subgraph()
    
    runtime_inputs = {}
    
    # input tensors for softmax_subgraph.onnx
    a = torch.Tensor(np.random.rand(1,512,13,13))

    # input tensors for softmax_subgraph.onnx
    runtime_inputs[maxpool_subgraph_onnx.graph.input[0].name] = a.numpy()
    
    onnx.save(maxpool_subgraph_onnx, model_path)
    
    ort_sess_orig = ort.InferenceSession(model_path)
    outputs_onnx = ort_sess_orig.run(None, runtime_inputs)