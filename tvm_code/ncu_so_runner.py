#!/usr/bin/python3
import tvm
import onnx
import numpy as np

from google.protobuf.json_format import MessageToDict

from tvm.runtime.module import load_module

DEVICE = tvm.cuda()
TARGET = tvm.target.cuda(arch='sm_70' , options="-max_threads_per_block=1024 -max_shared_memory_per_block=96000")

def get_input_node_info(onnx_model):
    # TVM from_onnx() requires shape_dict to be a dictionary of node name: List of dimensions
    shape_dict = {}
    input_name = ""
    DTYPE = ""
    input_shape = []
    
    for _input in onnx_model.graph.input:
        # ONNX format returns graph nodes as protobuf object
        m_dict = MessageToDict(_input)
        print("input_name : ", m_dict['name'])
        print("input_shape: ", m_dict["type"]["tensorType"]['shape'])
        print("input_dtype: ", m_dict["type"]["tensorType"]['elemType'])
        dim_info = m_dict["type"]["tensorType"]["shape"]["dim"]
        input_shape = [int(d.get("dimValue")) for d in dim_info]
        input_name = m_dict["name"]
        shape_dict[input_name] = input_shape
        
        # TODO: Convert enum elemType to required datatype
        DTYPE = "float32" if m_dict["type"]["tensorType"]['elemType'] == 1 else "float32"
        
    return shape_dict, input_name, input_shape, DTYPE


if __name__=="__main__":
    model_name = "relu"
    onnx_path = "../cases/candidate_cases/" + model_name + ".onnx"

    NUM_RUNS = 100
    MIN_REPEAT_MS = 500
    
    # Get input tensor shape from onnx graph
    onnx_model = onnx.load(onnx_path)
    shape_dict, input_name, input_shape, DTYPE = get_input_node_info(onnx_model)

    # Create random data
    data = tvm.nd.array(np.random.randn(*input_shape).astype(DTYPE), DEVICE)

    # Load .so object 
    loaded_mod = load_module("./tune_kernels/lib_export/relu.so")
    out_tvm = tvm.nd.empty(input_shape, device=DEVICE)
    loaded_mod(data, out_tvm)

    evaluator = loaded_mod.time_evaluator(loaded_mod.entry_name, DEVICE, repeat=NUM_RUNS, min_repeat_ms=MIN_REPEAT_MS)
