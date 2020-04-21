import yaml
import torch
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

from model import get_model
from dataset import get_dataloader
from metric import get_metric
import torch.onnx
from onnx_coreml import convert
import onnx
from coremltools.proto import NeuralNetwork_pb2

INT_MAX = 2**63 - 1
upsampleTargetSizes = {'383': 8, '415': 16, '447': 32, '479': 64, '511': 64}
def _convert_upsample(builder, node, graph, err):
    params = NeuralNetwork_pb2.CustomLayerParams()
    params.className = node.op_type
    params.description = "Custom layer that corresponds to the ONNX op {}".format(node.op_type)

    target_height = int(upsampleTargetSizes[node.name])
    target_width = int(upsampleTargetSizes[node.name])

    builder.add_resize_bilinear(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        target_height=target_height,
        target_width=target_width,
        mode='UPSAMPLE_MODE'
    )

def load_input_constants(builder, node, graph, err):
    for i in range(len(node.inputs)):
        if node.inputs[i] in node.input_tensors and node.inputs[i] not in graph.constants_loaded:
            value = node.input_tensors[node.inputs[i]]
            builder.add_load_constant_nd(
                name=node.name + '_load_constant_' + str(i),
                output_name=node.inputs[i],
                constant_value=value,
                shape=[1] if value.shape == () else value.shape
            )
            graph.constants_loaded.add(node.inputs[i])

def _convert_tile(builder, node, graph, err):
    '''
    convert to CoreML Tile Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L5117
    '''
    load_input_constants(builder, node, graph, err)
    # if node.inputs[1] not in node.input_tensors:
    #    err.unsupported_op_configuration(builder, node, graph, "CoreML Tile layer does not support dynamic 'reps'. 'reps' should be known statically")
    builder.add_tile(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        reps=[1]
    )

def _convert_slice_v9(builder, node, graph, err):
    '''
    convert to CoreML Slice Static Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L5082
    ''' 
    data_shape = graph.shape_dict[node.inputs[0]]
    len_of_data = len(data_shape)
    begin_masks = [True] * len_of_data
    end_masks = [True] * len_of_data

    default_axes = list(range(len_of_data))
    default_steps = [1] * len_of_data
    
    ip_starts = node.attrs.get('starts')
    ip_ends = node.attrs.get('ends')
    axes = node.attrs.get('axes', default_axes)
    steps = node.attrs.get('steps', default_steps)

    starts = [0] * len_of_data
    ends = [0] * len_of_data

    for i in range(len(axes)):
        current_axes = axes[i]
        starts[current_axes] = ip_starts[i]
        ends[current_axes] = ip_ends[i]
        if ends[current_axes] != INT_MAX or ends[current_axes] < data_shape[current_axes]:
            end_masks[current_axes] = False

        if starts[current_axes] != 0:
            begin_masks[current_axes] = False

    builder.add_slice_static(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        begin_ids=starts,
        end_ids=ends,
        strides=steps,
        begin_masks=begin_masks,
        end_masks=end_masks
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exper_folder', help='Provide experiment folder')
    parser.add_argument('dataset', help='Provide dataset')
    args = parser.parse_args()
    
    print('Evaluation {} started'.format(args.exper_folder))
    
    config_file = 'experiment/' + args.exper_folder + '/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    config['dir'] = args.exper_folder
    config['dataset'] = args.dataset
    
    
    if args.dataset in ['stereo', 'dexter+object']:
        config['is_real'] = 1
    else:
        config['is_real'] = 0
    
    with torch.no_grad():
        model = get_model(config)
        weight_file = 'experiment/' + config['dir'] + '/' + config['weights']
        model.load_state_dict(torch.load(weight_file)['model'])
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        # Export the model
        torch.onnx.export(model,               # model being run
                        {"image": x},                         # model input (or a tuple for multiple inputs)
                        "hand-pose-3d.onnx",   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=9,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['image'],   # the model's input names
                        output_names = ['vector_2d', 'heatmaps', 'vector_3d'], # the model's output names
                        dynamic_axes={'image' : {0 : 'batch_size'},    # variable lenght axes
                                        'vector_2d' : {0 : 'batch_size'},
                                        'heatmaps' : {0 : 'batch_size'},
                                        'vector_3d' : {0 : 'batch_size'}})

    
    ## Load ONNX Model
    model = onnx.load('hand-pose-3d.onnx')
    ## Convert ONNX Model into CoreML MLModel
    comMLModel = convert(model, image_input_names='image', minimum_ios_deployment_target='13', custom_conversion_functions={'Upsample':_convert_upsample, 'Tile': _convert_tile})

    comMLModel.save("hand-pose-3d.mlmodel")