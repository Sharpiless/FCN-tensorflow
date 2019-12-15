import os
from tensorflow.python import pywrap_tensorflow
 
checkpoint_path = './model/VGG_model.ckpt'
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)