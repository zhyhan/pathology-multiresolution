# Multiresolution pathological image processing.

Step 1: all.py (including prepocess.py and train.py)

Step 2: test_multiresolution.py

Path.cfg is a configuration file, which includes several key params.

[In DEFAULT section]: 

'target level' controls the level of expected rescaled times.

[In PREPROCESS section]:

'rescaled_times' controls the zoom out times compared with raw mask size, in order to accelerate the preprocessing speed.

'patch_size' controls the expected patch size, in which level 2 = 32, level 1 = 64,  level 0 = 256.

[In PREPROCESS section]:

'model' controls the target network. Each level has a corresponding network. For example, level 1 has batch size of 64, the network is net_64 in the custom_layer file.


[In GENERATE MAP section]:
'checkpoint_pth_file_level0' and 'optim_state_file_level0' are the well-trained checkpoints of highest resolution, in this project, it is the model of level 0.

