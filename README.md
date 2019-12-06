* Implemented [LadderNet: Multi-path networks based on U-Net for medical image segmentation
](https://arxiv.org/abs/1810.07810) for Pancres Segmentation

# Requirement
* Python3.6
* PyTorch 0.4

# How to run
* run ```python prepare_datasets_DRIVE.py``` to generate hdf5 file of training data
* run ```cd src```
* run ```python retinaNN_training.py``` to train and validate

# Parameter defination
* parameters (path, patch size, et al.) are defined in <b>"configuration.txt"</b>
* training parameters are defined in src/retinaNN_training.py line 49 t 84 with notes <b>"=====Define parameters here =========" </b>

# Model
![](figures/lad.png)
![](figures/att.png)

# Results
![](figures/res.png)
