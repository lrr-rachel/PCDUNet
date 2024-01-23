# PCDUNet

This is a PyTorch implementation of a PCDUNet (frame-aligned U-Net on feature level) designed to mitigate the noisy effects on video sequences (with large motion). 

The originasl U-Net architecture is based on the work by [O. Ronneberger, P. Fischer, and T. Brox, 'U-Net: Convolutional Networks for Biomedical Image Segmentation,' arXiv.org, 18-May-2015](https://arxiv.org/abs/1505.04597). Frame alignment is based on the state-of-the-art video restoration model [EDVR](https://github.com/xinntao/EDVR). Modifications have been made to create a revised architecture to deal with video sequences where large motion are present.

## Dependencies
The code has been mainly developed and tested in:
pytorch 2.0.1 cuda 11.8 (latest released version)

Select one of the followings for setting up training environment:
* [Pytorch](https://pytorch.org/) 
* BlueCrystal4: Recommand building your own anaconda environment. Please see detailed instrusctions on [installing your own conda env on bc4](https://www.acrc.bris.ac.uk/protected/hpc-docs/software/python_conda.html)


## Training
* Specify the noisy input directory, clean groundtruth directory, output directory, etc.
* ```main_heathaze.py --help``` or ```main_lowlight.py --help```to see useful help messages.
* For Atmospheric Distortion Mitigation:
```
python main_heathaze.py --network PCDUNet --root_distorted your_input_image_path --root_restored your_groundtruth_path --resultDir atmPCDUNet 
```
* For Low Light Enhancement:
```
python main_lowlight.py --network PCDUNet --root_distorted your_input_image_path --root_restored your_groundtruth_path --resultDir LLPCDUNet --NoNorm
```
* The best-trained model will be saved in the output directory specified.

Note: Please review the help messages for command line arguments before proceeding.
