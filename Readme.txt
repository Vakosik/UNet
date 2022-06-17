1) Installing the enviroment
__________________________________________________

First, it's necesarry to install the enviroment from file 'unet_env.yml'. The enviroment contains packages and their version needed to run the code.
If you have the Anaconda distribution, you can do so by running:

conda env create --file unet_env.yml

This is a one-time action only. If you use the same computer, next time go straight to the step 2).

2) Running the code
____________________________________________________

First, it's necesarry to activate the enviroment installed in the previous step. You can do that by

conda activate UNet

At the beginning of the line, you should now see (UNet).

The program for segmentation has three parameters. Two of them are mandatory to specify:
i) input: An image file to segment. If a directory is sent, it segments all images in the directory. A directory must be written with / at the end.
ii) net: The model to use, choices are 0, 1, 2, meaning 0 - QUANTA SE, 1 - ST BSE, 2, - ST SE

The last parameter is optional:
iii) -netpath: The directory (must be written with / at the end) with saved neural networks. If not specified, it works with the current working directory.
This directory should contain files unet_QUANTA_SE.h5, unet_ST_BSE.h5, unet_ST_SE.h5

The segmentation is run by writing python segmentation.py (or full path to the file), a space, specifying the first parametr, a space, specifying the second parametr 
(optinally a space, -netpath and specifying the last parameter parametr). 

Examples:
python segmentation.py "C:/segmentation/QUANTA SE/input/S3o spodek 1 orig.tif" 0
python segmentation.py "C:/segmentation/QUANTA SE/input/S3o spodek 1 orig.tif" 0 -netpath "C:/segmentation/models/"
python segmentation.py "C:/segmentation/QUANTA SE/input/" 0 -netpath "C:/segmentation/models/"
python segmentation.py "C:/segmentation/ST BSE/input/" 1 -netpath "C:/segmentation/models/"
python segmentation.py "C:/segmentation/ST SE/input/" 2 -netpath "C:/segmentation/models/"


The output is saved to the same directory where input is stored. 
