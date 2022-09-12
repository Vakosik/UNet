1) Installing the enviroment
__________________________________________________

First, it's necesarry to install the enviroment from file 'unet_env.yml'. The enviroment contains packages and their version needed to run the code.
If you have the Anaconda distribution, you can do so by running:

conda env create --file unet_env.yml

This is a one-time action only. If you use the same computer, next time go straight to the step 2).

2) Running the segmentation code
____________________________________________________

First, it's necesarry to activate the enviroment installed in the previous step. You can do that by

conda activate UNet

At the beginning of the line, you should now see (UNet).

The program for segmentation has three parameters. Two of them are mandatory to specify:
i) input: An image file to segment. If a directory is sent, it segments all images in the directory. A directory must be written with / at the end.
ii) net: The model to use, choices are unet_QUANTA_SE, unet_ST_BSE, unet_ST_SE.

The last parameter is optional:
iii) -netpath: The directory (must be written with / at the end) with saved neural networks. If not specified, it works with the current working directory.
This directory should contain files unet_QUANTA_SE.h5, unet_ST_BSE.h5, unet_ST_SE.h5

The segmentation is run by writing python segmentation.py (or full path to the file), a space, specifying the first parametr, a space, specifying the second parametr 
(optinally a space, -netpath and specifying the last parameter parametr). 

Examples:
python segmentation.py "C:/segmentation/QUANTA SE/input/S3o spodek 1 orig.tif" unet_QUANTA_SE
python segmentation.py "C:/segmentation/QUANTA SE/input/S3o spodek 1 orig.tif" unet_QUANTA_SE -netpath "C:/segmentation/models/"
python segmentation.py "C:/segmentation/QUANTA SE/input/" unet_QUANTA_SE -netpath "C:/segmentation/models/"
python segmentation.py "C:/segmentation/ST BSE/input/" unet_ST_BSE -netpath "C:/segmentation/models/"
python segmentation.py "C:/segmentation/ST SE/input/" unet_ST_SE -netpath "C:/segmentation/models/"


The output is saved to the same directory where input is stored. 


3) Running the automatic_train code
____________________________________________________

First, it's necesarry to activate the enviroment installed in the previous step. You can do that by

conda activate UNet

At the beginning of the line, you should now see (UNet).

The program for segmentation has two parameters, both mandatory:
i) model: A path to a model that you want to fine-tune (the code will load this model and start adjusting it to your new data). Don't forget about .h5 suffix.
ii) data: A directory with data. The directory must contain two folders train and test. Both these folders must contain two subfolders input and target. If the data folder doesn't satisfy this, the code will call an error. There is an example of such folder uploaded on this GitHub repository, you can run the code on that.

The training is run by writing python automatic_train.py (or full path to the file), a space, specifying the first parametr, a space, specifying the second parametr.

Example:
python automatic.py "C:/UNet/models/unet_QUANTA_SE2.h5" "C:/UNet/data/"
