# MechDataExtractor
<img width="700" alt="image" src="https://github.com/user-attachments/assets/99719931-00e9-471b-a2a8-e1d75b9fb945" />


## Usage
**Use a conda environment for clean installation**
```
$ conda create --name molseg python=3.8.0
$ conda activate molseg
$ conda install pip
$ python3 -m pip install -U pip
$ pip install -r requirements.txt
```
**Data preparation** <br/>
Ground truth images used for training are in RGB format. Image masks should be in Black and White format. they should be in identical names under _imgs_ and _masks_ folders. <br/>
All the images should be squares. Place them on a squared canvas if necessary. The model works well for images sizing under 600*600. <br/>
Mechanistic molecular ground truth data and image segmentation masks can be found on [ZENODO](https://zenodo.org/records/12741238). <br/>

**Model Training**<br/>
Run the training script or train.py.
```
$ sbatch scripts/train.sh
```
Save the best checkpoint to `MODEL.pth`<br/>
A pretrained checkpoint is saved to: `checkpoint.pth` in [huggingface](https://huggingface.co/datasets/Ting25/MechRxn/blob/main/checkpoint.pth). If you want to use this checkpoint, simply `-m checkpoint.pth` after downloading it and put in your root directory.

**Prediction**<br/>
After training your model and saving it to MODEL.pth, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:
```
$ python predict.py -i image.jpg -o output.jpg
```
To predict a multiple images and show them without saving them:
```
$ python predict.py -i image1.jpg image2.jpg --viz --no-save
```
For batch predictions, use `scripts/predict.sh` after setting up the bash environment. Feel free to change the input directory and the output directory for further accomodating your task.

## Application on Chemical Reaction Mechanism Images
We collected 296 reaction mechanism images from textbook: Named Reactions 4th edition (Li, 2009). <br/>

### Usage
Each image is named with its reaction name. The images are processed with this model and parsed by RxnScribe (Qian, 2023).
it contains information such as predicted molecular identity, positions and reaction conditions. 
Find the [images](https://huggingface.co/datasets/Ting25/MechRxn/blob/main/ver_mech.zip) and [parsed dataset](rxn_data/batch_prediction.json). <br/>

### Results
| Dess-Martin periodinane oxidation | Corresponding object masks |
|:-------------------------------:|:--------------------------------:|
| ![First Image](https://github.com/user-attachments/assets/a944c42b-e7ba-4b8f-8b84-9731b4807d29) | ![Second Image](https://github.com/user-attachments/assets/26e4fa11-b028-4b24-8dd8-70f643767748) |

### Image Postprocessing
This architecture is mainly used for noise removal in chemical reaction mechanism images. In order to remove the noise segmented out in the original image, use `process.py` for overlaying the image mask and the original image.
```python
imgs_path = "ver_mech/"
masks_path = "mechrxn_arrowmask/"
processed_path = "mechrxn_processed/"
```
`imgs_path` is the original image folder path; `masks_path` is the images masks obtain with U-Net; `processed_path` can be renamed for your own interest. 

### Disclaimer
Note that the dataset includes errors still even though it performs better with preprocessing of arrow removals. This dataset does not aim to serve as a benchmark, but more of a centralized and unified collection of reaction that benefit future researches in both chemistry and computer vision.

## References
- The original U-Net paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- The model took reference from [Milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- Molecular and reaction information extraction is employed by models from [thomas0809/Molscribe](https://github.com/thomas0809/MolScribe) and [thomas0809/RxnScribe](https://github.com/thomas0809/RxnScribe)

