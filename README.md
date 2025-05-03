# MechDataExtractor
<img width="352" alt="image" src="https://github.com/user-attachments/assets/99719931-00e9-471b-a2a8-e1d75b9fb945" />


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
Save the best checkpoint to MODEL.pth<br/>

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

## Application on Chemical Reaction Mechanism Images
We collected 296 reaction mechanism images from textbook: Named Reactions 4th edition (Li, 2009). <br/>

**Usage**<br/>
Each image is named with its reaction name. The images are processed with this model and parsed by RxnScribe (Qian, 2023).
it contains information such as predicted molecular identity, positions and reaction conditions. 
Find the [images](https://huggingface.co/datasets/Ting25/MechRxn/blob/main/ver_mech.zip) and [parsed dataset](rxn_data/batch_prediction.json). <br/>

**Disclaimer**<br/>
Note that the dataset includes errors still even though it performs better with preprocessing of arrow removals. This dataset does not aim to serve as a benchmark, but more of a centralized and unified collection of reaction that benefit future researches in both chemistry and computer vision.




## Updates
12/7/24 Repository created. Code coming soon.<br/>
14/7/24 Code updated.<br/>
7/8/24 Training data uploaded. <br/>
12/9/24 Reaction mechanism dataset uploaded. <br/>
26/9/24 Reaction mechanism dataset parsed and uploaded. <br/>

