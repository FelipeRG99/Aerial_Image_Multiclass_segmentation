Deep learning based in U-net and Deeplab architecture for multiclass segmentation task on aerial images

Dataset
-----------------------
Dataset from [kaggle](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) consist of 72 images grouped in Tiles, each tile has images 
of same shape but with diferent shape betweeen tiles.The images are aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation 
in 6 classes.The classes are:
Building: #3C1098       
Land (unpaved area): #8429F6        
Road: #6EC1E4       
Vegetation: #FEDD3A     
Water: #E2A929      
Unlabeled: #9B9B9B   

Info
-----------------------
The majority of the proyect has comments in spannish (my native language) cause is a for fun proyect and for learnning purpose.

Trained Models
-----------------------
The trained models weigh too much (>100Mb) so they cannot be uploaded. To create these model weights, it is necessary to run the [Notebooks](Notebooks) and obtain these weights (Of course create a directory named Trained_Models).

[Main](main.py)
-----------------------
There are two models, all of them work with cropped images of shape (224,244):
- unet
- deeplab

Results
-----------------------
Due to de ammount of detail in some images and the complexity of the mask (that are very poorly built in my opinion), the reshape method is discarted. These results are from the cropped method that captures all the detail and can work with images of diferent shapes. 

| Model | Dice (Test)   | Accuracy (Test)    |
| :---:   | :---: | :---: |
| Unet | 0.695   | 0.8764   |
| Deeplab | 0.7359   | 0.8864   |

<figure style="display:inline-block; margin-right: 10px;">
  <img src="Imagenes resultados/deeplab_image_part_003.png" />
  <figcaption>Deeplab Model</figcaption>
</figure>
<figure style="display:inline-block;">
  <img src="Imagenes resultados\unet_image_part_003..png" />
  <figcaption>Unet Model</figcaption>
</figure>

| Model | Dice (Img)   | Accuracy (Img)    |
| :---:   | :---: | :---: |
| Unet | 0.8823   | 0.8823   |
| Deeplab | 0.9222   | 0.9222   |