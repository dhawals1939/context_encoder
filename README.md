# Project : Context Encoders: Feature Learning by Inpainting
---
## Folder structure and instructions
The folder structure for the code is :

    .
    ├── images                          #epoch wise outputs when using dogs and cats dataset of image net
    ├── log                             #log files of outputs
    ├── paris_eval                      #set of validation files
    ├── paris_eval_results              #results of the validation files after training for 3000 epochs
    ├── results                         #epoch wise results for all images in paris_eval
    ├── misc                 
    │   ├── Screenshots                 #includes the screenshots of the model while running
    │   └──bash scripts                 #used for running the code
    ├── CAT_Smaller_Set                 #results on cat dataset
    ├── GIF_Result_Across_Epochs        #results shown at every few epochs and made into a gif
    ├── Context_encoder.ipynb           #code for running in google colab
    ├── context_encoder.py              #code for training and validating in gpu
    ├── data_loader.py                  #code for loading data into the model
    ├── discriminator.py                #model of dicriminator defined
    ├── generator.py                    #model for generator defined
    ├── test.py                         #code for testing the trained models.
    ├── .pdf files                      #reports and ppts of the project
    └── README.md

## Results over the paris dataset
The sample results of the dataset along with the ground truth and the cropped out region. Top is the cropped image input to the generator and the middle is the output of the generator and last image shows the ground truth.
Some of the results are:<br/>
![](paris_eval_results/generated_imgs-1.jpg)
![](paris_eval_results/generated_imgs-17.jpg)
![](paris_eval_results/generated_imgs-70.jpg)
![](paris_eval_results/generated_imgs-88.jpg)
<br/>
Gifs illustrating Epoch-wise improvements<br/>
![](paris_epoch_results_gifs/147.gif)
![](paris_epoch_results_gifs/174.gif)
![](paris_epoch_results_gifs/1.gif)
![](paris_epoch_results_gifs/4.gif)

## Graphs of the train and validation losses across the epochs
![](misc/graph.PNG)

## Results over cat dataset
The best results with cat dataset are shown below

![](cat_dataset_results/49350.jpg)
![](cat_dataset_results/49300.jpg)
![](cat_dataset_results/46000.jpg)
![](cat_dataset_results/49950.jpg)
![](cat_dataset_results/49850.jpg)
![](cat_dataset_results/49650.jpg)

## Links
## Paris_dataset
#Note:
The Paris Building Dataset cannot be made publicly available. Access to dataset can be obtained via mailing the original paper authors: Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell & Alexei A. Efros.
## [Code for plots]
(https://colab.research.google.com/drive/1qHWsU9b6sVo0FfPebkF1GWZlLIpI-Cs0)<br/>
## [Presentation]
(https://docs.google.com/presentation/d/1QF8oylaEKNHnNxCboERB1qOtrI7GsiZj1sY1es17YgM/edit?usp=sharing)<br/>
