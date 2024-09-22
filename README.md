# Figure, table, and reference extraction 
This project implements the extraction of figures, tables, and references from a scientific research paper. 

![Introduction](Figures/process.jpg)

## Data Preparation and Processing 
For data preparation, figures, tables, and references were labelled accordingly in 267 pages of scientific research papers (mainly in electrical engineering, physics, and computer science) using LabelImg. 

![Labeling process](Figures/data_prep.png)

The data_processing.py file divides 80% of the data for training and 20% for validation. 
The images in the training set were augmented 5 times each, leading to a total of 1065 training images. The images in the validation set were left unaugmented so that the validation set remains truly unseen by the model. 

## Model Design 
The `final_model` directory contains images of the training process, parameters of the model, and the final model itself - named as `final.pt`. 

`final.pt` was obtained by training the YOLOv5 object detection model (specifically the YOLOv5s model) with the following parameters: 
>Image resolution: 832 x 832  
>Batch: 16  
>Epochs: 80
>IoU: 0.2  

## Results 

# Metrics 
The specific metrics for the final.pt model was as follows:  
Loss:  
>mAP_0.5:  
>mAP_0.5:0.95:  
>mAP_recall:  
>mAP_precision:

The confusion matrix: 
![Confusion matrix]()


## Testing Method  
In order to test the final.pt model with other inputs, you can download and run the 'test_model.py' file with adjustments to the path to the model, output files, and selection of device. 

Change the directory to the final.pt model in:
```python
'model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/installed/extraction.pt/model')' 
```

The output directory can also be changed in: 
```python
test_dir = 'directory/to/test/images'
output_dir = 'directory/to/save/annotated/pages'
objects_dir = 'directory/to/save/extracted/images/as/separate/files'
```


