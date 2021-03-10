## Dependencies
- pytorch >= 1.2.0
- torchvision >= 0.4.0
- numpy >= 1.18.5
- scipy >= 1.4.1
- pandas >= 1.0.5
- opencv-python >= 4.3.0
- skimage >= 0.17.2
- Augmentor >= 0.2.8
- progress >= 1.5


## Training options
```
usage: train.py [-h] -i ROOT_PATH --option OPTION --image-type IMAGE_TYPE
                [-o OUT_PATH] [-a NET_OPTION] [-d DIST] [-b BATCH_SIZE]
                [-l LOSS] [-n N_EPOCHS] [-m MODEL_PATH] [-r LR]
                [-p PATIENCE_COUNTER] [--test] [--joint-pred] [--augment]
                [--early-stop] [--region-option]

Unet training options

required arguments:
  -i ROOT_PATH          Root directory of input image datasets for training/testing
  --option OPTION       Training option: (1). binary, (2). multi
  --image-type IMAGE_TYPE
                        Image type: (1). nuclei, (2). membrane

optional arguments:
  -o OUT_PATH           Directory to output file
  -a NET_OPTION         Model Architecture
                         Options: (1).unet; (2).resnet; (3).convnet; (4). fpn
  -d DIST               Distance function for weighted loss
                         Options: (1).dist; (2).saw; (3).class; (4).boundary
  -b BATCH_SIZE         Batch size
  -l LOSS               Loss function
                          Options: (1).bce; (2).jaccard; (3).dice; (4).boundary
  -n N_EPOCHS           Total number of epoches for training
  -m MODEL_PATH         Saved model file
  -r LR                 Learning rate
  -p PATIENCE_COUNTER   Patience counter for early-stopping or lr-tuning
  --test                Whether to perform prediction & postprocessing on the test set
  --joint-pred          Whether to perform joint prediction & postprocessing on the test set (with both nuclei & membrane marker channels)
  --augment             Whether to perform data augmentation in the current run
  --early-stop          Whether to perform early-stopping; If False, lr is halved when reaching each patience
  --region-option       Whether to use dice loss as the Region-based loss for boundary loss; If False, jaccard loss is used instead
```
