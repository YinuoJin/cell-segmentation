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
usage: train.py [-h] -i ROOT_PATH [-b BATCH_SIZE] [-c CHANNEL] [-l LOSS]
                [-n N_EPOCHS] [-r LR] [-p PATIENCE_COUNTER] [--augment]
                [--early-stop] [--region-option]

Unet training options

required arguments:
  -i ROOT_PATH         Root directory of input image datasets for training

optional arguments:
  -b BATCH_SIZE        Batch size
  -c CHANNEL           Output channel size
  -l LOSS              Loss function
                         Options: (1). bce; (2). jaccard; (3).dice; (4).boundary
  -n N_EPOCHS          Total number of epoches for training
  -r LR                Learning rate
  -p PATIENCE_COUNTER  Patience counter for early-stopping or lr-tuning
  --augment            Whether to perform data augmentation in the current run
  --early-stop         Whether to perform early-stopping; If False, lr is halved when reaching each patience
  --region-option      Whether to use dice loss as the Region-based loss for boundary loss; If False, jaccard loss is used instead
```

## Sample run
```
# Using Weighted BCE loss & early-stopping with patience=20
./train.py -i [data_path] -l bce -p 20 --early-stop  

# Using Jaccard loss & decaying learning rate (lr /= 2 when patience >= patience_counter)
./train.py -i [data_path] -l jaccard

# Using multi-label Weighted BCE with SAW map
./train.py -i [data_path] -c 3 -p 20 
```
