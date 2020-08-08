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
  -b BATCH_SIZE        Batch size  [default: 8]
  -c CHANNEL           Output channel size  [default: 3]
  -l LOSS              Loss function  [default: bce]
                         Options: (1). bce; (2). jaccard; (3).dice; (4).boundary
  -n N_EPOCHS          Total number of epoches for training  [default: 150]
  -r LR                Learning rate  [default: 0.01]
  -p PATIENCE_COUNTER  Patience counter for early-stopping or lr-tuning  [default: 30]
  --augment            Whether to perform data augmentation in the current run [Usually no need to run]
  --early-stop         Whether to perform early-stopping; If False, lr is halved when reaching each patience
  --region-option      Whether to use dice loss as the Region-based loss for boundary loss; If False, jaccard loss is used instead
```

## Sample run
```
# Training "nuclei" dataset using 3-class model, weighted BCE loss & early-stopping with patience=20
./train.py -i ../datasets/multi_cell_nuclei/ -l bce -p 20 --early-stop

# Training "membrane" dataset using binary model, Jaccard loss & decaying learning rate (lr /= 2 when patience >= patience_counter)
./train.py -i ../datasets/multi_cell_membrane/  -c 1 -l jaccard
```
