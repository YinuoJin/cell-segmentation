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
usage: train.py [-h] [-i ROOT_PATH] [-b BATCH_SIZE] [-l LOSS] [-n N_EPOCHS]
                [-r LR] [-p PATIENCE_COUNTER] [--early-stop] [--region-option]
                [--enhance-img] [--contour-mask]

Unet training options

optional arguments:
  -h, --help            show this help message and exit
  -i ROOT_PATH, --root-path ROOT_PATH
                        Root directory of input image datasets for training
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size (Default=8)
  -l LOSS, --loss LOSS  
                        Loss function option: (1). bce; (2). jaccard; (3).dice; (4).boundary (Default='bce')
  -n N_EPOCHS, --n-epochs N_EPOCHS
                        Total number of epoches for training (Default=150)
  -r LR, --loss-rate LR
                        Loss rate (Default=0.01)
  -p PATIENCE_COUNTER, --patience PATIENCE_COUNTER
                        Patience counter for early-stopping or lr-tuning (Default=30)
  --early-stop          
                        Whether to perform early-stopping; If False, lr is halved when reaching each patience
  --region-option       
                        Whether to use dice loss as the Region-based loss for boundary loss; If False, jaccard loss is used instead
  --enhance-img         
                        Whether to use Quantile transformation / Equalization normalization to enhance raw images
  --contour-mask        
                        Whether to take contours of ground-truth masks
```

## Sample run
```
# Using Weighted BCE loss & early-stopping with patience=20
./train.py -l bce -p 20 --early-stop  

# Using Jaccard loss & decaying learning rate (lr /= 2 when patience >= patience_counter)
./train.py -l jaccard

# Using Boundary loss with Jaccard loss; Enhancing image frames
./train.py -l boundary --region-option --enhance-img

# Using Boundary loss with Soft Dice loss; Enhance image frames & Contour image masks
./train.py -l boundary --enhance-img --contour-mask
```
