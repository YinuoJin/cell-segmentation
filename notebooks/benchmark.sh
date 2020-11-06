#!/bin/bash

# benchmark multiple loss functions, architecture options & output dimention (1d vs. 3d)
# Epoch: 100; lr: 0.01; patience: 20

input_path="../datasets/ds_bowl_2018/"
output_path="../ds_bowl_benchmark/"

###################
#  1d U-net
#################3#
#echo "Binary U-Net..."
#echo "---------------------------------------------"


# baseline u-net
#echo "Baseline U-Net"
#./train.py --option binary -i $input_path -o ${output_path}binary_results/unet/ -a unet -n 1 -p 30 --early-stop

# u-net with "distance" map
#echo "U-Net with SAW"
#./train.py --option binary -i $input_path -o ${output_path}binary_results/unet_dist/ -a unet -d dist -n 1 -p 30 --early-stop

# u-net with jaccard loss
#./train.py --option binary -i $input_path -o ../binary_results/unet_jaccard/ -a unet -l jaccard -n 100 -p 20

# u-net with dice loss
#./train.py --option binary -i $input_path -o ../binary_results/unet_dice/ -a unet -l dice -n 100 -p 20

# u-net with boundary loss
#./train.py --option binary -i $input_path -o ../binary_results/unet_boundary/ -a unet -l boundary -d boundary -n 100 -p 20

# resnet
#echo "ResNet with SAW"
#./train.py --option binary -i $input_path -o ${output_path}binary_results/resnet/ -a resnet -d dist -n 1 -p 20 --early-stop

# convlstm net
#echo "ConvLSTM with SAW"
#./train.py --option binary -i $input_path -o ${output_path}binary_results/convnet/ -a lstm -d dist -n 1 -p 20 --early-stop

###################
#  3d U-net
#################3#

echo "3-label U-Net..."
echo "---------------------------------------------"

# baseline u-net
echo "Baseline U-Net"
./train.py --option multi -i $input_path -o ${output_path}multi_results/unet/ -a unet -r 0.001 -n 50 -p 20 --early-stop

# u-net with saw map
echo "U-Net with SAW"
./train.py --option multi -i $input_path -o ${output_path}multi_results/unet_saw/ -a unet -r 0.001 -d saw -n 50 -p 20 --early-stop

# resnet
echo "ResNet with SAW"
./train.py --option multi -i $input_path -o ${output_path}multi_results/resnet/ -a resnet -r 0.001 -d saw -n 50 -p 20 --early-stop


# resnet without saw
#./train.py --option multi -i $input_path -o ../ds_bowl_benchmark/multi_results/resnet -a resnet -n 100 -p 20

# convlstm without saw
#./train.py --option multi -i $input_path -o ../ds_bowl_benchmark/multi_results/convnet -a convnet -n 100 -p 20

# FPN
echo "FPN with SAW"
echo "----------------------------------------------"
./train.py --option fpn -i $input_path -o ${output_path}multi_results/fpn/ -a fpn -r 0.001 -d saw -n 50 -p 20 --early-stop



