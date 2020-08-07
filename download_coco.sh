#!/bin/bash 

mkdir -p dataset/coco/

curl -o coco_train2017.zip -L http://images.cocodataset.org/zips/train2017.zip
unzip coco_train2017.zip 
mv coco_train2017/ dataset/coco/image

curl -o coco_segmap.zip -L https://github.com/HanYangZhao/SPADE-Tensorflow/releases/download/aed20k/coco_segmap.zip
mv segmap dataset/coco/segmap

curl -o coco_val2017.zip -L http://images.cocodataset.org/zips/val2017.zip
unzip coco_val2017.zip
mv coco_val2017/ dataset/coco/val

curl -o coco_segmap_test.zip -L https://github.com/HanYangZhao/SPADE-Tensorflow/releases/download/aed20k/coco_segmap_test.zip
unzip coco_segmap_test.zip
mv segmap_test/ dataset/coco/segmap_test