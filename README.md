# Rethinking Low-level Features for Interest Point Detection and Description

## Dependency
   - pytorch
   - torchvision
   - cv2
   - tqdm

    We use cuda 11.4/python 3.8.13/torch 1.10.0/torchvision 0.11.0/opencv 3.4.8 for training and testing.


## Pre-trained models
We provide two versions of LANet with different structure in [network_v0](network_v0) and [network_v1](network_v1), the corresponding pre-trained models are in [checkpoints](checkpoints).
   - v0: The original version used in our paper.
   - v1: An improved version that has a better over all performance.  


## Training
Download the COCO dataset:
```
cd datasets/COCO/
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```
Prepare the training file:
```
python datasets/prepare_coco.py --raw_dir datasets/COCO/train2017/ --saved_dir datasets/COCO/ 
```

To train the model (v0) on COCO dataset, run:
```
python main.py --train_root datasets/COCO/train2017/ --train_txt datasets/COCO/train2017.txt
```


## Evaluation
###  Evaluation on HPatches dataset
Download the HPatches dataset:
```
cd datasets/HPatches/
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xvf hpatches-sequences-release.tar.gz
```

To evaluate the pre-trained model, run:
```
python test.py --test_dir ./datasets/HPatches/hpatches-sequences-release
```


## License
The code is released under the [MIT license](LICENSE).


## Citation
Please use the following citation when referencing our work:
```
@InProceedings{Wang_2022_ACCV,
    author    = {Changhao Wang and Guanwen Zhang and Zhengyun Cheng and Wei Zhou},
    title     = {Rethinking Low-level Features for Interest Point Detection and Description},
    booktitle = {Computer Vision - {ACCV} 2022 - 16th Asian Conference on Computer
                 Vision, Macao, China, December 4-8, 2022, Proceedings, Part {II}},
    series    = {Lecture Notes in Computer Science},
    volume    = {13842},
    pages     = {108--123},
    year      = {2022}
}
```


## Related Projects
https://github.com/TRI-ML/KP2D
