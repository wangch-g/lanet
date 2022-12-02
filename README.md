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
python test.py --test_dir=./datasets/HPatches/hpatches-sequences-release
```

## License

The code is released under the [MIT license](LICENSE).

## Citation
Please use the following citation when referencing our work:
```
@InProceedings{Wang_2022_ACCV,
    author    = {Wang, Changhao and Zhang, Guanwen and Cheng, Zhengyun and Zhou, Wei},
    title     = {Rethinking Low-level Features for Interest Point Detection and Description},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {2059-2074}
}
```
