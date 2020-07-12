# ACAN: Attention-based Context Aggregation Model for Monocular Depth Estimation.

Pytorch implementation of ACAN for monocular depth estimation.</br>
More detalis [arXiv](https://arxiv.org/abs/1901.10137) </br>

## Architecture
<p align="center">
    <img src="https://github.com/miraiaroha/ACAN/tree/master/images/architecture.png"></br>
</p>

## Visualization of Attention Maps

<p align="center">
    <img src="/images/kitti_att.png"></br>
</p>

* The first and second row respectively denotes the attention maps trained with and w/o `Attention Loss`. </br>

## Soft Inference VS Hard Inference

<p align="center">
    <img src="/images/soft_vs_hard2.png"></br>
</p>

* The third column and the fourth column respectively denotes the results of soft inference and hard inference. </br>
## Quick start

### Requirements
~~~~
torch=0.4.1
torchvision
tensorboardX
pillow
tqdm
h5py
scikit-learn
cv2
~~~~
This code was tested with Pytorch 0.4.1, CUDA 9.1 and Ubuntu 18.04.  
Training takes about 48 hours with the default parameters on the **KITTI** dataset on a Nvidia GTX1080Ti machine.  </br>

### Data
There are two main datasets available: 
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
We used [Eigen](https://cs.nyu.edu/~deigen/depth/) split of the data, amounting for approximately 22k training samples, you can find them in the [kitti_path_txt](./kitti_path_txt) folder.  

### [NYU v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
We download the raw dataset, which weights about 428GB. We use the toolbox of NYU v2 to sample around 12k training samples, you can find them in the [matlab](code/matlab) folder and use `Get_Dataset.m` to produce the training set or download the processed dataset from [BaiduCloud](https://pan.baidu.com/s/1svDzuEruxIr5kEIPMmJKVg).

### Training

**Warning:** The input sizes need to be mutiples of 8. 

```shell
bash ./code/train_nyu_script.sh
```

### Testing  
```shell
bash ./code/test_nyu_script.sh
```

### Attention Map
If you want to get the task-specific attention maps, you should first train your model from scratch, then finetuning with attention loss, by setting
~~~~
BETA=1
RESUME=./workspace/log/best.pkl
EPOCHES=10
~~~~

## Thanks to the Third Party Libs
[Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)

[Pytorch-OCNet](https://github.com/PkuRainBow/OCNet.pytorch)

[NConv-CNN](https://github.com/abdo-eldesokey/nconv-nyu)


