# ACAN
Pytorch implementation of ACAN for monocular depth estimation.

**Attention-based Context Aggregation Network for Monocular Depth Estimation**  
This repository includes our model code.

This code works on Python 3 & TensorFlow 1.4 and the images in this dataset are of actual road scenes captured while unmanned vehicle driving.

If this code and dataset are provided for research purposes, please see License section below.

## Requirements
This code was tested with Pytorch 0.4.1, CUDA 9.1 and Ubuntu 18.04.  
Training takes about 48 hours with the default parameters on the **Eigen** split on a Nvidia GTX1080Ti machine.  

## Data
This model requires rectified stereo pairs for training.  
There are two main datasets available: 
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
We used **eigen** split of the data, amounting for approximately 22k training samples, you can find them in the [kitti_path_txt](kitti_path_txt) folder.  

### [NYU v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
We download the raw dataset, which weights about 428GB. We use the toolbox of NYU v2 to sample around 12k training samples, you can find them in the [matlab](matlab) folder.

## Training

**Warning:** The input sizes need to be mutiples of 8. 

```shell
python code/classification/main.py
```

## Testing  
To test change the `mode` variable to `test` and set up `test_index`, the network will output the depth maps in the main folder.

## Results
You can download our pre-trained models and results, we give the link below:
[NYU v2]()

[KITTI]()


## License
I provide this project for research purposes, please follow `Citing`.

For removal of copyrighted content, please contact me on GitHub.

## Citing
If you use this project in academic work, please cite as follows:
```
@misc{miraiaroha,
        title={Attention-based Context Aggregation Network for Monocular Depth Estimation},
        url={https://github.com/miraiaroha/ACAN},
        author={Yuru, Chen},
        year={2019},
        publisher = {GitHub},
        journal = {GitHub repository}
}
```
