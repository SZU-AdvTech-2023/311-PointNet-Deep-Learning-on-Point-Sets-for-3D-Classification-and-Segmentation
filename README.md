# PointNet.pytorch
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet/model.py`.

It is tested with pytorch-1.0.

# Download data and running

```
git clone https://github.com/fxia22/pointnet.pytorch
cd pointnet.pytorch
pip install -e .
```

Download and build visualization tool
```
cd scripts
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

Training 
```
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```

Building
```
cd utils/nearest_neighbors
python setup.py build_ext --inplace
```

Testing
```
cd utils
python show_cls.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python show_seg.py --dataset <dataset path> --nepoch=<number epochs> 
```

Use `--feature_transform` to use feature transform(T-net).


# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)
