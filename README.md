# Classification of ceramics using multi view geodesic farthest point sampling

The Source-Codes were tested on Python 3.6.10 and PyTorch 1.3.1 using GeForce RTX 2080.

<br/>

First, download the datasets according to the method.

<br/>
<br/>

## Pottery Dataset

<p align="center"><img src="./Images/pottery_dataset.png" width =200px> </p>

### Multiview-dataset
- [Normal](https://reactnative.dev)
- [Curvature](https://reactnative.dev)
- [Geodesic-max](https://reactnative.dev)
- [Geodesic-min](https://reactnative.dev)
### Point-Cloud-dataset
- [Normalized](https://reactnative.dev)
- [Non-Normalized](https://reactnative.dev)

## Peruvian Dataset (not available yet )

<p align="center"><img src="./Images/peruvian_dataset.png" width =300px> </p>

### Multiview-dataset
- [Normal]()
- [Curvature]()
- [Geodesic-max]()
- [Geodesic-min]()
### Point-Cloud-dataset
- [Normalized]()
- [Non-Normalized]()

<br/>

## Run projects

<br/>

### MVCNN (Curvature, Geodesic)
```
cd Sources-Codes/mvcnn_pytorch_$database$/
```
```
python train_mvcnn.py -name mvcnn -num_models 1000 -weight_decay 0.001 -num_views 12 -cnn_name vgg11
```

### BPS-MLP
```
cd Sources-Codes/basis-point-sets/
```
```
python bps_demos/train_modelnet_mlp.py
```

### DGCNN
```
cd Sources-Codes/dgcnn-pytorch/
```
```
python main.py --model=dgcnn
```

### POINTNET
```
cd Sources-Codes/dgcnn-pytorch/
```
```
python main.py --model=pointnet
```

## Results

<br/>

Comparison of the accuracy obtained from the proposed method (MVCNN-Geodesic) with different number of FPS points using the maximum and minimum vertex of the Y-axis as the starting point.
<p align="center"><img src="./Images/table1.png" width =400px> </p>
<br/>

Comparison of the accuracy obtained from the proposed method (MVCNN-Geodesic) with other classification methods on the two datasets of archaeological ceramics (Peruvian and 3D Pottery). 
<br/>

<p align="center"><img src="./Images/table2.png" width =500px> </p>