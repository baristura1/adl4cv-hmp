
<div align="center">
<h1>Graph Scattering Network for Human Motion Prediction</h1>
<h3> <i>Baris Tura, Vitaliy Rusinov</i></h3>
 <h4> <i>Technical University of Munich</i></h4>
 

</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">
The task of human motion prediction is concerned with forecasting a sequence of motion in the future given the past motion sequences. The difficulty of addressing this issue lies in the fact that it is difficult to embed both spatial and temporal interdependencies of the joints and carry over the information from the history while still leaving room for unseen motion modalities. Some models may be prone to overfitting and oversmoothing. We propose a novel graph neural network architecture that includes a graph scattering network with spectral attention applied to a spatiotemporal graph that encodes human motion. We demonstrate that both the wavelets and the spectral attention help significantly reduce the mean per joint position error. We propose the use of wavelet filters with an undirected learnable adjacency matrix. We show that our model outperforms the state-of-the-art model for deterministic human motion prediction using certain configurations of the graph adjacency matrix in long-term predictions on the Human3.6M dataset.
</div>
--------

### Models

model.py -- Sandwich Architecture  
model2.py -- Space-Time Seperable Adjacency  
model3.py -- Space-Time Seperable Scattering, Shared Adjacencies  
model4.py -- Space-Time Seperable Scattering  
model5.py -- Full Adjacency  
model6.py -- Full Adjacency + Weight  
model_7_gcnii.py -- GCNII  
model_scatter.py -- Scattering, Normalized Ajacencies  
model_scatter_2.py -- Scattering, Non-Normalized Adjacencies  
model_scatter_3.py -- Scattering, Undirected Adjacencies in Wavelets  
model_scatter_4.py -- Scattering, Undirected Adjacency Matrix  
model_scatter_5.py -- Scattering, Undirected Adjacencies in Wavelet, Diff Attention Mech.  


 ### Install dependencies:
```
 $ pip install -r requirements.txt
```
 
 ### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).
 
Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

Extract the downloaded dataset in ./datasets directory.

### Train
The arguments for running the code are defined in [parser.py](utils/parser.py). We have used the following commands for training the network,on different datasets and body pose representations(3D and euler angles):
 
```bash
 python main_h36_3d_3.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 
 ```

 ### Test
 To test on the pretrained model, we have used the following commands:
 ```bash
 python main_h36_3d_3.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 --mode test --model_path ./checkpoints/CKPT_3D_H36M
  ```

### Visualization
 For visualizing from a pretrained model, we have used the following commands:
 ```bash
  python main_h36_3d_3.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5
 ```

 
 ### Acknowledgments
 
 Some of our code was adapted from [HisRepsItself](https://github.com/wei-mao-2019/HisRepItself) by [Wei Mao](https://github.com/wei-mao-2019) and from [STSGCN](https://github.com/FraLuca/STSGCN) by [Luca Franco](https://github.com/FraLuca).

