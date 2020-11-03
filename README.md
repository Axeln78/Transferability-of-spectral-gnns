# Paper Material
### Experiments for the paper ´experimental transferability of graph neural networks´

Most of the structure of the code are simplifications made on the benchmarking-gnns code base that can be found at this LINK
## Reproducibility

In order to launch all of the simulations, use the command `shell main_exec.sh`

Add docker environment?
Setup your environment:

    conda env create -f environment_gpu.yml

OGB:

    python main_dgl.py --dataset $DATASET --gnn Cheb_net --filename $FILENAME
    
Benchmarking_gnns:
- First you will need to download the datasets

    cd Benchmark-gnn/data
    bash script_download_all_datasets.sh
    
    
## Repository structure
    .
    ├── Benchmark-gnn/
        ├── Configs file        # Containing the configuration for the model for each task
        ├── Data/ 
                ├──  Molecules
                └──  SBMS
        ├── Layers              # Definition of the ChebNet layer
        ├── nets                # Definition of the structure of the NNs for each model
        ├── train               # Trainig script for each task
        └── scripts             # Sets of scripts to run individually eachtask
    ├── OGB/
        ├── Dataset/
        ├── gnn_dgl.py          # Model definiton
        └── main_dgl.py         # Model training anf testing script
    ├── LICENSE
    └── README.md