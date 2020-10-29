# Paper Material
### Experiments for the paper ´experimental transferability of graph neural networks´

Most of the structure of the code are simplifications made on the benchmarking-gnns code base that can be found at this LINK
## Reproducibility

In order to launch all of the simulations, use the command `shell main_exec.sh`

Add docker environment?

## Repository structure
    .
    ├── Configs file        # Containing the configuration for the model for each task
    ├── Data 
            ├──  Molecules
            └──  SBMS
    ├── Layers              # Definition of the ChebNet layer
    ├── nets                # Definition of the structure of the NNs for each model
    ├── train               # Trainig script for each task
    ├── scripts             #sets of scripts to run individually eachtask
    ├── LICENSE
    └── README.md
