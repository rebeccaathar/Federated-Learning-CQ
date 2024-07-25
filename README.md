

# Client Selection methods

To use client selection methods, use the `-cs` flag

- `random`: Random Selection.
- `rr`: Round Robin.
- `cq`: Based on channel quality where users who have a communication channel with less noise will have a higher chance of being selected.
- `entropy`: Based on entropy metrics following the paper [*"Entropy-based Client Selection Mechanism for Vehicular Federated Environments"*](https://doi.org/10.5753/wperformance.2023.230700).

## Example
```
cd ./system
python main.py -data MNIST -m cnn -algo FedAvg -gr 2000 -cs {client selection method you want} # using the MNIST dataset, the FedAvg algorithm, and the 4-layer CNN model
```

# Channel

The metrics generated in the `system/channel/channel_metrics.csv` file were generated from a simulation using [Sionna](https://developer.nvidia.com/sionna), an Open-Source Library for 6G Physical-Layer Research. 


#### Repository created from [PFLlib library](https://github.com/TsingZ0/PFLlib/tree/master).
