## Molecule generation loop benchmark
A benchmark for testing of different molecule generation loops.

Loop implementations are tested on the `ZINC` dataset.


### Usage
To start the benchmark, you need to provide the necessary configuration parameters through the command line.

The benchmark is started by running the `main.py` script with the desired configuration parameters.

Currently, the required configuration parameters are:
- `model` - the model to use for the benchmark
- `loop` - the loop to use for the benchmark

Example command to run the benchmark:
```bash
python main.py model=GraphNeuralNetwork loop=gnn_loop
```

### Configuration
Possible configurations and command line arguments are located in the `config` directory.

The configuration is implemented using the [Hydra](https://hydra.cc/) library through Python dataclasses.

### Custom model and loop implementations
To add a custom model or loop implementation. 

- Create a new file in the `models`, `loops` or `embeddings` directory with the implementatin of the model or loop.
- Create a new configuration file in the `config` directory with the new model or loop, following the existing configuration files as an example.
- Register the model, loop or embedding with a `@register_model`, `@register_loop` or `@register_embedding` decorator.

### Results analysis
The benchmark results are saved in the `outputs` directory. For analysis of the results, you can use the `results_analysis.ipynb` Jupyter notebook.
