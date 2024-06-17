## Molecule generation loop benchmark
A benchmark for testing of different molecule generation loops.

Loop implementations are tested on the `ZINC` dataset.


### Usage
To start the benchmark, run the following command:

Example command to run the benchmark:
```bash
python main.py num_epochs=5 model=GraphNeuralNetwork loop=gnn_loop
```

### Configuration
Possible configurations and command line arguments are located in the `config` directory.

The configuration is implemented using the [Hydra](https://hydra.cc/) library through Python dataclasses.

### Custom model and loop implementations
To add a custom model or loop implementation. 

- Create a new file in the `models` or `loops` directory with the implementatin of the model or loop.
- Register the model or loop with the `@register_model` or `@register_loop` decorator.
- Create a new configuration file in the `config` directory with the new model or loop, following the existing configuration files as an example.