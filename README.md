
# Rethinking Large Language Model-Based Optimization for Mixed-Integer Linear Programming


## Running the experiments

### Set Covering
```
# Generate MILP instances
python 01_generate_instances.py setcover

# Test
python 06_llm-evaluate.py setcover

```

### Combinatorial Auction
```
# Generate MILP instances
python 01_generate_instances.py cauctions

# Test
python 06_llm-evaluate.py cauctions

```

### Capacitated Facility Location
```
# Generate MILP instances
python 01_generate_instances.py facilities

# Test
python 06_llm-evaluate.py facilities
# Evaluation
python 05_evaluate.py facilities
```

### Maximum Independent Set
```
# Generate MILP instances
python 01_generate_instances.py indset

# Test
python 06_llm-evaluate.py indset
# Evaluation
python 05_evaluate.py indset
```


## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.

