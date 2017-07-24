# Nest

## Dependencies
```
pip install dill tqdm tensorflow
```

## Default Run
- Run the entire pipeline, including instance matching with motifs, training and evaluating 
```
python maim.py
```
- Only match instances with motifs
```
python preprocess.py
```

## Parameters
- To change dataset, modify the data_dir parameter in flags in main.py
- kernel.json under each dataset directory defines the kernels to be matched, modify it to customize the kernels
- For details of hyper-parameters, please refer to the comment in flags in main.py


## Dataset
- graph.txt contains the edge list of the complete graph, graph is undirected
- train.txt contains the training data, each line is a data point, each data point is a subgraph
- train/subgraph/ contains all the data points, one data point per file, each represented as an edge list
- train/meta/ contains all the matched instances of kernels, one data point per file
