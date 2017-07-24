# Nest

## Dependencies
```
pip install dill tqdm tensorflow
```

## Default Run
- Run the entire pipeline, including instance matching with kernels, training and evaluating 
```
python maim.py
```
- Only match instances with kernels
```
python preprocess.py
```

## Parameters
- To change dataset, modify the data_dir parameter in flags in main.py
- kernel.json under each dataset directory defines the kernels to be matched, modify it to customize the kernels
- For details of hyper-parameters, please refer to the comment in flags in main.py
