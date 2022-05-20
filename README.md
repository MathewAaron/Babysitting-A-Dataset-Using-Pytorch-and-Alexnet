
## How to run the code?

### Dataset:

    Extract the FDDB_dataset.zip. There will be two folders called training and testing respectively.
    
    **Copy the local directory till ../../dataset/ **
    
    *Change the argparse in the main.py file*
    
     parser.add_argument("directory", help="Directory of the dataset",nargs = '?',default = " *Add directory here*/dataset/")
    

### main.py

change the sys.append() to the directory of the project.

### Run the code:

python3 main.py 

### The default argparser will run for FDDB dataset, with SGD optimizer. Change the optimizer to SGD in the default value.

## The code for CIFAR10 is run on google collab due to GPU.