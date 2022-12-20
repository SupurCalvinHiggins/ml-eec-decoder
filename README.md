# Getting Started

## Necessary packages

This project relies on a patched version of the scikit-dsp-comm package. The documentation for the package can be found [here](https://scikit-dsp-comm.readthedocs.io/en/latest/index.html). 

In order to run the project, you will need to install the patched package by doing the following:

1. Navigate to the project directory and clone the package repository here.

```bash
cd <project directory>
git clone https://github.com/SupurCalvinHiggins/scikit-dsp-comm.git
```

2. Rename the new directory to use '_' instead of '-'. 

```bash
mv scikit-dsp-comm scikit_dsp_comm
```

3. Navigate into the new directory and install it. 
   
```bash
cd scikit_dsp_comm
pip install -e .
```