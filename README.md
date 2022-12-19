# Getting Started

## Necessary packages

This project relies on the scikit-dsp-comm package. The documentation for the package can be found [here](https://scikit-dsp-comm.readthedocs.io/en/latest/index.html). 

In order to run the project, you will need to install this package [INSTRUCTIONS](https://scikit-dsp-comm.readthedocs.io/en/latest/readme.html#getting-set-up-on-your-system). The developers recommends the following:

1. Navigate to the project directory and clone the package repository here.

```bash
cd <project directory>
git clone https://github.com/mwickert/scikit-dsp-comm.git
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