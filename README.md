# Bi-LSTM Decoder for Convolutional Codes

A proof of concept Bi-LSTM decoder for convolutional codes on bursty binary channels. 

## Overview

This project contains implementation for convolutional code encoders, decoders and several different channels. The channels are as follows:

1. Binary Symmetric Channel
2. Definite Binary Symmetric Channel
3. Bursty Binary Symmetric Channel

This project also contains scripts to build and train Bi-LSTM convolutional code decoders.

## Installation

### Git

This project requires a Git installation. See [here](https://git-scm.com/downloads) for details on installing Git.

### Python

This project requires Python 3.9.6. See [here](https://www.python.org/downloads/) for details on installing Python 3.9.6.

### Python Dependencies

The majority of the Python dependencies are listed in requirements.txt. Installation instructions for these dependcies are given below.

Navigate to the root directory of the project and execute the following commands. 

```bash
pip install -r requirements.txt
```

The project does not have a standalone Viterbi decoder implementation. The decoder class relies on a modified version of the scikit-dsp-comm package. Installation instructions for the modified package are given below. 

Navigate to the root directory of the project and execute the following commands. 

```bash
git clone https://github.com/SupurCalvinHiggins/scikit-dsp-comm.git
cd scikit_dsp_comm
pip install -e .
```

After executed the commands, the modified version of the package should be installed. The motivation behind the modified package is provided below.

The decoder in the default scikit-dsp-comm package is a ``streaming" convolutional code decoder. That is, the decoder assumes that a continuous stream of symbols will be received and will wait to decode symbols until a certain number of subsequent symbols have been received. This prevents establishing baseline performance metrics because the decoder will refuse to decode the entire message. The modified version of the scikit-dsp-comm package ensures that the decoder will finish decoding the entire message.

## Testing

This project has partial smoke test coverage using the pytest framework.

### Executing Tests

Navigate to the root directory of the project and execute the following commands. 

```bash
pytest .
```

A lack of errors does not indicate that the project is working properly. However, the presence of errors DOES indicate that project is NOT working properly. 

## Benchmarking

This project has partial benchmark coverage. Benchmarks can be executed directly with Python.
