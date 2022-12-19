import numpy as np
from scikit_dsp_comm import sk_dsp_comm
from scikit_dsp_comm.sk_dsp_comm import fec_conv


class ViterbiDecoder:
    def __init__(self, transfer_matrix: list[list[int]]) -> None:
        # convert transfer_matrix list[list[int]] into package format tuple(string)
        polys = ()
        for i in range(len(transfer_matrix)):
            polys += "".join(str(val) for val in transfer_matrix[i]),

        self.FECConv = fec_conv.FECConv(polys)

    def decode(self, data: np.ndarray) -> np.ndarray:
        # call decoder from imported package
        return self.FECConv.viterbi_decoder(data, 'hard')


