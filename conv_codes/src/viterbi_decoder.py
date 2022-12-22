import numpy as np
from sk_dsp_comm import fec_conv


class ViterbiDecoder:
    def __init__(self, transfer_matrix: list[list[int]]) -> None:
        # convert transfer_matrix list[list[int]] into package format tuple(string)
        polys = ()
        for i in range(len(transfer_matrix)):
            polys += ("".join(str(val) for val in transfer_matrix[i]),)

        self.FECConv = fec_conv.FECConv(polys, Depth=15)

    def decode(self, data: np.ndarray) -> np.ndarray:
        # call decoder from imported package
        data = data.astype(np.uint8)
        return self.FECConv.viterbi_decoder(data, "hard")


if __name__ == "__main__":
    from conv_codes import FeedForwardEncoder

    en = FeedForwardEncoder([[1, 1, 1], [1, 0, 1]])
    data = np.random.choice((0, 1), size=64)
    coded = en.transform(data)
    de = ViterbiDecoder([[1, 1, 1], [1, 0, 1]])
    print(data)
    # print(de.decode(coded).astype(np.uint8))
    print(de.decode(coded[: coded.size - 4]).astype(np.uint8))
