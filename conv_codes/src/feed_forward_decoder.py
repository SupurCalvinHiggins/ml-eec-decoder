import numpy as np
from sk_dsp_comm import fec_conv


class FeedForwardDecoder:
    def __init__(self, transfer_matrix: list[list[int]]) -> None:
        # convert transfer_matrix list[list[int]] into package format tuple(string)
        polys = ()
        for i in range(len(transfer_matrix)):
            polys += ("".join(str(val) for val in transfer_matrix[i]),)

        self._polys = polys
        self._decision_depth = len(transfer_matrix[0]) * 5

    def transform(self, data: np.ndarray) -> np.ndarray:
        # call decoder from imported package
        data = data.astype(np.uint8)
        decoder = fec_conv.FECConv(self._polys, Depth=self._decision_depth)
        return decoder.viterbi_decoder(data, "hard")


if __name__ == "__main__":
    from conv_codes import FeedForwardEncoder

    en = FeedForwardEncoder([[1, 1, 1], [1, 0, 1]])
    data = np.random.choice((0, 1), size=64)
    coded = en.transform(data)
    de = FeedForwardDecoder([[1, 1, 1], [1, 0, 1]])
    print(data)
    # print(de.decode(coded).astype(np.uint8))
    print(de.decode(coded[: coded.size - 4]).astype(np.uint8))
