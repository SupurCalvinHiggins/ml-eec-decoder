import numpy as np
from numpy.random import randint
from scikit_dsp_comm.sk_dsp_comm import fec_conv, digitalcom
from conv_codes import ViterbiDecoder

import unittest


class TestDecoder(unittest.TestCase):
    def test_wrapper_against_package(self) -> None:
        num_tests = 1000
        transfer_matrix = [[1, 1, 1, 0, 1], [1, 0, 0, 1, 1]]
        vdecode = ViterbiDecoder(transfer_matrix)

        # Lines 15 to 28 are from package documentation
        cc1 = fec_conv.FECConv(("11101", "10011"))
        state = "0000"

        for i in range(num_tests):
            # Create and encode 200 random bits
            x = randint(0, 2, 200)
            y, state = cc1.conv_encoder(x, state)

            # Add channel noise to bits
            yn_soft = digitalcom.cpx_awgn(2 * y - 1, 1, 1)
            yn_hard = ((np.sign(yn_soft.real) + 1) / 2).astype(int)

            # decode with package and wrapper's methods and compare
            expected = cc1.viterbi_decoder(yn_hard, "hard")
            test = vdecode.decode(yn_hard)

            assert np.array_equal(expected, test)


if __name__ == "__main__":
    unittest.main()
