import numpy as np


class TransformerPipeline:
    def __init__(self, transformers: list) -> None:
        assert len(transformers) >= 1
        self._transformers = transformers

    def transform(self, data: np.ndarray) -> np.ndarray:
        for transformer in self._transformers:
            data = transformer.transform(data)
        return data
