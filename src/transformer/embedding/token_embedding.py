import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """Token Embedding."""

    def __init__(self, vocab_size: int, d_model: int, pad_idx: int):
        super(TokenEmbedding, self).__init__(
            num_embeddings=vocab_size,
            embedding_dim=pad_idx,
            padding_idx=pad_idx,
        )
