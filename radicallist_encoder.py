import torch
from torch import nn

from character_decomposer import BoundingBox
from dataset import Radical
from radical import ClusteringLabel


# 文字を分解しないとき，1 つの字が 1 つの部首からなるとみなして位置なしの部首埋め込みを行う
class NonpositionalRadicallistEncoder(nn.Module):
    def __init__(
        self,
        num_radicals: int,
        embedding_dim: int,
        len_radicals_of_char: int,
    ):
        super(NonpositionalRadicallistEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim

        if len_radicals_of_char != 1:
            raise Exception("len_radicals_of_char must be 1")

        self.rademb_dim = embedding_dim
        self.rademb_layer = nn.Embedding(num_radicals, self.rademb_dim, max_norm=1)

    @property
    def device(self) -> torch.device:
        return self.rademb_layer.weight.device
    
    def forward(self, batch_radicallist: list[list[Radical]]):
        batch_radicalindices = []
        for radicallist in batch_radicallist:
            assert len(radicallist) == 1
            assert radicallist[0].position is None
            batch_radicalindices.append([radicallist[0].idx])
        
        batch_radicalindices = torch.tensor(batch_radicalindices, dtype=torch.long, device=self.device)

        batch_embedding = self.rademb_layer(batch_radicalindices)
        return batch_embedding # (batch_size, 1, self.rademb_dim)


# https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/unet.py#L711
class BoundingBoxRadicallistEncoder(nn.Module):
    def __init__(
        self,
        num_radicals: int,
        embedding_dim: int,
        len_radicals_of_char: int,
    ):
        super(BoundingBoxRadicallistEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.len_radicals_of_char = len_radicals_of_char

        self.rademb_dim = embedding_dim - 4
        self.posemb_dim = 4
        
        self.rademb_padding_idx = num_radicals
        self.rademb_layer = nn.Embedding(num_radicals + 1, self.rademb_dim, max_norm=1, padding_idx=self.rademb_padding_idx)

    @property
    def device(self) -> torch.device:
        return self.rademb_layer.weight.device
    
    def embed_radical(self, batch_radicallist: list[list[Radical]]) -> torch.Tensor:
        batch_radicalindices = []
        for radicallist in batch_radicallist:
            radicalindices = [self.rademb_padding_idx for _ in range(self.len_radicals_of_char)]
            for i_r, radical in enumerate(radicallist):
                radicalindices[i_r] = radical.idx
            batch_radicalindices.append(radicalindices)

        batch_radicalindices = torch.tensor(batch_radicalindices, dtype=torch.long, device=self.device)

        batch_embedding = self.rademb_layer(batch_radicalindices)
        return batch_embedding # (batch_size, self.len_radicals_of_char, self.rademb_dim)

    def embed_position(self, batch_radicallist: list[list[Radical]]) -> torch.Tensor:
        batch_embedding = []
        for radicallist in batch_radicallist:
            embedding = []
            for radical in radicallist:
                assert isinstance(radical.position, BoundingBox)
                embedding.append([radical.position.center_x, radical.position.center_y, radical.position.width, radical.position.height])
            embedding += [[0, 0, 0, 0]] * (self.len_radicals_of_char - len(embedding))
            batch_embedding.append(embedding)

        batch_embedding = torch.tensor(batch_embedding, dtype=torch.float, device=self.device)
        return batch_embedding # (batch_size, self.len_radicals_of_char, self.posemb_dim)

    def forward(self, batch_radicallist: list[list[Radical]]):
        batch_rademb = self.embed_radical(batch_radicallist)
        batch_posemb = self.embed_position(batch_radicallist)
        batch_embedding = torch.cat((batch_rademb, batch_posemb), dim=2)
        return batch_embedding # (batch_size, char_length, hidden_size)


class ClusteringLabelRadicallistEncoder(nn.Module):
    def __init__(
        self,
        num_radicals: int,
        embedding_dim: int,
        len_radicals_of_char: int,
    ):
        super(ClusteringLabelRadicallistEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.len_radicals_of_char = len_radicals_of_char

        self.rademb_dim = embedding_dim - 128
        self.posemb_dim = 128
        
        self.rademb_padding_idx = num_radicals
        self.rademb_layer = nn.Embedding(num_radicals + 1, self.rademb_dim, max_norm=1, padding_idx=self.rademb_padding_idx)
        self.posemb_padding_idx = 512
        self.posemb_layer = nn.Embedding(512 + 1, self.posemb_dim, max_norm=1)

    @property
    def device(self) -> torch.device:
        return self.rademb_layer.weight.device

    def embed_radical(self, batch_radicallist: list[list[Radical]]) -> torch.Tensor:
        batch_radicalindices = []
        for radicallist in batch_radicallist:
            radicalindices = [self.rademb_padding_idx for _ in range(self.len_radicals_of_char)]
            for i_r, radical in enumerate(radicallist):
                radicalindices[i_r] = radical.idx
            batch_radicalindices.append(radicalindices)

        batch_radicalindices = torch.tensor(batch_radicalindices, dtype=torch.long, device=self.device)

        batch_embedding = self.rademb_layer(batch_radicalindices)
        return batch_embedding # (batch_size, self.len_radicals_of_char, self.rademb_dim)

    def embed_position(self, batch_radicallist: list[list[Radical]]) -> torch.Tensor:
        batch_labels = []
        for radicallist in batch_radicallist:
            labels = [self.posemb_padding_idx for _ in range(self.len_radicals_of_char)]
            for i_r, radical in enumerate(radicallist):
                assert isinstance(radical.position, ClusteringLabel)
                labels[i_r] = radical.position.label
            batch_labels.append(labels)

        batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=self.device)

        batch_embedding = self.posemb_layer(batch_labels)
        return batch_embedding # (batch_size, self.len_radicals_of_char, self.posemb_dim)

    def forward(self, batch_radicallist: list[list[Radical]]):
        batch_rademb = self.embed_radical(batch_radicallist)
        batch_posemb = self.embed_position(batch_radicallist)
        batch_embedding = torch.cat((batch_rademb, batch_posemb), dim=2)
        return batch_embedding # (batch_size, char_length, hidden_size)
