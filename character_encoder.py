import torch
from torch import nn

from character_decomposer import BoundingBox
from dataset import Radical


encode_type = "bounding-box"
print(f"character encode_type: {encode_type}")

if encode_type == "bounding-box":

    # https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/unet.py#L711
    class CharacterEncoder(nn.Module):
        def __init__(
            self,
            num_radicals: int,
            embedding_dim: int,
            len_radicals_of_char: int,
        ):
            super(CharacterEncoder, self).__init__()
            
            self.embedding_dim = embedding_dim
            self.len_radicals_of_char = len_radicals_of_char

            self.rademb_dim = embedding_dim - 4
            self.posemb_dim = 4
            
            self.rademb_padding_idx = num_radicals
            self.rademb_layer = nn.Embedding(num_radicals + 1, self.rademb_dim, max_norm=1, padding_idx=self.rademb_padding_idx)

        @property
        def device(self) -> torch.device:
            return self.rademb_layer.weight.device
        
        def embed_radical(self, batch_char: list[list[Radical]]) -> torch.Tensor:
            batch_radicalindices = []
            for char in batch_char:
                radicalindices = [r.idx for r in char]
                radicalindices += [self.rademb_padding_idx] * (self.len_radicals_of_char - len(radicalindices))
                batch_radicalindices.append(radicalindices)
            
            batch_radicalindices = torch.tensor(batch_radicalindices, dtype=torch.long, device=self.device)

            batch_embedding = self.rademb_layer(batch_radicalindices)
            return batch_embedding # (batch_size, self.len_radicals_of_char, self.embedding_dim)

        def embed_position(self, batch_char: list[list[Radical]]) -> torch.Tensor:
            batch_embedding = []
            for char in batch_char:
                embedding = []
                for radical in char:
                    assert isinstance(radical.position, BoundingBox)
                    embedding.append([radical.position.center_x, radical.position.center_y, radical.position.width, radical.position.height])
                embedding += [[0, 0, 0, 0]] * (self.len_radicals_of_char - len(embedding))
                batch_embedding.append(embedding)
            
            batch_embedding = torch.tensor(batch_embedding, dtype=torch.float, device=self.device)
            return batch_embedding # (batch_size, self.len_radicals_of_char, self.embedding_dim)

        def forward(self, batch_char: list[list[Radical]]):
            batch_rademb = self.embed_radical(batch_char)
            batch_posemb = self.embed_position(batch_char)
            batch_embedding = torch.cat((batch_rademb, batch_posemb), dim=2)
            return batch_embedding # (batch_size, char_length, hidden_size)


else:
    raise Exception()
