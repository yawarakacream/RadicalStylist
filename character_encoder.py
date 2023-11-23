import math

import torch
from torch import nn

from dataset import Radical


encode_type = 3
print(f"character encode_type={encode_type}")


if encode_type == 0:
    
    # https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/unet.py#L688
    class CharacterAttention(nn.Module): # type: ignore
        def __init__(self, input_size, hidden_size):
            super(CharacterAttention, self).__init__() # type: ignore
            self.query = nn.Linear(input_size, hidden_size)
            self.key = nn.Linear(input_size, hidden_size)
            self.value = nn.Linear(input_size, hidden_size)
            self.softmax = nn.Softmax(dim=-1)
            
        def forward(self, x):
            """
            :param x: (batch_size, sequence_length, input_size)
            """
            
            query = self.query(x)
            key = self.key(x)
            value = self.value(x)
            
            # Calculate attention scores
            scores = query @ key.transpose(-2, -1)
            scores = self.softmax(scores)
            
            # Calculate weighted sum of the values
            rads_embs = scores @ value
            return rads_embs


    # https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/unet.py#L711
    class CharacterEncoder(nn.Module): # type: ignore
        def __init__(self, input_size, hidden_size, char_length):
            import numpy as np

            super(CharacterEncoder, self).__init__() # type: ignore
            
            self.embedding_dim = hidden_size
            self.char_length = char_length
            
            self.embedding = nn.Embedding(input_size, hidden_size, max_norm=1)
            self.attention = CharacterAttention(hidden_size, hidden_size)
            
            # 部首埋め込み (正規化されている) に掛ける係数
            self.rad_emb_norm = 1
            
            # 位置埋め込みの精度
            self.pos_enc_precision = 1000
            
            # 位置埋め込みの前計算
            self.pos_enc_list = [] # [i][p * k] -> encoding tensor
            
            pos_enc_type = 4
            print(f"{pos_enc_type=}")
            
            if pos_enc_type == 0:
                self.rad_emb_norm = ((self.embedding_dim / 2) ** 0.5) * 4 # 埋め込みのノルムの分
                
                for i in range(4):
                    T = 1000 * (i + 1)
                    
                    self.pos_enc_list.append([])
                    for p in np.linspace(0, 1, self.pos_enc_precision):
                        self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))

                        for i in range(0, self.embedding_dim, 2):
                            theta = (p * self.embedding_dim) / (T ** (i / self.embedding_dim))
                            self.pos_enc_list[-1][-1][i] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i + 1] = math.cos(theta)

                        assert (
                            abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4) # type: ignore
                            < 1e-4
                        )
            
            elif pos_enc_type == 1:
                assert self.embedding_dim % 12 == 0
                
                self.rad_emb_norm = ((self.embedding_dim / 2) ** 0.5) * 4 # 埋め込みのノルムの分
                
                for _ in range(4):
                    T = 1000
                    
                    self.pos_enc_list.append([])
                    for p in np.linspace(0, 1, self.pos_enc_precision):
                        self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))
                        
                        for i in range(0, self.embedding_dim, 12):
                            theta = (p * self.embedding_dim) / (T ** (i / self.embedding_dim))
                            
                            self.pos_enc_list[-1][-1][i + 0] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i + 1] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i + 2] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i + 3] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i + 4] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i + 5] = math.sin(theta)
                            
                            self.pos_enc_list[-1][-1][i + 6] = math.cos(theta)
                            self.pos_enc_list[-1][-1][i + 7] = math.cos(theta)
                            self.pos_enc_list[-1][-1][i + 8] = math.cos(theta)
                            self.pos_enc_list[-1][-1][i + 9] = math.cos(theta)
                            self.pos_enc_list[-1][-1][i + 10] = math.cos(theta)
                            self.pos_enc_list[-1][-1][i + 11] = math.cos(theta)
                        
                        assert (
                            abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4) # type: ignore
                            < 1e-4
                        )
                    
                for i_p in range(len(self.pos_enc_list[0])):
                    for i in range(0, self.embedding_dim, 6):
                        self.pos_enc_list[0][i_p][i + 0] *= -1
                        self.pos_enc_list[1][i_p][i + 0] *= -1
                        
                        self.pos_enc_list[0][i_p][i + 1] *= -1
                        self.pos_enc_list[2][i_p][i + 1] *= -1
                        
                        self.pos_enc_list[0][i_p][i + 2] *= -1
                        self.pos_enc_list[3][i_p][i + 2] *= -1
                        
                        self.pos_enc_list[1][i_p][i + 3] *= -1
                        self.pos_enc_list[2][i_p][i + 3] *= -1
                        
                        self.pos_enc_list[1][i_p][i + 4] *= -1
                        self.pos_enc_list[3][i_p][i + 4] *= -1
                        
                        self.pos_enc_list[2][i_p][i + 5] *= -1
                        self.pos_enc_list[3][i_p][i + 5] *= -1
                        
            elif pos_enc_type == 2:
                assert self.embedding_dim % 4 == 0
                
                self.rad_emb_norm = ((self.embedding_dim / 2) ** 0.5) * 4 # 埋め込みのノルムの分
                
                for _ in range(4):
                    T = 1000
                    
                    self.pos_enc_list.append([])
                    for p in np.linspace(0, 1, self.pos_enc_precision):
                        self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))
                        
                        for i in range(0, self.embedding_dim, 4):
                            theta = (p * self.embedding_dim) / (T ** (i / self.embedding_dim))
                            
                            self.pos_enc_list[-1][-1][i + 0] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i + 1] = math.cos(theta)
                            self.pos_enc_list[-1][-1][i + 2] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i + 3] = math.cos(theta)
                        
                        assert (
                            abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4) # type: ignore
                            < 1e-4
                        )
                    
                for i_p in range(len(self.pos_enc_list[0])):
                    for i in range(0, self.embedding_dim, 4):
                        self.pos_enc_list[0][i_p][i + 0] *= -1
                        self.pos_enc_list[1][i_p][i + 1] *= -1
                        self.pos_enc_list[2][i_p][i + 2] *= -1
                        self.pos_enc_list[3][i_p][i + 3] *= -1
                        
            elif pos_enc_type == 3:
                assert self.embedding_dim % 8 == 0
                
                self.rad_emb_norm = ((self.embedding_dim / 8) ** 0.5) * 4 # 埋め込みのノルムの分
                
                k = self.embedding_dim // 4
                for d in range(4):
                    T = 1000
                    
                    self.pos_enc_list.append([])
                    for p in np.linspace(0, 1, self.pos_enc_precision):
                        self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))
                        
                        i_0 = d * k
                        for i in range(0, k, 2):
                            theta = p / (T ** (i / k))
                            
                            self.pos_enc_list[-1][-1][i_0 + i + 0] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i_0 + i + 1] = math.cos(theta)
                        
                        assert (
                            abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4) # type: ignore
                            < 1e-4
                        )
                        
            elif pos_enc_type == 4:
                assert self.embedding_dim % 8 == 0
                
                self.pos_enc_precision = self.embedding_dim - 1 # 位置埋め込みの精度
                self.rad_emb_norm = ((self.embedding_dim / 8) ** 0.5) * 4 # 埋め込みのノルムの分
                
                k = self.embedding_dim // 4
                for d in range(4):
                    T = 1000
                    
                    self.pos_enc_list.append([])
                    for p in np.linspace(0, 1, self.pos_enc_precision):
                        self.pos_enc_list[-1].append(torch.zeros(self.embedding_dim))
                        
                        i_0 = d * k
                        for i in range(0, k, 2):
                            theta = p / (T ** (i / k))
                            
                            self.pos_enc_list[-1][-1][i_0 + i + 0] = math.sin(theta)
                            self.pos_enc_list[-1][-1][i_0 + i + 1] = math.cos(theta)
                        
                        assert (
                            abs(torch.norm(self.pos_enc_list[-1][-1], 2) - self.rad_emb_norm / 4) # type: ignore
                            < 1e-4
                        )
        
        def forward(self, chars):
            """
            :param x: list[list[Radical]], length: batch_size
            """
            device = self.embedding.weight.device
            
            # バッチ毎にひとつひとつ計算 (最適化したい)
            rads_embs = []
            for char in chars:
                rad_embs = []
                
                for radical in char.radicals:
                    idx = torch.tensor([radical.idx], device=device)
                    
                    rad_emb = self.embedding(idx)
                    rad_emb *= self.rad_emb_norm
                    
                    for i, p in enumerate((radical.center_x, radical.center_y, radical.width, radical.height)):
                        encoding = self.pos_enc_list[i][int(p * self.pos_enc_precision)].to(device)
                        rad_emb += encoding
                    
                    rad_embs.append(rad_emb)
                    
                while len(rad_embs) < self.char_length:
                    rad_embs.append(torch.zeros(rad_embs[0].shape, device=device))
                
                rad_embs = torch.cat(rad_embs, dim=0)
                rads_embs.append(rad_embs)
            
            rads_embs = torch.stack(rads_embs)
            rads_embs = self.attention(rads_embs)
            return rads_embs # (batch_size, char_length, hidden_size)


elif encode_type == 3:

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
                    embedding.append([radical.center_x, radical.center_y, radical.width, radical.height])
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
