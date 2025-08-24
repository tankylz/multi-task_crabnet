"""Contains classes for transformer architecture within CrabNet."""
from os.path import join, dirname

import numpy as np
import pandas as pd

import torch
from torch import nn
from collections import OrderedDict
from typing import Optional

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32
how_to_extend_ls = ['tile_at_input', 'concat_at_input', 'concat_at_output', None]
one_hot_layer_ls = ['concat_at_attn']


# %%
class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.

    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(self, input_dim, output_dim=3, hidden_layer_dims=[1024, 512, 256, 128], bias=False, num_properties=1, one_hot_layer=None):
        """Instantiate a ResidualNetwork model.
        
        Parameters
        ----------
        input_dim : int
            Dimensions of input layer

        output_dim : int
            Dimensions of output layer, by default 3

        hidden_layer_dims : list(int)
            Dimensions of hidden layers, by default [1024, 512, 256, 128]

        bias : bool
            Whether to bias the skip connections, by default False

        num_properties : int
            Number of properties to predict, by default 1. Values greater than 1 indicate multi-task learning.

        one_hot_layer : int
            Layer to concatenate one-hot vectors, by default None. If multi-task learning, this must be specified. Negative indices supported. 0 means the input to the first layer, -1 means the input to the last layer. 'concat_at_attn' is also supported, meaning that the one-hot vector is concatenated to the input of the attention layer.

        Raises
        ------ 
        ValueError
            If one_hot_layer is not specified for multi-task learning

        ValueError
            If one_hot_layer exceeds the number of layers
        
        """
        super().__init__()
        self.num_properties = num_properties

        if isinstance(one_hot_layer, int) and one_hot_layer < 0:
            one_hot_layer = len(hidden_layer_dims) + 1 + one_hot_layer
        
        self.one_hot_layer = one_hot_layer

        if isinstance(self.one_hot_layer, int) and (self.one_hot_layer > len(hidden_layer_dims) or self.one_hot_layer < 0):
            raise ValueError("one hot layer cannot exceed the number of layers")

        dims = [input_dim] + hidden_layer_dims
        
        def get_dim(i):
            return dims[i] + (num_properties if num_properties > 1 and i == self.one_hot_layer else 0)

        self.fcs = nn.ModuleList([nn.Linear(get_dim(i), dims[i + 1]) for i in range(len(hidden_layer_dims))])
        self.res_fcs = nn.ModuleList([nn.Linear(get_dim(i), dims[i + 1], bias=bias) if get_dim(i) != dims[i + 1] else nn.Identity() for i in range(len(hidden_layer_dims))])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(hidden_layer_dims))])
        self.fc_out = nn.Linear(get_dim(len(hidden_layer_dims)), output_dim)

    def forward(self, fea, property_one_hot=None):
        """Propagate Residual Network weights forward."""
        if property_one_hot is None and self.num_properties > 1:
            raise ValueError("property_one_hot must be provided for multi-task learning")

        for i, (fc, res_fc, act) in enumerate(zip(self.fcs, self.res_fcs, self.acts)):
            if i == self.one_hot_layer:
                property_one_hot = property_one_hot.to(fea.device).unsqueeze(1).expand(-1, fea.shape[1], -1)
                fea = torch.cat((fea, property_one_hot), dim=-1)
            fea = act(fc(fea)) + res_fc(fea)

        if self.num_properties > 1 and self.one_hot_layer == len(self.fcs): # concatenate at the last hidden layer
            property_one_hot = property_one_hot.to(fea.device).unsqueeze(1).expand(-1, fea.shape[1], -1)
            fea = torch.cat((fea, property_one_hot), dim=-1)

        return self.fc_out(fea)

    def __repr__(self):
        """Return the class name."""
        return f"{self.__class__.__name__}"


class TransferNetwork(nn.Module):
    """Learn extended representations of materials during transfer learning.

    This network was designed to have little impact on predictions during
    training and enhance learning with the inclusion of extended features.
    """

    def __init__(self, input_dims, output_dims):
        """Instantiate a TransferNetwork to learn extended representations.

        Parameters
        ----------
        input_dims : int
            Dimensions of input layer

        output_dims : int
            Dimensions of output layer
        """
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(input_dims, 512)),
                    ("leakyrelu1", nn.LeakyReLU()),
                    ("fc2", nn.Linear(512, output_dims)),
                    ("leakyrelu2", nn.LeakyReLU()),
                ]
            )
        )

    def forward(self, x):
        """Perform a forward pass of the TransferNetwork.

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = self.layers(x)
        return x


class Embedder(nn.Module):
    """Perform composition-based embeddings of elemental features."""

    def __init__(
        self,
        d_model: int,
        compute_device: str = None,
        elem_prop: str = "mat2vec",
    ):
        """Embed elemental features, similar to CBFV.

        Parameters
        ----------
        d_model : int
            Row dimenions of elemental emeddings, by default 512
        compute_device : str
            Name of device which the model will be run on
        elem_prop : str
            Which elemental feature vector to use. Possible values are "jarvis",
            "magpie", "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by
            default "mat2vec"
        """
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = join(dirname(__file__), "data", "element_properties")
        # Choose what element information the model receives
        mat2vec = join(elem_dir, elem_prop + ".csv")  # element embedding
        # mat2vec = f'{elem_dir}/onehot.csv'  # onehot encoding (atomic number)
        # mat2vec = f'{elem_dir}/random_200.csv'  # random vec for elements

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        # NOTE: Parameters within nn.Embedding
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(
            self.compute_device, dtype=data_type_torch
        )

    def forward(self, src):
        """Compute forward call for embedder class to perform elemental embeddings.

        Parameters
        ----------
        src : torch.tensor
            Tensor containing element numbers corresponding to elements in compound

        Returns
        -------
        torch.tensor
            Tensor containing elemental embeddings for compounds, reduced to d_model dimensions
        """
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


# %%
class FractionalEncoder(nn.Module):
    """Encode element fractional amount using a "fractional encoding".

    This is inspired by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        """Instantiate the FractionalEncoder.

        Parameters
        ----------
        d_model : int
            Model size, see paper, by default 512
        resolution : int
            Number of discretizations for the fractional prevalence encoding, by default 100
        log10 : bool
            Whether to apply a log operation to fraction prevalence encoding, by default False
        compute_device : str
            The compute device to store and run the FractionalEncoder class
        """
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(
            0, self.resolution - 1, self.resolution, requires_grad=False
        ).view(self.resolution, 1)
        fraction = (
            torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False)
            .view(1, self.d_model)
            .repeat(self.resolution, 1)
        )

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer("pe", pe)

    def forward(self, x):
        """Perform the forward pass of the fractional encoding.

        Parameters
        ----------
        x : torch.tensor
            Tensor of linear spaced values based on defined resolution

        Returns
        -------
        out
            Sinusoidal expansions of elemental fractions
        """
        x = x.clone()
        if self.log10:
            # todo: could explore using a higher base log for dopants
            # x = 0.0025 * (torch.log2(x)) ** 2 # original 
            # x = 0.025 * (torch.log2(x)) ** 2 # edited 1
            x = (torch.log10(x)) ** 2 # edited 2
            x[x > 1] = 1 # max value clamp, for cases of extremely small dopant values
        x[x < 1 / self.resolution] = 1 / self.resolution # Minimum value clamp
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out
    
class ExtraEncoder(nn.Module):
    """Encode an extra single value feature (e.g., temperature, bandgap) using sinusoidal encoding.

    This is inspired by the positional encoder discussed by Vaswani.
    """

    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        """Instantiate the ExtraEncoder.

        Parameters
        ----------
        d_model : int
            Model size, by default 512
        resolution : int
            Number of discretizations for the feature encoding, by default 100
        log10 : bool
            Whether to apply a log operation to feature encoding, by default False
        compute_device : str
            The compute device to store and run the ExtraEncoder class
        """
        super().__init__()
        self.d_model = d_model // 2 # we will use two ExtraEnconders, one for linear and one for log10
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        # Create linearly spaced values and fractions similar to the original encoder
        x = torch.linspace(0, self.resolution - 1, self.resolution, requires_grad=False).view(self.resolution, 1)
        fraction = (
            torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False)
            .view(1, self.d_model)
            .repeat(self.resolution, 1)
        )

        # Create the sinusoidal encoding
        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        self.register_buffer("pe", pe) # Represents constants that are not trainable

    def forward(self, feature_value):
        """Perform the forward pass of the extra feature encoding.

        Parameters
        ----------
        feature_value : torch.tensor
            A tensor containing the extra feature value (e.g., temperature) for each sample

        Returns
        -------
        out : torch.tensor
            Sinusoidal expansions of the input feature
        """
        # # Normalize the feature value to the resolution
        # feature_value = feature_value.clone()
        # feature_value = feature_value / (feature_value.max() + 1e-8)  # Min-max normalize to [0, 1] range
        
        if self.log10:
            # feature_value = 0.0025 * (torch.log2(feature_value)) ** 2 # original 
            # feature_value = 0.025 * (torch.log2(feature_value)) ** 2 # edited 1
            feature_value = (torch.log10(feature_value)) ** 2 # edited 2
        feature_value[feature_value > 1] = 1 # maximum value clamp

        feature_value[feature_value < 1 / self.resolution] = 1 / self.resolution  # Minimum value clamp
        feature_idx = torch.round(feature_value * (self.resolution)).to(dtype=torch.long) - 1
        
        # Look up the sinusoidal encoding for the feature value
        out = self.pe[feature_idx]

        return out


# %%
class Encoder(nn.Module):
    """Create elemental descriptor matrix via element embeddings and frac. encodings.

    See the CrabNet paper for further details:
    https://www.nature.com/articles/s41524-021-00545-1
    """

    def __init__(
        self,
        d_model,
        N,
        heads,
        extend_features=None,
        how_to_extend=None,
        fractional=True,
        attention=True,
        compute_device=None,
        extra_enc_resolution=5000,
        extra_enc_log_resolution=5000,
        pe_resolution=5000,
        ple_resolution=5000,
        elem_prop="mat2vec",
        emb_scaler=1.0,
        pos_scaler=1.0,
        pos_scaler_log=1.0,
        extra_scaler=1.0,
        extra_scaler_log=1.0,
        dim_feedforward=2048,
        dropout=0.1,
        num_properties=1,
        one_hot_layer=None
    ):
        """Instantiate the Encoder class to create elemental descriptor matrix (EDM).

        Parameters
        ----------
        d_model : _type_
            _description_
        N : int, optional
            Number of encoder layers, by default 3
        heads : int, optional
            Number of attention heads to use, by default 4
        extend_features : Optional[List[str]]
            Additional features to grab from columns of the other DataFrames (e.g. state
            variables such as temperature or applied load), by default None
        fractional : bool, optional
            Whether to weight each element by its fractional contribution, by default True.
        attention : bool, optional
            Whether to perform self attention, by default True
        pe_resolution : int, optional
            Number of discretizations for the prevalence encoding, by default 5000
        ple_resolution : int, optional
            Number of discretizations for the prevalence log encoding, by default 5000
        elem_prop : str, optional
            Which elemental feature vector to use. Possible values are "jarvis",
            "magpie", "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by
            default "mat2vec"
        emb_scaler : float, optional
            _description_, by default 1.0. Note that this is just a starting value and the model will learn the optimal value.
        pos_scaler : float, optional
            Scaling factor applied to fractional encoder, by default 1.0. Note that this is just a starting value and the model will learn the optimal value.
        pos_scaler_log : float, optional
            Scaling factor applied to log fractional encoder, by default 1.0. Note that this is just a starting value and the model will learn the optimal value.
        dim_feedforward : int, optional
            Dimenions of the feed forward network following transformer, by default 2048
        dropout : float, optional
            Percent dropout in the feed forward network following the transformer, by default 0.1
        how_to_extend : str, optional
            How to extend the feature vector with external features, by default None
        extra_enc_resolution : int, optional
            Number of discretizations for the extra feature encoding, by default 5000
        extra_enc_log_resolution : int, optional
            Number of discretizations for the log extra feature encoding, by default 5000
        extra_scaler : float, optional
            Scaling factor applied to extra feature encoder, by default 1.0. Note that this is just a starting value and the model will learn the optimal value.
        extra_scaler_log : float, optional
            Scaling factor applied to log extra feature encoder, by default 1.0. Note that this is just a starting value and the model will learn the optimal value.
        """

        if how_to_extend not in how_to_extend_ls:
            raise ValueError(f"how_to_extend must be one of {how_to_extend_ls}")

        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.extend_features = extend_features
        self.fractional = fractional
        self.attention = attention
        self.compute_device = compute_device
        self.pe_resolution = pe_resolution
        self.ple_resolution = ple_resolution
        self.elem_prop = elem_prop
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        self.prevalence_encoder = FractionalEncoder(
            self.d_model, resolution=pe_resolution, log10=False
        )
        self.prevalence_log_encoder = FractionalEncoder(
            self.d_model, resolution=ple_resolution, log10=True
        )

        # Initialize the scalers and make them learnable
        self.emb_scaler = nn.parameter.Parameter(torch.tensor([emb_scaler]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([pos_scaler]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([pos_scaler_log]))

        self.num_properties = num_properties
        self.one_hot_layer = one_hot_layer # for multi-task learning


        # Extra Feature Encoding
        if self.extend_features is not None:
            self.how_to_extend = how_to_extend
        else:
            self.how_to_extend = None
        if self.how_to_extend == 'tile_at_input' or self.how_to_extend == 'concat_at_input': 
            # Perform encoding
            self.extra_feat_at_input = True
            self.extra_enc_resolution = extra_enc_resolution
            self.extra_enc_log_resolution = extra_enc_log_resolution
            self.extra_encoder = ExtraEncoder(self.d_model, resolution=extra_enc_resolution, log10=False)
            self.extra_log_encoder = ExtraEncoder(self.d_model, resolution=extra_enc_log_resolution, log10=True)
            self.extra_scaler = nn.parameter.Parameter(torch.tensor([extra_scaler]))
            self.extra_scaler_log = nn.parameter.Parameter(torch.tensor([extra_scaler_log]))
        else:
            self.extra_feat_at_input = False
            self.extra_enc_resolution = None
            self.extra_encoder = None
            self.extra_scaler = None
            self.extra_log_encoder = None
            self.extra_scaler_log = None

        if self.one_hot_layer == 'concat_at_attn' and self.num_properties > 1:
            self.attn_dim = self.d_model + num_properties * self.heads
        else:
            self.attn_dim = self.d_model

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                self.attn_dim,
                nhead=self.heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.N
            )

    def forward(self, src, frac, extra_features=None, property_one_hot=None):
        """Compute the forward pass for encoding the elemental descriptor matrix.

        Parameters
        ----------
        src : torch.tensor
            Tensor containing integers corresponding to elements in compound
        frac : torch.tensor
            Tensor containing the fractions of each element in compound
        extra_features : torch.tensor, optional
            Tensor containing additional features to append after encoding, by default None.

        Returns
        -------
        torch.tensor
            Tensor containing flattened transformer representations of compounds
            concatenated with extended features.
        """

        x = self.embed(src) * self.emb_scaler  # * 2 ** self.emb_scaler

        combined_pe = torch.zeros_like(x)
        pe_scaler = self.pos_scaler
        ple_scaler = self.pos_scaler_log
        extra_scaler = self.extra_scaler
        extra_scaler_log = self.extra_scaler_log

        #* These lines changes the first half of the fractional encoding to use linear fractions and the second half to use logaritmic fractions
        combined_pe[:, :, : self.d_model // 2] = self.prevalence_encoder(frac) * pe_scaler
        combined_pe[:, :, self.d_model // 2 :] = self.prevalence_log_encoder(frac) * ple_scaler  

        #! Complicated, maybe can just use frac itself to build src_mask... (turn non-zero to False, 0 to True)
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1 # same as mask[:, 0, :]

        extra_enc = torch.zeros_like(x)
        extra_log_enc = torch.zeros_like(x)
        if self.how_to_extend == 'tile_at_input':
            extra_tensor = torch.zeros_like(x)
            #* pytorch will automatically broadcast the extra features to the same shape as x
            extra_tensor[:, :, : self.d_model // 2] = self.extra_encoder(extra_features) * extra_scaler
            extra_tensor[:, :, self.d_model // 2 :] = self.extra_log_encoder(extra_features) * extra_scaler_log
        elif self.how_to_extend == 'concat_at_input':
            extra_enc = self.extra_encoder(extra_features) * extra_scaler
            extra_log_enc = self.extra_log_encoder(extra_features) * extra_scaler_log

            extra_tensor = torch.zeros(extra_enc.size(0), extra_enc.size(1), extra_enc.size(2) * 2, device=extra_enc.device, dtype=extra_enc.dtype)
            extra_tensor[:, :, :self.d_model // 2] = extra_enc
            extra_tensor[:, :, self.d_model // 2:] = extra_log_enc

        if self.attention:
            x_src = x + combined_pe
            if self.how_to_extend == 'tile_at_input':
                x_src = x_src + extra_tensor
            elif self.how_to_extend == 'concat_at_input':
                x_src = torch.cat((x_src, extra_tensor), dim=1)
                extra_mask = torch.zeros((extra_tensor.size(0), extra_tensor.size(1)), dtype=torch.bool, device=src_mask.device)
                src_mask = torch.cat((src_mask, extra_mask), dim=1)

            if self.one_hot_layer == 'concat_at_attn':
                # concat one-hot vector to each head of the attention layer
                property_one_hot = property_one_hot.to(x_src.device).unsqueeze(1).expand(-1, x_src.shape[1], -1)
                batch_size, seq_length, x_dim = x_src.shape
                x_src_split = x_src.view(batch_size, seq_length, self.heads, x_dim // self.heads)

                property_one_hot_expanded = property_one_hot.unsqueeze(2).repeat(1, 1, self.heads, 1)

                x_src_combined = torch.cat((x_src_split, property_one_hot_expanded), dim=-1)

                x_src = x_src_combined.view(batch_size, seq_length, -1) 

            x_src = x_src.transpose(0, 1)
            # src_key_padding_mask shape : (batch_size, sequence_length)
            # x_src shape : (sequence_length, batch_size, embed_dim)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)
        else:
            if self.one_hot_layer == 'concat_at_attn':
                property_one_hot = property_one_hot.to(x.device).unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat((x, property_one_hot), dim=-1)

        if self.fractional:
            if self.how_to_extend == 'concat_at_input':
                # Split `x` into its original and concatenated parts
                original_length = x.size(1) - extra_tensor.size(1)

                # Split `x` into `x_original` and `x_extra`
                x_frac = x[:, :original_length, :]
                x_extra = x[:, original_length:, :]

                # Apply the desired operation to `x_original`
                x_frac = x_frac * frac.unsqueeze(2).repeat(1, 1, self.attn_dim)

                # Concatenate `x_original` and `x_extra` back together
                x = torch.cat((x_frac, x_extra), dim=1)

            else:
                x = x * frac.unsqueeze(2).repeat(1, 1, self.attn_dim)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.attn_dim)
        if self.how_to_extend == 'concat_at_input':
            extra_hmask = torch.ones((extra_tensor.size(0), extra_tensor.size(1), self.attn_dim), dtype=torch.bool, device=hmask.device)
            hmask = torch.cat((hmask, extra_hmask), dim=1)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)

        if not self.extra_feat_at_input and self.extend_features is not None:
            if self.how_to_extend == 'concat_at_output':
                # print("Adding extra features at the end of the encoder")
                n_elements = x.shape[1]
                X_extra = extra_features.repeat(1, 1, n_elements).permute([1, 2, 0])
                x = torch.concat((x, X_extra), axis=2)
        return x


# %%
class SubCrab(nn.Module):
    """SubCrab model class which implements the transformer architecture."""

    def __init__(
        self,
        out_dims=3,
        d_model=512,
        extend_features=None,
        d_extend=0,
        N=3,
        heads=4,
        how_to_extend='concat_at_input',
        fractional=True,
        attention=True,
        compute_device=None,
        out_hidden=[1024, 512, 256, 128],
        pe_resolution=5000,
        ple_resolution=5000,
        extra_enc_resolution=5000,
        extra_enc_log_resolution=5000,
        elem_prop="mat2vec",
        num_properties=1,
        one_hot_layer=None,
        bias=False,
        emb_scaler=1.0,
        pos_scaler=1.0,
        pos_scaler_log=1.0,
        extra_scaler=1.0,
        extra_scaler_log=1.0,
        dim_feedforward=2048,
        dropout=0.1
    ):
        """Instantiate a SubCrab class to be used within CrabNet.

        Parameters
        ----------
        out_dims : int, optional
            Output dimensions for Residual Network, by default 3
        d_model : int, optional
            Model size. See paper, by default 512
        extend_features : _type_, optional
            Additional features to grab from columns of the other DataFrames (e.g. state
            variables such as temperature or applied load), by default None
        d_extend : int, optional
            Number of extended features, by default 0
        N : int, optional
            Number of attention layers, by default 3
        heads : int, optional
            Number of attention heads, by default 4
        frac : bool, optional
            Whether to multiply `x` by the fractional amounts for each element, by default True
        attn : bool, optional
            Whether to perform self attention, by default True
        compute_device : _type_, optional
            Computing device to run model on, by default None
        out_hidden : list(int), optional
            Architecture of hidden layers in the Residual Network, by default [1024, 512, 256, 128]
        pe_resolution : int, optional
            Number of discretizations for the prevalence encoding, by default 5000
        ple_resolution : int, optional
            Number of discretizations for the prevalence log encoding, by default 5000
        elem_prop : str, optional
            Which elemental feature vector to use. Possible values are "jarvis", "magpie",
            "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by default "mat2vec"
        bias : bool, optional
            Whether to bias the Residual Network, by default False
        emb_scaler : float, optional
            Float value by which to scale the elemental embeddings, by default 1.0
        pos_scaler : float, optional
            Float value by which to scale the fractional encodings, by default 1.0
        pos_scaler_log : float, optional
            Float value by which to scale the log fractional encodings, by default 1.0
        dim_feedforward : int, optional
            Dimensions of the feed forward network following transformer, by default 2048
        dropout : float, optional
            Percent dropout in the feed forward network following the transformer, by default 0.1
        """

        super().__init__()
        if num_properties > 1:
            if one_hot_layer is None:
                raise ValueError("one hot vector concatenation layer must be specified for multi-task learning")
            if not (one_hot_layer in one_hot_layer_ls or isinstance(one_hot_layer, int)):
                raise ValueError(f"one hot layer must be one of {one_hot_layer_ls} or an integer")
        elif one_hot_layer is not None:
            raise ValueError("one hot vector concatenation layer must not be specified for single-task learning")

        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.extend_features = extend_features
        self.how_to_extend = how_to_extend
        self.d_extend = d_extend
        self.N = N
        self.heads = heads
        self.fractional = fractional
        self.attention = attention
        self.compute_device = compute_device
        self.bias = bias
        self.num_properties = num_properties
        self.one_hot_layer = one_hot_layer
        self.encoder = Encoder(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            extend_features=self.extend_features,
            attention=self.attention,
            compute_device=self.compute_device,
            pe_resolution=pe_resolution,
            ple_resolution=ple_resolution,
            extra_enc_resolution=extra_enc_resolution,
            extra_enc_log_resolution=extra_enc_log_resolution,
            elem_prop=elem_prop,
            emb_scaler=emb_scaler,
            pos_scaler=pos_scaler,
            pos_scaler_log=pos_scaler_log,
            extra_scaler=extra_scaler,
            extra_scaler_log=extra_scaler_log,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            how_to_extend=self.how_to_extend,
            fractional=self.fractional,
            num_properties=self.num_properties,
            one_hot_layer=self.one_hot_layer
        )
        self.out_hidden = out_hidden

        if self.one_hot_layer == 'concat_at_attn':
            self.res_dim = self.d_model + num_properties * self.heads + self.d_extend
        else:
            self.res_dim = self.d_model + self.d_extend

        self.output_nn = ResidualNetwork(
            self.res_dim,
            self.out_dims,
            self.out_hidden,
            self.bias,
            self.num_properties,
            one_hot_layer=self.one_hot_layer
        )
        

    def forward(self, src, frac, extra_features=None, property_one_hot=None):
        """Compute forward pass of the SubCrab model class (i.e. transformer).

        Parameters
        ----------
        src : torch.tensor
            Tensor containing element numbers corresponding to elements in compound
        frac : torch.tensor
            Tensor containing fractional amounts of each element in compound
        extra_features : torch.tensor, optional
            Tensor containing additional features to append after encoding, by default None.
        property_one_hot : torch.tensor, optional
            One-hot encoded vectors to select the property, by default None. Used for multi-task learning.

        Returns
        -------
        torch.tensor
            Model output containing predicted value and uncertainty for that value
        """
        output = self.encoder(src, frac, extra_features, property_one_hot)
        # output = self.transfer_nn(output)

        output = self.output_nn(output, property_one_hot)  # simple linear

        # average the "element contribution", mask so you only average "elements" (i.e.
        # not padded zero values)
        # print(f"Encoder output shape: {output.shape}")
        if self.how_to_extend == 'concat_at_input':
            batch_size, seq_length = src.shape
            num_extra_rows = len(self.extend_features)  # Number of extra rows to add
            extra_tensor = torch.ones((batch_size, num_extra_rows),
                                      dtype=src.dtype, device=src.device)
            # Concatenate `src` with `extra_tensor` along the sequence length dimension (dim=1)
            src = torch.cat((src, extra_tensor), dim=1)


        elem_pad_mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)

        if self.avg:
            output = output.masked_fill(elem_pad_mask, 0)
            output = output.sum(dim=1) / (~elem_pad_mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, : logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output


# %%
if __name__ == "__main__":
    model = SubCrab()
