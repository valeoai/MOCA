# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.misc as misc
import random

import timm
from timm.utils import accuracy
from timm.models.vision_transformer import PatchEmbed, Block, Mlp
Block = partial(Block, qk_scale=None) if (timm.version == "0.3.2") else Block

from util.model_utils import get_2d_sincos_pos_embed

NORMALIZE_EPS = 1e-5

class L2Normalize(nn.Module):
    def __init__(self, dim):
        super(L2Normalize, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=NORMALIZE_EPS)


class BoWExtractor(nn.Module):
    def __init__(
        self,
        num_words,
        num_channels,
        inv_delta=15,
        num_new_words=16,
        skip_offset=2,
        update_type="random_token"):
        super(BoWExtractor, self).__init__()

        assert isinstance(inv_delta, (float, int))
        self.inv_delta = inv_delta
        self.Knew = num_new_words
        self.skip_offset = skip_offset
        self.decay = 0.99
        self.update_type = update_type
        assert update_type in ("random_token", "avg_token")

        embedding = torch.randn(num_words, num_channels).clamp(min=0)
        embedding = F.normalize(embedding, p=2, dim=1, eps=NORMALIZE_EPS) # L2-normalization
        self.register_buffer("_embedding", embedding)
        self.register_buffer("_embedding_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_track_num_batches", torch.zeros(1))
        self.register_buffer("_dist_norm",  torch.ones(1) * 0.5)
        self.register_buffer("_dist_norm_prev",  torch.ones(1) * 0.5)
        self._dist_norm_prev.data.copy_(self._dist_norm.data)

        self._ave_min_dist = 100

    @torch.no_grad()
    def update(self, x, attn=None):
        """Given a feature map x, it updates the queue-based vocabulary."""
        assert self.training
        N, L, C = x.size()
        height = width = int(math.sqrt(L-1))
        offset = int(self.skip_offset)
        Knew = self.Knew // misc.get_world_size()
        assert Knew <= N
        # Randomly select Knew images from the mini-batch.
        batch_idx = torch.randperm(N)[:Knew].long().cuda()

        if self.update_type == "random_token":
            x = x.view(N * L, C)
            # Randomly select a patch token from each image; skip the [CLS] tokens by adding 1.
            keep = torch.arange(height*width, device=x.device).view(height, width)
            keep = 1 + keep[offset:-offset, offset:-offset].contiguous().view(-1)
            token_idx = keep[torch.randint(0, keep.shape[0], (Knew,), device=x.device)]
            new_words = x[batch_idx * L + token_idx] # shape [Knew, C]
        elif self.update_type == "avg_token":
            new_words = x[batch_idx]
            keep = torch.arange(height*width, device=x.device).view(height, width)
            keep = 1 + keep[offset:-offset, offset:-offset].contiguous().view(-1) # adds 1 for the cls token.
            new_words = new_words[:, keep, :].mean(dim=1) # shape [Knew, (height-offset*2)*(height-width*2), C] ==> [Knew, C]

        new_words = F.normalize(new_words, p=2, dim=1, eps=NORMALIZE_EPS) # L2-normalization.

        # To simplify the queue update implementation, it is assumed that the
        # number of words K is a multiple of Knew.
        assert self._embedding.shape[0] % new_words.shape[0] == 0
        # Replace the oldest visual word embeddings with the selected ones
        # using the self._embedding_ptr pointer. Note that each training step
        # self._embedding_ptr points to the older visual words.
        ptr = int(self._embedding_ptr)
        self._embedding[ptr:(ptr + new_words.shape[0]),:] = new_words
        # move the pointer.
        self._embedding_ptr[0] = (ptr + new_words.shape[0]) % self._embedding.shape[0]

        self._dist_norm_prev.data.copy_(self._dist_norm.data)
        self._track_num_batches += 1

    @torch.no_grad()
    def get_dictionary(self):
        """Returns the visual word embeddings of the dictionary/vocabulary."""
        return self._embedding.detach().clone()

    def compute_bow(self, codes, height, width):
        # shape of codes: [N, L, K]
        # Reduce assignment codes to bag-of-word vectors with global pooling.
        assert codes.shape[1] == (height * width)
        if self.skip_offset > 0:
            offset = int(self.skip_offset)
            keep = torch.arange(height*width, device=codes.device).view(height, width)
            keep = keep[offset:-offset, offset:-offset].contiguous().view(-1)
            codes = codes[:, keep, :]
        bow = codes.mean(dim=1)
        # shape of bow: [N, K]
        bow = F.normalize(bow, p=1, dim=1, eps=NORMALIZE_EPS) # L1-normalization.
        return bow

    def assign_words(self, x):
        # shape of x [N, L, C]
        words = self._embedding # shape [K, C]
        x = F.normalize(x, p=2, dim=2, eps=NORMALIZE_EPS) # L2-normalization.
        dist = -torch.nn.functional.linear(x, weight=words, bias=None)

        dist = dist.float()
        # dist shape: [N, L, K]
        min_dist, enc_indices = torch.min(dist, dim=2) # shapes [N, L]
        if self.training:
            # exponential moving average update of self._dist_norm.
            self._ave_min_dist = min_dist.mean().item()
            dist_norm_tmp = (torch.mean(dist, dim=2) - min_dist).mean()
            dist_norm_tmp = dist_norm_tmp.abs()
            dist_norm_tmp = misc.reduce_all(dist_norm_tmp) / misc.get_world_size() # Possibly this communication is not needed. To be tested?
            self._dist_norm.data.mul_(self.decay).add_(dist_norm_tmp, alpha=(1. - self.decay))

        # Soft assignment codes.
        inv_delta = self.inv_delta / self._dist_norm_prev
        codes = F.softmax(-inv_delta * dist, dim=2)
        return codes

    def forward(self, x):
        """
        Input:
            x: 3D tensor with shape [N, L, C], where N is the batch size,
            L is the number of tokens, and C is the number of dimensions.
        """
        x = x[:, 1:, :] # remove the [CLS] token.
        height = width = int(math.sqrt(x.shape[1]))
        codes = self.assign_words(x)
        # Reduce assignment codes to bag-of-word vectors with global pooling.
        bow = self.compute_bow(codes, height, width)
        return bow, codes

    def extra_repr(self):
        str_options = (
            f"embedding.shape={list(self._embedding.data.shape)}, "
            f"inv_delta={self.inv_delta}, "
            f"diff_norm={self._dist_norm.data.item()}, "
            f"Knew={self.Knew}, "
            f"update_type={self.update_type}"
            f"track_num_batches={self._track_num_batches.item()}")
        return str_options


class BoWExtractorMultipleLevels(nn.Module):
    def __init__(self, opts_list, bow_fn=BoWExtractor):
        """Builds a BoW extractor for each BoW level."""
        super(BoWExtractorMultipleLevels, self).__init__()
        assert isinstance(opts_list, (list, tuple))
        self.bow_extractor = nn.ModuleList([bow_fn(**opts) for opts in opts_list])

    @torch.no_grad()
    def get_dictionary(self):
        """Returns the dictionary of visual words from each BoW level."""
        return [b.get_dictionary() for b in self.bow_extractor]

    def forward(self, features):
        """Given a list of feature levels, it generates multi-level BoWs."""
        assert isinstance(features, (list, tuple))
        assert len(features) == len(self.bow_extractor)
        out = list(zip(*[b(f) for b, f in zip(self.bow_extractor, features)]))
        return out

    def update(self, features, attn=None):
        assert isinstance(features, (list, tuple))
        assert len(features) == len(self.bow_extractor)
        for b, f in zip(self.bow_extractor, features):
            b.update(f, attn)


class ResWGEN(nn.Module):
    def __init__(self, generator, num_channels_in, num_channels_out):
        super(ResWGEN, self).__init__()
        self.l2norm = L2Normalize(dim=1)
        self.generator = generator
        if num_channels_in == num_channels_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(num_channels_in, num_channels_out)

    def forward(self, dictionary):
        x = self.l2norm(dictionary)
        x_res = self.generator(x)
        x_skip = self.skip(x)
        return self.l2norm(x_res + x_skip)


class BoWPredictor(nn.Module):
    def __init__(
        self,
        num_channels_out=384,
        num_channels_in=[384,],
        num_channels_hidden=1024,
        kappa=8,
        learn_kappa=False,
        num_layers=2,
        residual=True,
    ):
        """ Builds the dynamic BoW prediction head of the student network.

        It essentially builds a weight generation module for each BoW level for
        which the student network needs to predict BoW. For example, in its
        full version, OBoW uses two BoW levels, one for conv4 of ResNet (i.e.,
        penultimate feature scale of ResNet) and one for conv5 of ResNet (i.e.,
        final feature scale of ResNet). Therefore, in this case, the dynamic
        BoW prediction head has two weight generation modules.

        Args:
        num_channels_in: a list with the number of input feature channels for
            each weight generation module. For example, if OBoW uses two BoW
            levels and a ResNet50 backbone, then num_channels_in should be
            [1024, 2048], where the first number is the number of channels of
            the conv4 level of ResNet50 and the second number is the number of
            channels of the conv5 level of ResNet50.
        num_channels_out: the number of output feature channels for the weight
            generation modules.
        num_channels_hidden: the number of feature channels at the hidden
            layers of the weight generator modules.
        kappa: scalar with scale coefficient for the output weight vectors that
            the weight generation modules produce.
        learn_kappa (default False): if True kappa is a learnable parameter.
        num_layers: num_layers for the weight generation module.
        """
        super(BoWPredictor, self).__init__()
        assert num_layers >= 1
        assert isinstance(num_channels_in, (list, tuple))
        num_code_levels = len(num_channels_in)
        assert num_code_levels == 1

        bottleneck_dim = num_channels_out

        generators = nn.Sequential()
        if residual is False:
            generators.add_module(f"b0_l2norm_in", L2Normalize(dim=1))
        if num_layers == 1:
            num_channels_last = num_channels_in[0]
        else:
            generators.add_module(f"b0_fc", nn.Linear(num_channels_in[0], num_channels_hidden, bias=False))
            generators.add_module(f"b0_bn", nn.BatchNorm1d(num_channels_hidden))
            generators.add_module(f"b0_rl", nn.ReLU(inplace=False))
            for layer in range(2, num_layers):
                generators.add_module(f"b0_fc{layer}", nn.Linear(num_channels_hidden, num_channels_hidden, bias=False))
                generators.add_module(f"b0_bn{layer}", nn.BatchNorm1d(num_channels_hidden))
                generators.add_module(f"b0_rl{layer}", nn.ReLU(inplace=False))

            num_channels_last = num_channels_hidden
        generators.add_module(f"b0_last_layer", nn.Linear(num_channels_last, bottleneck_dim))
        if residual is False:
            generators.add_module(f"b0_l2norm_out", L2Normalize(dim=1))
        else:
            generators = ResWGEN(generators, num_channels_in[0], bottleneck_dim)

        self.layers_w = nn.ModuleList([generators,])

        self.scale = nn.Parameter(
            torch.FloatTensor(num_code_levels).fill_(kappa),
            requires_grad=learn_kappa)

    def forward(self, features, dictionary):
        """Dynamically predicts the BoW from the features of cropped images.

        During the forward pass, it gets as input a list with the features from
        each type of extracted image crop and a list with the visual word
        dictionaries of each BoW level. First, it uses the weight generation
        modules for producing from each dictionary level the weight vectors
        that would be used for the BoW prediction. Then, it applies the
        produced weight vectors of each dictionary level to the given features
        to compute the BoW prediction logits.

        Args:
        features: list of 2D tensors where each of them is a mini-batch of
            features (extracted from the image crops) with shape
            [(batch_size * num_crops) x num_channels_out] from which the BoW
            prediction head predicts the BoW targets. For example, in the full
            version of OBoW, in which it reconstructs BoW from (a) 2 image crops
            of size [160 x 160] and (b) 5 image patches of size [96 x 96], the
            features argument includes a 2D tensor of shape
            [(batch_size * 2) x num_channels_out] (extracted from the 2
            160x160-sized crops) and a 2D tensor of shape
            [(batch_size * 5) x num_channels_out] (extractted from the 5
            96x96-sized crops).
        dictionary: list of 2D tensors with the visual word embeddings
            (i.e., dictionaries) for each BoW level. So, the i-th item of
            dictionary has shape [num_words x num_channels_in[i]], where
            num_channels_in[i] is the number of channels of the visual word
            embeddings at the i-th BoW level.

        Output:
        logits_list: list of lists of 2D tensors. Specifically, logits_list[i][j]
            contains the 2D tensor of size [(batch_size * num_crops) x num_words]
            with the BoW predictions from features[i] for the j-th BoW level
            (made using the dictionary[j]).
        """
        assert isinstance(dictionary, (list, tuple))
        assert len(dictionary) == len(self.layers_w)

        weight = [gen(dict).t() for gen, dict in zip(self.layers_w, dictionary)]
        kappa = torch.split(self.scale, 1, dim=0)

        if isinstance(features, torch.Tensor):
            preds = [torch.mm(features * k, w) for k, w in zip(kappa, weight)]
        else:
            preds = [[torch.mm(f * k, w) for k, w in zip(kappa, weight)] for f in features]

        return preds

    def extra_repr(self):
        kappa = self.scale.data
        s = f"(kappa, learnable={kappa.requires_grad}): {kappa.tolist()}"
        return s


class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024,
                 depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def get_features(self, x, layers):
        num_layers = len(self.blocks)
        layers = sorted([num_layers-i-1 for i in layers])

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i + 1) == num_layers: # last layer.
                x = self.norm(x)
            if i in layers:
                output.append(x)
        return output

    def forward(self, x, layers=[0,], mask_ids_shuffle=None, mask_len_keep=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if mask_ids_shuffle is not None:
            assert mask_len_keep is not None
            x = misc.mask_input(x, mask_ids_shuffle, mask_len_keep)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        return self.get_features(x, layers=layers)

    def forward_features(self, x, layers=[0,]):
        return self.forward(x, layers=layers, mask_ids_shuffle=None, mask_len_keep=None)
    
    def forward_last_features(self, x):
        return self.forward_features(x, layers=[0,])[0]

    def forward_masked(self, x_in, mask_ratio_list, intermediate_layer, use_complementary, dec_mask_ratio_list, avg_pooling=True):
        output = {"x_last": [], "x_cls": [], "x_intermediate": []}
        # Prepare masks for each masking round
        N, L = x_in.shape[0], self.patch_embed.num_patches # batch size, sequence length without the [CLS] token
        if use_complementary and len(mask_ratio_list) == 2:
            # Here the 2nd masking round is "complementary" to the 1st one. 
            # This means that the visible to the encoder tokens of the 2nd masking round would be different (not overlap) 
            # with the visible tokens of the 1st round, while the tokens that the decoder must reconstruct would be the same.
            ids_shuffle, len_keep = misc.make_two_complementary_masks(N, L, mask_ratio_list, dec_mask_ratio_list, x_in.device)
        else:
            ids_shuffle, len_keep = misc.create_random_mask_mult_rounds(N, L, mask_ratio_list, x_in.device)
        
        layers = [intermediate_layer, 0]
        for i in range(len(mask_ratio_list)):
            # i-th masking round of x_in
            x_out = self.forward(x_in, layers=layers, mask_ids_shuffle=ids_shuffle[i], mask_len_keep=len_keep[i])
            assert len(x_out) == (1 + int(intermediate_layer > 0))
            
            x_last = x_out[-1]
            # Compute the image-wise token embedding ([AVG] or [CLS] token)
            x_cls = x_last[:, 1:, :].mean(dim=1) if avg_pooling else x_last[:, 0, :].contiguous()

            output["x_last"].append(x_last)
            output["x_cls"].append(x_cls)
            output["x_intermediate"].append(x_out[0])
        
        output["ids_shuffle"] = ids_shuffle
        output["len_keep"] = len_keep

        return output        

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)


class MOCA(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, depth=12, num_heads=6,
                 decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 inv_delta=10.0, num_words=4096, num_new_words=16,
                 skip_offset=2, kappa=5.0, pred_mlp_ratio=2, early_layer=4, use_loc_loss=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # Encoder specifics
        self.encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        
        # Predictor for the image-wise masked cross-view code assignement loss
        code_predictor_opts = {
            "kappa": kappa,
            "num_channels_out": embed_dim,
            "num_channels_hidden": int(embed_dim * pred_mlp_ratio),
            "num_channels_in": [embed_dim,],
            "residual": True}
        self.encoder_pred = BoWPredictor(**code_predictor_opts)

        self.encoder_teacher = ViTEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        code_extractor_opts_list = [{
            "num_channels": embed_dim,
            "inv_delta": inv_delta,
            "num_words": num_words,
            "num_new_words": num_new_words,
            "skip_offset": skip_offset}]
        # Code assignement generation from the teacher encoder.
        self.bow_extractor = BoWExtractorMultipleLevels(code_extractor_opts_list)
        # --------------------------------------------------------------------------

        # ----------------------------------------------------------------------
        
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Decoder specifics
        num_patches = self.encoder.patch_embed.num_patches
        self.early_layer = early_layer
        self.use_loc_loss = use_loc_loss
        if self.use_loc_loss:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            # Define the modules and params of the decoder. The "cls_pt" suffix if from "legacy" naming.
            self.norm_cls_pt = norm_layer(embed_dim)
            self.decoder_embed_cls_pt = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token_cls_pt = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_blocks_cls_pt = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])
            self.decoder_norm_cls_pt = norm_layer(decoder_embed_dim)

            code_predictor_opts_decoder = copy.deepcopy(code_predictor_opts)
            code_predictor_opts_decoder["kappa"] = kappa
            code_predictor_opts_decoder["num_channels_in"] = [embed_dim for _ in code_extractor_opts_list]
            code_predictor_opts_decoder["num_channels_out"] = decoder_embed_dim
            self.decoder_pred = BoWPredictor(**code_predictor_opts_decoder)
            # --------------------------------------------------------------------------

        self.initialize_weights()

        for param, param_teacher in zip(
            self.encoder.parameters(),
            self.encoder_teacher.parameters()):
            param_teacher.data.copy_(param.data)  # initialize
            param_teacher.requires_grad = False  # not update by gradient

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.use_loc_loss:
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                int(self.encoder.patch_embed.num_patches**.5),
                cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            if self.early_layer is not None:
                torch.nn.init.normal_(self.mask_token_cls_pt, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def update_encoder_teacher(self, momentum):
        """ Exponetial moving average for the feature_extractor_teacher params:
            param_teacher = param_teacher * momentum + param * (1-momentum)
        """
        if not self.training:
            return
        if momentum >= 1.0:
            return
        for param, param_teacher in zip(
            self.encoder.parameters(),
            self.encoder_teacher.parameters()):
            if param.requires_grad:
                param_teacher.data = (
                    param_teacher.data * momentum +
                    param.detach().data * (1.-momentum))

    def extract_targets(self, x1, x2, momentum, update_teacher=True):
        """
        Given as input the images x1 and x2, extract the code assignement targets.
        """
        with torch.no_grad():  # no gradient
            if update_teacher:
                self.update_encoder_teacher(momentum)  # update the momentum encoder
            dictionary = self.bow_extractor.get_dictionary()
            teacher_features1 = self.encoder_teacher.forward_features(x1)
            teacher_features2 = self.encoder_teacher.forward_features(x2)
            # Extract the target code assignments and average code assignments (aka bag-of-words).
            bow_code_x1, codes_x1 = self.bow_extractor(teacher_features1)
            bow_code_x2, codes_x2 = self.bow_extractor(teacher_features2)

        if self.training and update_teacher:
            # Update the teacher's codebook / dictionry.
            if random.random() < 0.5:
                self.bow_extractor.update(teacher_features1)
            else:
                self.bow_extractor.update(teacher_features2)
        
        same_view_codes = torch.cat([codes_x1[0], codes_x2[0]], dim=0)
        cross_view_bows = torch.cat([bow_code_x2[0], bow_code_x1[0]], dim=0)

        return cross_view_bows, same_view_codes, dictionary

    def forward_masked_decoder(self, x_cls, x, ids_shuffle, dec_mask_ratio=0.5):
        # embed tokens
        x = torch.cat([x_cls.unsqueeze(1), self.norm_cls_pt(x[:, 1:, :])], dim=1)

        x = self.decoder_embed_cls_pt(x)
        N, _, D = x.shape
        len_unmask = x.shape[1] - 1 # Number of tokens that were given as input to the encoder (excluding the [CLS] token)

        len_dec_keep = int(ids_shuffle.shape[1] * (1. - dec_mask_ratio)) # Number of tokens that will be given as input to the decoder.
        assert len_dec_keep > len_unmask
        mask_tokens = self.mask_token_cls_pt.repeat(x.shape[0], len_dec_keep + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no [CLS] token

        ids_keep = ids_shuffle[:, :len_dec_keep]
        ids_keep_skip_img = ids_keep + 1
        dec_pos_patch = torch.gather(self.decoder_pos_embed.repeat(N, 1, 1), dim=1, index=ids_keep_skip_img.unsqueeze(-1).repeat(1, 1, D))
        dec_pos_img = self.decoder_pos_embed[:, :1, :]
        x = torch.cat([x[:, :1, :] + dec_pos_img, x_ + dec_pos_patch], dim=1)  # add position embeddings and append [AVG] token (global image embedding)

        # apply Transformer blocks
        for blk in self.decoder_blocks_cls_pt:
            x = blk(x)
        x = self.decoder_norm_cls_pt(x)
        x = x[:, 1: :].contiguous().view(-1, x.shape[2]) # remove the [AVG]/[CLS] token and flatten

        return x, ids_keep

    def forward_img_loss(self, target, pred):
        return F.kl_div(F.log_softmax(pred, dim=1), target, reduction="batchmean")

    def forward_loc_loss(self, target, pred, ids_pred):
        K = target.shape[2]
        target = torch.gather(target, dim=1, index=ids_pred.unsqueeze(-1).repeat(1, 1, K))
        return F.kl_div(F.log_softmax(pred, dim=1), target.view(-1, K), reduction="batchmean")

    def forward(self, x1, x2, momentum, args, update_teacher=True):
        ####################################### TEACHER ##################################################
        cross_view_bows, same_view_codes, dictionary = self.extract_targets(x1, x2, momentum, update_teacher)
        ##################################################################################################

        ################################## STUDENT ENCODER ###############################################
        x = torch.cat([x1, x2], dim=0) # Concat the two views
        enc_out = self.encoder.forward_masked(
            x, args.mask_ratio, intermediate_layer=self.early_layer, use_complementary=True, 
            dec_mask_ratio_list=args.dec_mask_ratio, avg_pooling=args.avg_pooling)
        ##################################################################################################
        
        ######################## MASKED CROSS-VIEW AVERAGE ASSIGNMENT PREDICTIONS ########################
        bow_code_preds = self.encoder_pred(enc_out["x_cls"], dictionary)
        loss_img = [self.forward_img_loss(cross_view_bows, pred[0]) for pred in bow_code_preds]
        loss_img = torch.stack(loss_img).mean() * 2
        loss_tot = loss_img * args.img_weight
        ##################################################################################################

        ################## MASKED SAME-VIEW TOKEN ASSIGNMENT PREDICTIONS #############################
        if self.use_loc_loss:
            # Extract decoder features from all masked views.
            dec_features, ids_pred = list(zip(*[
                self.forward_masked_decoder(enc_out["x_cls"][i], enc_out["x_intermediate"][i], enc_out["ids_shuffle"][i], args.dec_mask_ratio[i])
                for i in range(args.num_mviews)]))
            codes_preds = self.decoder_pred(dec_features, dictionary) # Make decoder predictions
            loss_loc = [self.forward_loc_loss(same_view_codes, pred[0], ids) for (pred, ids) in zip(codes_preds, ids_pred)]
            loss_loc = torch.stack(loss_loc).mean() * 2
        else:
            loss_loc = torch.zeros_like(loss_img)
        loss_tot += (loss_loc * args.loc_weight)
        ##################################################################################################

        stats = {"losses": {"img": loss_img.item(), "loc": loss_loc.item()}}
        return loss_tot, stats


def moca_vit_base_patch16_dec512d4b(**kwargs):
    model = MOCA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def moca_vit_base_patch16_dec512d8b(**kwargs):
    model = MOCA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def moca_vit_base_patch16_dec512d2b(**kwargs):
    model = MOCA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def moca_vit_base_patch16_dec512d1b(**kwargs):
    model = MOCA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def moca_vit_large_patch16_dec512d2b(**kwargs):
    model = MOCA(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended arch
moca_vit_base_patch16_dec8 = moca_vit_base_patch16_dec512d8b # decoder: 512 dim, 8 blocks
moca_vit_base_patch16_dec4 = moca_vit_base_patch16_dec512d4b # decoder: 512 dim, 4 blocks
moca_vit_base_patch16_dec2 = moca_vit_base_patch16_dec512d2b # decoder: 512 dim, 2 blocks
moca_vit_base_patch16_dec1 = moca_vit_base_patch16_dec512d1b # decoder: 512 dim, 1 blocks
moca_vit_large_patch16_dec2 = moca_vit_large_patch16_dec512d2b # decoder: 512 dim, 2 blocks