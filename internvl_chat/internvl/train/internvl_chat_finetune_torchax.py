# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import math
import os
import random
import sys
import time
import traceback
import warnings
import jax
jax.config.update("jax_default_matmul_precision", "float32")
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional

import numpy as np

try:
    import orjson as json
except:
    import json

from datetime import datetime
import torch
import transformers
from google.cloud import aiplatform
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel)
from internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    LOC_START_TOKEN, LOC_END_TOKEN,
    FRONT_VIEW_TOKEN, FRONT_LEFT_VIEW_TOKEN, FRONT_RIGHT_VIEW_TOKEN,
    BACK_LEFT_VIEW_TOKEN, BACK_RIGHT_VIEW_TOKEN, BACK_VIEW_TOKEN)
from internvl.train.dataset import (
    ConcatDataset,
    TCSLoader,
    WeightedConcatDataset,
    build_transform,
    check_conversations_repetition,
    dynamic_preprocess,
    preprocess,
    preprocess_internlm,
    preprocess_internvl2_5, preprocess_mpt,read_frames_decord,read_frames_gif,
    preprocess_phi3)
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed)
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity)

# torchax imports
import torchax
import jax
import jax.numpy as jnp
import optax
from torchax.interop import JittableModule
from torchax import tensor as torchax_tensor
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'} 
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'} 
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'} 
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'} 
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the LM head. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    tensorboard_id: Optional[str] = field(
        default=None,
        metadata={'help': 'Vertex AI TensorBoard ID.'}
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        if video_path.endswith('.gif'):
            image_list = read_frames_gif(
            video_path,
            num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method)
        else:
            image_list = read_frames_decord(
            video_path,
            num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))
        
        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] ' 
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    # For torchax single device, rank is 0, world_size is 1
    data_rank = 0
    data_world_size = 1
    
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def main():
    # Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Ensure output_dir is absolute path for orbax checkpointing, but handle GCS paths
    training_args.output_dir = training_args.output_dir.strip().strip('"').strip("'").strip()
    if not training_args.output_dir.startswith('gs://'):
        training_args.output_dir = os.path.abspath(training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    logger.info(f"DEBUG: output_dir: '{training_args.output_dir}'")
    logger.info(f'Training/evaluation parameters {training_args}')

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN, LOC_START_TOKEN, LOC_END_TOKEN,
                  FRONT_VIEW_TOKEN, FRONT_LEFT_VIEW_TOKEN, FRONT_RIGHT_VIEW_TOKEN,
                  BACK_LEFT_VIEW_TOKEN, BACK_RIGHT_VIEW_TOKEN, BACK_VIEW_TOKEN]
    
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        # Force standard attention for TPU
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'eager' 
        else:
            config.llm_config._attn_implementation = 'eager'
        
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            config=config)
    else:
        # ... (omitted for brevity, similar changes needed if this path is taken)
        raise NotImplementedError("Only loading from model_name_or_path is supported for this migration script for now.")

    # Enable torchax globally AFTER loading the model to avoid safetensors/storage issues
    torchax.enable_globally()
    
    # VERIFY: Print available JAX devices
    logger.info(f"JAX Devices: {jax.devices()}")

    model.img_context_token_id = img_context_token_id

    # Resize embeddings if needed
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    # Gradient checkpointing might need adjustment for torchax, but let's keep it for now
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
        if hasattr(model.vision_model, 'encoder'):
            model.vision_model.encoder.gradient_checkpointing = True

    # Freezing logic
    if model_args.freeze_backbone:
        logger.info("Freezing ViT backbone...")
        for param in model.vision_model.parameters():
            param.requires_grad = False
    if model_args.freeze_llm:
        logger.info("Freezing LLM...")
        for param in model.language_model.parameters():
            param.requires_grad = False
    if model_args.freeze_mlp:
        logger.info("Freezing MLP...")
        for param in model.mlp1.parameters():
            param.requires_grad = False
    if model_args.unfreeze_vit_layers > 0:
        logger.info(f"Unfreezing last {model_args.unfreeze_vit_layers} ViT layers...")
        # InternVisionModel has encoder.layers
        layers = model.vision_model.encoder.layers
        for i in range(len(layers) - model_args.unfreeze_vit_layers, len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True
    if model_args.use_backbone_lora:
        # LoRA parameters should already have requires_grad=True from peft
        pass
    if model_args.use_llm_lora:
        # LoRA parameters should already have requires_grad=True from peft
        pass

    # Move model to JAX device
    # logger.info("Moving model to JAX device...")
    # model = model.to('jax')

    # Build dataset
    train_dataset = build_datasets(
        data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type, min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame)

    # Data Collator
    from internvl.patch.pad_data_collator import concat_pad_data_collator
    collator = concat_pad_data_collator

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=False
    )

    # --- JAX Distributed Setup (FSDP) ---
    num_devices = len(jax.devices())
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=('data',))
    
    # Sharding Spec: Shard the first dimension (0) of all arrays if divisible by num_devices
    # This acts as FSDP (sharding parameters) and Data Parallelism (sharding batch)
    def get_sharding(x):
        if hasattr(x, 'shape') and len(x.shape) > 0 and x.shape[0] % num_devices == 0:
            return NamedSharding(mesh, P('data'))
        else:
            return NamedSharding(mesh, P()) # Replicated

    # Helper to put PyTree to sharded device memory
    def to_sharded(pytree):
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, get_sharding(x)), pytree)

    # Optimizer
    learning_rate = training_args.learning_rate
    # AdamW epsilon for BF16 should be larger (standard 1e-8 is too small for bf16)
    adam_eps = 1e-5 if training_args.bf16 else 1e-8
    
    # Add gradient clipping and force FP32 optimizer state
    # Note: optax.adamw might not support mu_dtype in older versions, so we check/fallback or just pass it if we are sure.
    # But to be safe and precise, we use the explicit chain which is standard for mixed precision in JAX.
    # Learning rate schedule with warmup
    total_steps = training_args.max_steps if training_args.max_steps > 0 else (len(train_dataloader) * training_args.num_train_epochs)
    warmup_steps = int(total_steps * training_args.warmup_ratio)
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), # Standard clipping for InternVL
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=adam_eps, mu_dtype=jnp.float32), # Force FP32 state
        optax.add_decayed_weights(training_args.weight_decay),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0) # Important: Optax chains need negative scale for minimization
    )
    
    # Get initial params as torchax tensors (on CPU)
    trainable_params = {}
    static_params = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param.detach()
        else:
            static_params[name] = param.detach()
    
    for name, buffer in model.named_buffers():
        static_params[name] = buffer.detach()
    
    # Convert to JAX params (on CPU)
    def torch_to_jax(tensor):
        is_bf16 = tensor.dtype == torch.bfloat16
        if is_bf16:
            tensor = tensor.float()
        arr = jnp.array(tensor.detach().cpu().numpy())
        if is_bf16:
            arr = arr.astype(jnp.bfloat16)
        return arr

    jax_trainable_params_cpu = jax.tree_util.tree_map(torch_to_jax, trainable_params)
    jax_static_params_cpu = jax.tree_util.tree_map(torch_to_jax, static_params)
    
    # Move params to TPU with sharding
    logger.info("Sharding parameters to TPU...")
    jax_trainable_params = to_sharded(jax_trainable_params_cpu)
    jax_static_params = to_sharded(jax_static_params_cpu)
    
    # Initialize optimizer state on TPU (sharded)
    # We need to define out_shardings for opt_state. 
    # Since opt_state structure matches params, we can use jax.eval_shape to infer structure and apply sharding.
    # Use FP32 params for initialization to ensure opt_state (mu, nu) is FP32, matching update path
    jax_trainable_params_fp32_init = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), jax_trainable_params)
    opt_state_shape = jax.eval_shape(optimizer.init, jax_trainable_params_fp32_init)
    opt_state_sharding = jax.tree_util.tree_map(get_sharding, opt_state_shape)
    
    @partial(jax.jit, out_shardings=opt_state_sharding)
    def init_opt_state(p):
        return optimizer.init(p)
        
    logger.info("Initializing optimizer state on TPU...")
    opt_state = init_opt_state(jax_trainable_params_fp32_init)

    # Vertex AI TensorBoard
    tb_run = None
    if data_args.tensorboard_id:
        logger.info(f"Initializing Vertex AI TensorBoard with ID: {data_args.tensorboard_id}")
        try:
            aiplatform.init(project='grhuang-02', location='us-east5', experiment='internvl-tpu', experiment_tensorboard=data_args.tensorboard_id)
            aiplatform.start_run(f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            tb_run = True
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI TensorBoard: {e}")
            tb_run = None

    # Training Step Function
    # in_shardings: params (sharded), batch (sharded), opt_state (sharded)
    # out_shardings: params (sharded), opt_state (sharded), loss (replicated)
    
    trainable_params_sharding = jax.tree_util.tree_map(get_sharding, jax_trainable_params)
    
    def loss_fn(t_params, s_params, batch):
        # Need to wrap back to torchax tensors for functional_call
        # We are inside JIT, so we use the provided JAX arrays
        
        # Cast JAX arrays to torchax tensors
        # Since we are in JAX JIT, we need to be careful. 
        # Torchax allows wrapping JAX arrays.
        
        # We need to reconstruct the full params dictionary for functional_call
        # t_params and s_params are JAX arrays
        
        # Optimization: keep env consistent
        env = torchax.default_env()
        
        # Reconstruct full params
        # Note: t_params and s_params are already JAX arrays
        
        # For torchax.tensor.Tensor, we need to provide the JAX array
        
        # To avoid overhead, we can try to use jax.tree_map to wrap all at once
        # But functional_call expects a dict of tensors.
        
        # Prepare parameters for functional_call
        # We need to merge t_params and s_params
        # Both are dicts of JAX arrays
        
        # Create a unified dict of JAX arrays
        # We can't easily merge dicts in JAX JIT without knowing keys, 
        # but here we are in Python during JIT compilation, so it's fine.
        
        # Actually, loss_fn is called by jax.value_and_grad, which is JIT compiled.
        # The arguments t_params, s_params are JAX arrays.
        
        # We need to know the keys to reconstruct the dictionary structure if needed, 
        # but here t_params and s_params are already dictionaries of JAX arrays.
        
        t_params_fp32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), t_params)
        s_params_fp32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), s_params)
        batch_fp32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, batch)
        
        full_params = {}
        full_params.update(t_params_fp32)
        full_params.update(s_params_fp32)
        
        torch_params = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), full_params)
        torch_batch = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), batch_fp32)
        outputs = torch.func.functional_call(model, torch_params, args=(), kwargs=torch_batch)
        
        loss = outputs.loss.jax()
        # Mean Loss: sum of masked loss / sum of mask
        loss_mask = (batch_fp32['labels'] != -100).astype(jnp.float32)
        num_valid_tokens = jnp.sum(loss_mask) + 1e-8
        
        if loss.ndim == 0:
            # If loss is already a scalar, it is likely already averaged by the model (standard for InternVL/Transformers)
            # We do NOT divide again.
            pass
        else:
            # If loss is per-token, we take the mean over valid tokens.
            loss = jnp.sum(loss * loss_mask) / num_valid_tokens
        return loss

    @partial(jax.jit, out_shardings=(trainable_params_sharding, jax.sharding.NamedSharding(mesh, P())))
    def compute_grad_jit(trainable_params, static_params, batch):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(trainable_params, static_params, batch)
        return grads, loss

    @partial(jax.jit, out_shardings=(trainable_params_sharding, opt_state_sharding))
    def apply_updates_jit(grads, opt_state, trainable_params):
        # Check for NaN in gradients
        all_grads_flat = jax.tree_util.tree_leaves(grads)
        any_nan_grad = jnp.any(jnp.array([jnp.isnan(g).any() for g in all_grads_flat]))
        
        def skip_update_fn(grads, opt_state, params):
            # To print parameter names, we need to iterate over the pytree.
            # Since this is inside JIT, we use jax.debug.callback for each parameter.
            
            # We need the paths. jax.tree_util.tree_flatten_with_path gives us (path, value)
            flat_grads_with_path, _ = jax.tree_util.tree_flatten_with_path(grads)
            for path, grad in flat_grads_with_path:
                # Path is a tuple of DictKey, IndexKey, etc. Convert to string.
                path_str = "".join(str(p) for p in path)
                has_nan = jnp.isnan(grad).any()
                jax.debug.callback(lambda h, p: h and print(f"DEBUG: Gradient for {p} has NaN"), has_nan, path_str)
            
            return params, opt_state

        def apply_update_fn(grads, opt_state, params):
            # Cast to FP32 for stable update
            grads_fp32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), grads)
            params_fp32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
            
            updates, opt_state = optimizer.update(grads_fp32, opt_state, params_fp32)
            new_params_fp32 = optax.apply_updates(params_fp32, updates)
            
            # Cast back to BF16 for storage
            new_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), new_params_fp32)
            return new_params, opt_state

        new_trainable_params, opt_state = jax.lax.cond(
            any_nan_grad,
            lambda g, os, p: skip_update_fn(g, os, p),
            lambda g, os, p: apply_update_fn(g, os, p),
            grads, opt_state, trainable_params
        )
        return new_trainable_params, opt_state

    # Training Loop
    logger.info("Starting training loop...")
    model.train()
    
    global_step = 0
    epoch = 0  # Initialize epoch for use in finally block
    accumulated_grads = None
    accumulated_loss = 0.0
    micro_step = 0
    try:
        for epoch in range(int(training_args.num_train_epochs)):
            for batch in train_dataloader:
                if micro_step % training_args.gradient_accumulation_steps == 0:
                    step_start_time = time.time()
                
                # Preprocess batch: cast to bf16 and shard
                new_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        if torch.isnan(v).any():
                            print(f"DEBUG: CPU input {k} has NaN!")
                        
                        # Pad to max_seq_length if needed to avoid JAX recompilation
                        if k in ['input_ids', 'labels', 'attention_mask', 'position_ids'] and len(v.shape) == 2 and v.shape[1] < data_args.max_seq_length:
                            pad_len = data_args.max_seq_length - v.shape[1]
                            pad_val = 0
                            if k == 'labels':
                                pad_val = -100 # Standard ignore index for labels
                            elif k == 'input_ids':
                                pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                            
                            # Pad last dimension (dim 1)
                            v = torch.nn.functional.pad(v, (0, pad_len), value=pad_val)
                        
                        if training_args.bf16 and v.dtype == torch.float32:
                            v = v.to(torch.bfloat16)
                        # Convert to JAX array (on CPU first)
                        jax_arr = v.to('jax').jax()
                        # Shard to TPU
                        new_batch[k] = jax.device_put(jax_arr, get_sharding(jax_arr))
                batch = new_batch
                
                # Perform training step: compute gradient
                grads, loss = compute_grad_jit(jax_trainable_params, jax_static_params, batch)
                
                # Accumulate gradients and loss
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jax.tree_util.tree_map(lambda x, y: x + y, accumulated_grads, grads)
                
                accumulated_loss += loss
                micro_step += 1
                
                # Update parameters if accumulation steps reached
                if micro_step % training_args.gradient_accumulation_steps == 0:
                    # Average gradients and loss over accumulation steps
                    accumulated_grads = jax.tree_util.tree_map(lambda x: x / training_args.gradient_accumulation_steps, accumulated_grads)
                    current_loss = accumulated_loss / training_args.gradient_accumulation_steps
                    
                    # Apply updates
                    jax_trainable_params, opt_state = apply_updates_jit(accumulated_grads, opt_state, jax_trainable_params)
                    
                    # Block until loss is computed to measure actual execution time (JAX is async)
                    current_loss.block_until_ready()
                    
                    step_end_time = time.time()
                    step_duration = step_end_time - step_start_time
                    
                    if global_step % training_args.logging_steps == 0:
                        logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {current_loss}, Time: {step_duration:.4f}s")
                        if tb_run:
                            aiplatform.log_time_series_metrics({
                                'train/loss': float(current_loss),
                                'train/lr': float(schedule(global_step)),
                                'train/step_time': step_duration
                            }, step=global_step)
                    
                    global_step += 1
                    
                    if training_args.save_steps > 0 and global_step % training_args.save_steps == 0:
                        save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}.pt")
                        
                        # Gather params to CPU for saving (to avoid OOM on single device gather)
                        trainable_params_cpu = jax.device_get(jax_trainable_params)
                        static_params_cpu = jax.device_get(jax_static_params)
                        
                        # Wrap back to torchax tensors
                        env = torchax.default_env()
                        current_torch_params = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), trainable_params_cpu)
                        current_torch_static_params = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), static_params_cpu)
                        
                        state = {
                            'params': current_torch_params,
                            'static_params': current_torch_static_params,
                            'epoch': epoch,
                            'global_step': global_step
                        }
                        torchax.save_checkpoint(state, save_path, step=global_step)
                        logger.info(f"Saved checkpoint to {save_path}")
                    
                    # Reset accumulation
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    step_start_time = time.time() # Reset timer for next step
                
                if training_args.max_steps > 0 and global_step >= training_args.max_steps:
                    logger.info(f"Reached max_steps {training_args.max_steps}. Stopping training.")
                    break
            
            if training_args.max_steps > 0 and global_step >= training_args.max_steps:
                break
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise
    finally:
        # Save final model
        final_save_path = os.path.join(training_args.output_dir, "final_checkpoint.pt")
        
        trainable_params_cpu = jax.device_get(jax_trainable_params)
        static_params_cpu = jax.device_get(jax_static_params)
        env = torchax.default_env()
        final_torch_params = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), trainable_params_cpu)
        final_torch_static_params = jax.tree_util.tree_map(lambda x: torchax_tensor.Tensor(x, env), static_params_cpu)
        
        state = {
            'params': final_torch_params,
            'static_params': final_torch_static_params,
            'epoch': epoch,
            'global_step': global_step
        }
        torchax.save_checkpoint(state, final_save_path, step=global_step)
        logger.info(f"Saved final checkpoint to {final_save_path}")

        if tb_run:
            try:
                aiplatform.end_run()
            except Exception as e:
                logger.warning(f"Failed to end Vertex AI TensorBoard run: {e}")

if __name__ == '__main__':
    main()