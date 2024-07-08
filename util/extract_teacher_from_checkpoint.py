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

import argparse
import os
import torch

from collections import OrderedDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Extract the teacher encoder from a MOCA model', add_help=False)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()
    args.use_teacher = True


    print(f"Read checkpoint from {args.checkpoint}")
    assert os.path.isfile(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    pretraining_args = checkpoint['args']
    pretraining_epoch = checkpoint['epoch']
    print(f"Model with args={pretraining_args} from epoch {pretraining_epoch}")

    checkpoint_model = OrderedDict()
    prefix = 'encoder_teacher.' if args.use_teacher else 'encoder.'
    # Keep only the parameters/buffers of the ViT teacher encoder.
    all_keys = list(checkpoint['model'].keys())
    counter = 0
    for key in all_keys:
        if key.startswith(prefix):
            counter += 1
            new_key = key[len(prefix):]
            print(f"\t #{counter}: {key} ==> {new_key}")
            checkpoint_model[new_key] = checkpoint['model'][key]

    print(f"#model keys={counter} / {len(checkpoint_model.keys())}")
    assert args.dst is not None
    assert not os.path.isfile(args.dst)
    torch.save({'model': checkpoint_model}, args.dst)
