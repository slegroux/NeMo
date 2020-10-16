# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

from omegaconf import OmegaConf

from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.models.model import Model
from nemo.collections.cv.modules import LeNet5 as LeNet5Module
from nemo.core.neural_types import NeuralType


class LeNet5(Model):
    """
    The LeNet-5 model.
    """

    def __init__(self):
        super().__init__(cfg=OmegaConf.create())

        # Initialize modules.
        self.module = LeNet5Module()
        self.loss = NLLLoss()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`LeNet5Module` input types.
        """
        return self.module.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`LeNet5Module` output types.
        """
        return self.module.output_types

    @typecheck()
    def forward(self, images):
        """ Propagates data by calling the module :class:`LeNet5Module` forward, calculates and returns loss. """
        return self.module.forward(images=images)

    def training_step(self, batch, what_is_this_input):
        """ Training step, calculate loss. """
        # "Unpack" the batch.
        images, targets = batch

        # Get predictions.
        predictions = self(images=images)

        # Calculate loss.
        loss = self.loss(predictions=predictions, targets=targets)

        # Return it.
        return {"loss": loss}
