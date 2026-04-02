# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from inspect import isclass
import torch
import torch.nn as nn
from pyro.distributions.util import broadcast_shape


class ConcatModule(nn.Module):
    """
    A custom module for concatenation of tensors along the last dimension.
    Handles broadcasting automatically if specified.
    """

    def __init__(self, allow_broadcast=False):
        """
        Args:
            allow_broadcast (bool): If True, broadcasts input tensors to a common shape
                                    before concatenation.
        """
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        # Handle case where a single list/tuple is passed instead of unpacked args
        if len(input_args) == 1:
            # We unpack it to treat it as a list of inputs
            input_args = input_args[0]

        # If it's already a single tensor, simply return it
        if torch.is_tensor(input_args):
            return input_args
        else:
            # Concatenate multiple tensors
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)


class ListOutModule(nn.ModuleList):
    """
    A custom module for handling multiple output heads.
    Passes the same input to a list of separate modules and returns a list of their outputs.
    """

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        # Loop over all sub-modules (heads), applying the same input arguments
        return [mm.forward(*args, **kwargs) for mm in self]


def call_nn_op(op):
    """
    Helper function to instantiate standard nn.Module operations
    with appropriate default parameters (e.g., dimension for Softmax).

    Args:
        op: The nn.Module class or instance to wrap.
    """
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()


class MLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron (MLP) builder.

    Features:
    - Dynamic hidden layer depth.
    - Support for concatenated inputs.
    - Custom callbacks for injecting layers (BatchNorm, Dropout) between standard layers.
    - Support for multiple output heads.
    """

    def __init__(
        self,
        name,
        mlp_sizes,
        activation=nn.ReLU,
        output_activation=None,
        post_layer_fct=lambda layer_ix, total_layers, layer: None,
        post_act_fct=lambda layer_ix, total_layers, layer: None,
        allow_broadcast=False,
        use_cuda=False,
    ):
        """
        Args:
            name (str): Name identifier for the module.
            mlp_sizes (list): Dimensions of layers [input, hidden1, hidden2, ..., output].
            activation (nn.Module): Activation function for hidden layers.
            output_activation (nn.Module): Activation function for output layer(s).
            post_layer_fct (callable): Function to create a layer to insert AFTER Linear but BEFORE Activation.
            post_act_fct (callable): Function to create a layer to insert AFTER Activation.
            allow_broadcast (bool): Whether to allow broadcasting on inputs.
            use_cuda (bool): (Legacy) Previously used for DataParallel wrapping.
        """

        # init the module object
        super().__init__()

        self.name = name
        assert len(mlp_sizes) >= 2, "Must have input and output layer sizes defined"

        # Unpack structure: input -> [hidden layers] -> output
        input_size, hidden_sizes, output_size = (
            mlp_sizes[0],
            mlp_sizes[1:-1],
            mlp_sizes[-1],
        )

        assert isinstance(
            input_size, (int, list, tuple)
        ), "input_size must be int, list, tuple"

        # Calculate total input dimension (summing up if multiple inputs are provided)
        last_layer_size = input_size if type(input_size) == int else sum(input_size)

        # Initialize module list with the input concatenator
        all_modules = [ConcatModule(allow_broadcast)]

        # --- Loop over hidden layers ---
        for layer_ix, layer_size in enumerate(hidden_sizes):
            assert type(layer_size) == int, "Hidden layer sizes must be ints"

            # Create linear transformation layer
            cur_linear_layer = nn.Linear(last_layer_size, layer_size)

            # Initialize weights for numerical stability
            cur_linear_layer.weight.data.normal_(0, 0.001)
            cur_linear_layer.bias.data.normal_(0, 0.001)

            # Note: nn.DataParallel removed here for single-GPU optimization

            # Add the linear layer
            all_modules.append(cur_linear_layer)

            # 1. Handle post_linear callback (e.g., Batch Normalization)
            post_linear = post_layer_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )
            if post_linear is not None:
                all_modules.append(post_linear)

            # 2. Add Activation Function
            all_modules.append(activation())

            # 3. Handle post_activation callback (e.g., Dropout)
            post_activation = post_act_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )
            if post_activation is not None:
                print(f"Adding post activation layer: {post_activation}")
                all_modules.append(post_activation)

            # Update input size for the next layer
            last_layer_size = layer_size

        # --- Handle Output Layer(s) ---
        assert isinstance(
            output_size, (int, list, tuple)
        ), "output_size must be int, list, tuple"

        if type(output_size) == int:
            # Single output head
            all_modules.append(nn.Linear(last_layer_size, output_size))
            if output_activation is not None:
                all_modules.append(
                    call_nn_op(output_activation)
                    if isclass(output_activation)
                    else output_activation
                )
        else:
            # Multiple output heads (Multi-task)
            # We create a list of separate sequential models for each head
            out_layers = []

            for out_ix, out_size in enumerate(output_size):
                # Create a mini-sequential for this specific head
                split_layer = []

                # Linear projection
                split_layer.append(nn.Linear(last_layer_size, out_size))

                # Determine specific activation for this head
                act_out_fct = (
                    output_activation
                    if not isinstance(output_activation, (list, tuple))
                    else output_activation[out_ix]
                )

                if act_out_fct:
                    split_layer.append(
                        call_nn_op(act_out_fct) if isclass(act_out_fct) else act_out_fct
                    )

                # Add this head to the list
                out_layers.append(nn.Sequential(*split_layer))

            # Use ListOutModule to run all heads in parallel
            all_modules.append(ListOutModule(out_layers))

        # --- Final Assembly ---
        # Combine everything into the main sequential container
        self.sequential_mlp = nn.Sequential(*all_modules)

        # --- Weight Initialization ---
        # Applies specific initialization rule to all Linear layers in the model
        def weights_init_uniform_rule(m):
            import numpy as np

            classname = m.__class__.__name__
            # Apply to Linear layers only
            if classname.find("Linear") != -1:
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                # Uniform initialization based on input size
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)

        # Apply the custom initialization
        self.sequential_mlp.apply(weights_init_uniform_rule)

    def forward(self, *args, **kwargs):
        """
        Forward pass through the generated sequential model.
        """
        return self.sequential_mlp.forward(*args, **kwargs)
