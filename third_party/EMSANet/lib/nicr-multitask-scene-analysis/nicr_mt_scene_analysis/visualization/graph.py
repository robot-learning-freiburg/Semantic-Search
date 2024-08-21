# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch

# sudo apt install graphviz / brew install graphviz
# pip install torchviz
from torchviz import make_dot


def export_backward_graph(
    filepath: str,
    loss_tensor: torch.Tensor,
    model: torch.nn.Module
) -> None:
    """Render and export the backward graph of a model's loss tensor"""
    params = dict(model.named_parameters())

    # get graph
    graph = make_dot(loss_tensor,
                     params=params,
                     show_attrs=True,
                     show_saved=True)

    # render graph and save - do not use png for large graphs ;)
    graph.render(outfile=filepath)
