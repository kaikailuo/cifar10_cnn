import pytest
import torch

from src.config import make_config
from src.models import build_model


@pytest.mark.parametrize("norm", ["none", "bn", "ln", "in", "gn"])
@pytest.mark.parametrize("pool", ["max", "blur"])
def test_vgg_shapes(norm, pool):
    model = build_model(make_config(norm=norm, pool=pool, conv_dropout=0.1, fc_dropout=0.5))
    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)


@pytest.mark.parametrize("residual", [True, False])
def test_resnet_depth_and_shape(residual):
    model = build_model(make_config(model="resnet", depth=20, norm="bn", residual=residual))
    convs = sum(isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3) for m in model.modules())
    assert convs == 19  # plus the linear layer = 20 weighted layers
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
