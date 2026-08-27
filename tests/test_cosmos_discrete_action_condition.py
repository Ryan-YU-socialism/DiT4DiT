import pytest
import torch

from DiT4DiT.model.modules.vlm.Cosmos25 import encode_discrete_action_condition


def test_discrete_action_condition_encodes_each_axis_as_one_hot():
    class_values = torch.tensor([-1.0, 0.0, 1.0])
    actions = class_values[torch.arange(2 * 64 * 3).reshape(2, 64, 3) % 3]

    encoded = encode_discrete_action_condition(actions, class_values)

    assert encoded.shape == (2, 64, 3, 3)
    assert encoded.dtype == torch.long
    assert torch.equal(encoded.sum(dim=-1), torch.ones(2, 64, 3, dtype=torch.long))
    assert torch.equal(encoded.argmax(dim=-1), (actions + 1).long())


def test_discrete_action_condition_respects_custom_class_order():
    class_values = torch.tensor([1.0, -1.0, 0.0])
    actions = torch.tensor([[[-1.0, 0.0, 1.0]]]).expand(1, 64, 3)

    encoded = encode_discrete_action_condition(actions, class_values)

    expected_indices = torch.tensor([[[1, 2, 0]]]).expand(1, 64, 3)
    assert torch.equal(encoded.argmax(dim=-1), expected_indices)


def test_discrete_action_condition_rejects_continuous_values():
    actions = torch.zeros(1, 64, 3)
    actions[0, 8, 2] = 0.25

    with pytest.raises(ValueError, match="outside action_class_values"):
        encode_discrete_action_condition(actions, torch.tensor([-1.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    "actions",
    [
        torch.zeros(64, 3),
        torch.zeros(1, 64, 3, 1),
    ],
)
def test_discrete_action_condition_rejects_wrong_rank(actions):
    with pytest.raises(ValueError, match=r"shape \[B, horizon, action_dim\]"):
        encode_discrete_action_condition(actions, torch.tensor([-1.0, 0.0, 1.0]))
