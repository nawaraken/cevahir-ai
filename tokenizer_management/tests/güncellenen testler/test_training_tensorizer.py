import torch
import pytest
from tokenizer_management.training.training_tensorizer import TrainingTensorizer

def test_tensorize_multiple():
    """
    Test that tensorize converts a list of postprocessed strings into a padded tensor.
    For example, for input ["hello", "world!"]:
      - "hello" -> [104, 101, 108, 108, 111]
      - "world!" -> [119, 111, 114, 108, 100, 33]
    The expected output is a tensor of shape (2, 6), where "hello" is padded with a 0.
    """
    tensorizer = TrainingTensorizer()
    inputs = ["hello", "world!"]
    result = tensorizer.tensorize(inputs)

    # Expected conversion:
    # "hello" -> [104, 101, 108, 108, 111]
    # "world!" -> [119, 111, 114, 108, 100, 33]
    # After padding "hello" becomes [104, 101, 108, 108, 111, 0]
    expected = torch.tensor([
        [104, 101, 108, 108, 111, 0],
        [119, 111, 114, 108, 100, 33]
    ], dtype=torch.long)

    assert result.shape == (2, 6)
    assert torch.equal(result, expected)

def test_tensorize_empty():
    """
    Test that providing an empty list returns an empty tensor.
    """
    tensorizer = TrainingTensorizer()
    result = tensorizer.tensorize([])
    # When no texts are provided, the tensor should have 0 elements.
    assert result.numel() == 0

def test_tensorize_single():
    """
    Test that tensorizing a single string returns a tensor of correct shape and values.
    """
    tensorizer = TrainingTensorizer()
    inputs = ["test"]
    result = tensorizer.tensorize(inputs)

    # "test" -> [116, 101, 115, 116]
    expected = torch.tensor([[116, 101, 115, 116]], dtype=torch.long)

    assert result.shape == (1, 4)
    assert torch.equal(result, expected)
