import torch
import pytest
from tokenizer_management.data_loader.tensorizer import convert_to_tensor, tensorize_batch, Tensorizer

def test_convert_to_tensor():
    token_ids = [1, 2, 3, 4]
    tensor = convert_to_tensor(token_ids, max_length=6)
    expected = torch.tensor([1, 2, 3, 4, 0, 0], dtype=torch.long)
    assert torch.equal(tensor, expected)

def test_tensorize_batch():
    batch = [
        [1, 2, 3],
        [4, 5],
        [6]
    ]
    batch_tensor = tensorize_batch(batch, max_length=4)
    expected = torch.tensor([
        [1, 2, 3, 0],
        [4, 5, 0, 0],
        [6, 0, 0, 0]
    ], dtype=torch.long)
    assert torch.equal(batch_tensor, expected)

def test_tensorizer_class():
    tensorizer = Tensorizer(max_length=5)
    token_ids = [7, 8]
    tensor = tensorizer.tensorize_text(token_ids)
    expected = torch.tensor([7, 8, 0, 0, 0], dtype=torch.long)
    assert torch.equal(tensor, expected)

    batch = [[1, 2, 3], [4, 5]]
    batch_tensor = tensorizer.tensorize_batch_text(batch)
    expected = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 0, 0, 0]
    ], dtype=torch.long)
    assert torch.equal(batch_tensor, expected)
