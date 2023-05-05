from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from libs import tt_lib as ttm

from transformers import BloomForCausalLM
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from loguru import logger

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_attention as bloom_attention


def run_bloom_attention_test(device):

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    block = 0
    hidden_size = hugging_bloom_reference_model.config.hidden_size # 1024
    n_head = hugging_bloom_reference_model.config.n_head
    hidden_dropout = hugging_bloom_reference_model.config.hidden_dropout
    beta = 0 # hugging_bloom_reference_model.config.? Not sure what this is

    tt_bloom_attention = bloom_attention.TtBloomAttention(device, "transformer.h", block, hugging_bloom_reference_model, hidden_size, n_head, hidden_dropout, beta)
    pt_bloom_attention = hugging_bloom_reference_model.transformer.h[block].self_attention

    # Prepare input
    torch.manual_seed(0)

    hidden_states = ((torch.rand(1, 64, hidden_size) * 2) - 1) / hidden_size
    residual = ((torch.rand(1, 64, hidden_size) * 2) - 1) / hidden_size
    alibi = ((torch.rand(n_head, 64, 64) * 2) - 1) / 64
    attention_mask = torch.randint(0, 2, (1, 1, 64, 64))

    pt_out = pt_bloom_attention.forward(hidden_states, residual, alibi, attention_mask)[0]
    print("Finished calc pt")

    tt_out = tt_bloom_attention.forward(device, hidden_states, residual, alibi, attention_mask)
    print("Finished calc tt")

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
    pt_out_unsqueezed = pt_out.unsqueeze(0)

    does_pass, pcc_message = comp_pcc(pt_out_unsqueezed, tt_out_converted, 0.98)

    print(pcc_message)

    if does_pass:
        logger.info("bloom_attention: Passed!")
    else:
        logger.warning("bloom_attention: Failed!")

    assert does_pass


def test_bloom_attention():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_attention_test(device)
    ttm.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_attention()
