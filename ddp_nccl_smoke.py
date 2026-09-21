#!/usr/bin/env python3
"""Minimal four-GPU NCCL and optimizer smoke test."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = torch.nn.Linear(16, 8, bias=True).to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    generator = torch.Generator(device=device).manual_seed(1000 + local_rank)
    inputs = torch.randn(4, 16, generator=generator, device=device)
    loss = ddp_model(inputs).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    checksum = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).sum()
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    if dist.get_rank() == 0:
        values = [value.item() for value in gathered]
        assert max(values) - min(values) < 1e-5, values
        print(f"DDP_NCCL_OK world_size={dist.get_world_size()} checksums={values}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
