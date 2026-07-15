import torch
import torch.distributed as dist
import math


def get_token_bytes(tokenizer):
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64)
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode([token_id])
        token_bytes[token_id] = len(decoded.encode("utf-8"))
    return token_bytes


@torch.no_grad()
def eval_bpb(
    model, dataloader, eval_steps, device, pconfig, use_amp, amp_dtype, token_bytes
):
    # bpb = cross_entropy_loss_per_token × (tokens / bytes) × (1 / ln(2))
    model.eval()
    token_bytes = token_bytes.to(device)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    for _ in range(eval_steps):
        x, y, _ = next(dataloader)
        with torch.autocast(device_type=pconfig.device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(input_ids=x).logits
        loss_per_token = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="none"
        )
        num_bytes = token_bytes[y.view(-1)]
        valid = num_bytes > 0
        total_nats += (loss_per_token * valid).sum()
        total_bytes += num_bytes.sum()
    
    if pconfig.is_distributed and pconfig.dp_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    
    model.train()
    total_bytes_val = total_bytes.item()
    if total_bytes_val == 0:
        return float("inf")
    return total_nats.item() / (math.log(2) * total_bytes_val)
