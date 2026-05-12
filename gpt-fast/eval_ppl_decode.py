import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from generate import _load_model, device_sync, default_device
from tokenizer import get_tokenizer


@torch.no_grad()
def eval_decode_ppl(
    model,
    tokenizer,
    texts,
    batch_size: int,
    seq_len: int,
    max_batches: int,
    device: str,
):
    """
    Decode-style PPL.

    Instead of feeding the whole sequence at once, this evaluates token-by-token:
        token[t] -> predict token[t+1]

    That matters for TEAL because the sparse kernels are decode kernels where seq_len == 1.
    """
    joined = "\n\n".join(t.strip() for t in texts if t and t.strip())
    token_ids = tokenizer.encode(joined)

    needed = batch_size * max_batches * (seq_len + 1)
    if len(token_ids) < needed:
        print(f"Warning: dataset only has {len(token_ids)} tokens, wanted {needed}.")

    chunks = []
    stride = seq_len + 1

    for start in range(0, len(token_ids) - stride, stride):
        chunk = token_ids[start : start + stride]
        if len(chunk) == stride:
            chunks.append(chunk)
        if len(chunks) >= batch_size * max_batches:
            break

    if len(chunks) < batch_size:
        raise RuntimeError("Not enough token chunks to form even one batch.")

    total_nll = 0.0
    total_tokens = 0
    num_batches = len(chunks) // batch_size

    print(f"Eval batches: {num_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Seq len: {seq_len}")

    for b in range(num_batches):
        batch_chunks = chunks[b * batch_size : (b + 1) * batch_size]

        batch = torch.tensor(batch_chunks, dtype=torch.int, device=device)  # [B, seq_len+1]
        input_tokens = batch[:, :-1]  # [B, seq_len]
        targets = batch[:, 1:]        # [B, seq_len]

        # Reset/prepare KV cache for this batch.
        # Must create caches/causal_mask on the same device as input_pos.
        with torch.device(device):
            model.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len)

        batch_nll = 0.0
        batch_tokens = 0

        for pos in range(seq_len):
            x = input_tokens[:, pos : pos + 1]  # [B, 1]
            input_pos = torch.tensor([pos], dtype=torch.int, device=device)

            logits = model(x, input_pos)  # [B, 1, vocab]
            next_logits = logits[:, -1, :].float()
            next_targets = targets[:, pos].long()

            loss = F.cross_entropy(next_logits, next_targets, reduction="sum")

            batch_nll += loss.item()
            batch_tokens += batch_size

        total_nll += batch_nll
        total_tokens += batch_tokens

        print(
            f"batch {b+1}/{num_batches}: "
            f"nll/token={batch_nll / batch_tokens:.4f}, "
            f"ppl={math.exp(batch_nll / batch_tokens):.4f}"
        )

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    return avg_nll, ppl, total_tokens


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--hist_path", type=str, default=None)
    parser.add_argument("--sparsity", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=8)

    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--device", type=str, default=default_device)

    args = parser.parse_args()

    assert args.checkpoint_path.is_file(), args.checkpoint_path

    tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    print("=" * 80)
    print("gpt-fast decode-style PPL")
    print("=" * 80)
    print(f"checkpoint_path: {args.checkpoint_path}")
    print(f"hist_path: {args.hist_path}")
    print(f"sparsity: {args.sparsity}")
    print(f"batch_size: {args.batch_size}")
    print(f"seq_len: {args.seq_len}")
    print(f"max_batches: {args.max_batches}")

    precision = torch.float16
    use_tp = False

    print("Loading model...")
    model = _load_model(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        precision=precision,
        use_tp=use_tp,
        hist_path=args.hist_path,
        sparsity=args.sparsity,
    )
    device_sync(args.device)

    tokenizer = get_tokenizer(tokenizer_path, args.checkpoint_path)

    print("Loading dataset...")
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts = ds["text"]

    avg_nll, ppl, total_tokens = eval_decode_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_batches=args.max_batches,
        device=args.device,
    )

    print("=" * 80)
    print(f"Total eval tokens: {total_tokens}")
    print(f"Average NLL/token: {avg_nll:.6f}")
    print(f"Perplexity: {ppl:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()