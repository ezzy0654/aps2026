#!/usr/bin/env python3
"""Visualize one APSS26 prefill input and its greedy next-token output."""

import argparse
import struct
from pathlib import Path

import os

import numpy as np
from transformers import AutoTokenizer


def read_inputs(path):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError("input file is too small")

    batch, max_seq_len = struct.unpack_from("<II", data, 0)
    offset = 8
    rows = []
    for index in range(batch):
        if offset + 4 > len(data):
            raise ValueError(f"truncated input at sequence {index}")
        (length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        end = offset + 4 * length
        if length == 0 or length > max_seq_len or end > len(data):
            raise ValueError(f"invalid sequence length at index {index}: {length}")
        rows.append(list(struct.unpack_from(f"<{length}i", data, offset)))
        offset = end

    if offset != len(data):
        raise ValueError("input file has trailing bytes")
    return rows, max_seq_len


def read_outputs(path):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError("output file is too small")
    batch, vocab_size = struct.unpack_from("<II", data, 0)
    expected = 8 + batch * vocab_size * 4
    if len(data) != expected:
        raise ValueError(
            f"invalid output size: expected {expected} bytes, got {len(data)}"
        )
    logits = np.frombuffer(data, dtype=np.float32, offset=8)
    return logits.reshape(batch, vocab_size), vocab_size


def read_decode_outputs(path):
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError("decode output file is too small")
    batch, max_steps, vocab_size = struct.unpack_from("<III", data, 0)
    lengths_offset = 12
    ids_offset = lengths_offset + 4 * batch
    logits_offset = ids_offset + 4 * batch * max_steps
    expected = logits_offset + 4 * batch * max_steps * vocab_size
    if len(data) != expected:
        raise ValueError(
            f"invalid decode output size: expected {expected} bytes, got {len(data)}"
        )

    lengths = np.frombuffer(
        data, dtype=np.uint32, count=batch, offset=lengths_offset
    )
    generated_ids = np.frombuffer(
        data, dtype=np.int32, count=batch * max_steps, offset=ids_offset
    ).reshape(batch, max_steps)
    logits = np.frombuffer(
        data,
        dtype=np.float32,
        count=batch * max_steps * vocab_size,
        offset=logits_offset,
    ).reshape(batch, max_steps, vocab_size)
    return lengths, generated_ids, logits, vocab_size


def format_token(tokenizer, token_id):
    token = tokenizer.decode(
        [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    visible = repr(token)
    raw = tokenizer.convert_ids_to_tokens(int(token_id))
    return f"{visible}  (token={raw!r}, id={int(token_id)})"


def top_k_ids(row_logits, top_k):
    top_k = min(top_k, row_logits.size)
    ids = np.argpartition(row_logits, -top_k)[-top_k:]
    return ids[np.argsort(row_logits[ids])[::-1]]


def main():
    parser = argparse.ArgumentParser(
        description="Show one prompt and its greedy next-token prediction."
    )
    parser.add_argument("--input", default="/apss26/project-data/inputs.bin")
    parser.add_argument("--output", default="outputs.bin")
    parser.add_argument(
        "-d", "--decode", action="store_true",
        help="read prefill+decode output format and show every generated step",
    )
    parser.add_argument(
        "--index", type=int, required=True,
        help="zero-based sequence index (use --index 0 for the first input)",
    )
    parser.add_argument(
        "--tokenizer", default="~/.cache/apss26/phi-tiny-tokenizer",
        help="local tokenizer directory or Hugging Face model ID",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows, max_seq_len = read_inputs(args.input)
    if args.decode:
        lengths, generated_ids, logits, vocab_size = read_decode_outputs(args.output)
    else:
        logits, vocab_size = read_outputs(args.output)

    if logits.shape[0] > len(rows):
        raise ValueError(
            f"output has more sequences than input: {logits.shape[0]} vs {len(rows)}"
        )
    if not 0 <= args.index < logits.shape[0]:
        raise ValueError(
            f"index must be in [0, {logits.shape[0] - 1}], got {args.index}"
        )
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    tokenizer_source = os.path.expanduser(args.tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, trust_remote_code=True
    )
    ids = rows[args.index]

    prompt_with_special_tokens = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    prompt_text = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("=" * 72)
    print(f"Sequence #{args.index}  (zero-based index)")
    print(f"Input length: {len(ids)} / max_seq_len: {max_seq_len}")
    print("=" * 72)
    print("\n[Prompt / input]")
    print(prompt_with_special_tokens)
    print("\n[Prompt without special tokens]")
    print(prompt_text)
    print("\n[Token IDs]")
    print(ids)
    if args.decode:
        generated_length = int(lengths[args.index])
        if generated_length > logits.shape[1]:
            raise ValueError("generated length exceeds decode output capacity")
        generated = generated_ids[args.index, :generated_length].tolist()
        row_logits = logits[args.index, :generated_length]

        full_ids = ids + generated
        full_text = tokenizer.decode(
            full_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_text = tokenizer.decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print("\n[Generated text]")
        print(generated_text if generated_text else "(empty / special token only)")
        print("\n[Full prompt + generated text]")
        print(full_text)
        print("\n[Generated token steps]")
        if not generated:
            print("(no tokens generated)")
        for step, (token_id, step_logits) in enumerate(zip(generated, row_logits)):
            candidates = top_k_ids(step_logits, args.top_k)
            selected_rank = next(
                (rank for rank, candidate in enumerate(candidates, start=1)
                 if int(candidate) == int(token_id)),
                None,
            )
            rank_text = f", rank={selected_rank}" if selected_rank else ""
            print("\n" + "-" * 72)
            print(f"Step {step}: selected token{rank_text}")
            print(f"Selected: logit={step_logits[token_id]: .6f}  "
                  f"{format_token(tokenizer, int(token_id))}")
            print("Top candidates:")
            for rank, candidate in enumerate(candidates, start=1):
                marker = " <- selected" if int(candidate) == int(token_id) else ""
                print(
                    f"  {rank:2d}. logit={step_logits[candidate]: .6f}  "
                    f"{format_token(tokenizer, int(candidate))}{marker}"
                )
        print("-" * 72)
    else:
        row_logits = logits[args.index]
        top_ids = top_k_ids(row_logits, args.top_k)
        greedy_id = int(top_ids[0])

        print("\n[Greedy next-token answer]")
        print(format_token(tokenizer, greedy_id))
        print("Decoded continuation:")
        print(repr(tokenizer.decode(
            [greedy_id], skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )))
        print("\n[Top-k next-token candidates]")
        for rank, token_id in enumerate(top_ids, start=1):
            print(
                f"{rank:2d}. logit={row_logits[token_id]: .6f}  "
                f"{format_token(tokenizer, int(token_id))}"
            )
    print("=" * 72)


if __name__ == "__main__":
    main()
