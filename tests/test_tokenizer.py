"""Tokenizer tests that run on the local tier -- no GPU, no PyTorch.

These exist because of a real bug: `assert_special_tokens` originally asserted
that a bare `EncodeAsIds("<|im_start|>")` returned exactly one id. It does not.
SentencePiece prepends a dummy whitespace prefix by default, so a perfectly
atomic control token encodes as two ids and the gate failed on a correct
tokenizer -- discovered only after 15 minutes of corpus packing on Colab.

Training a 512-piece vocabulary on a synthetic corpus takes about a second, so
there is no excuse for not checking this locally.

Run:  python tests/test_tokenizer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinyllm.config import tok_cfg  # noqa: E402
from tinyllm.tokenizer import (  # noqa: E402
    assert_special_tokens,
    load_sp,
    roundtrip_report,
    train_sentencepiece,
    vocab_stats,
    write_corpus_sample,
)

# Enough repetition for BPE to find merges, enough variety to be realistic.
SUBJECTS = ["Lily", "Tom", "Sara", "Ben", "the cat", "the dog", "a little girl"]
VERBS = ["found", "wanted", "saw", "liked", "played with", "lost"]
OBJECTS = ["a red ball", "a big tree", "her toy", "a shiny rock", "the blue kite"]
ENDINGS = [
    "They were very happy.",
    "It was a sunny day.",
    "Then they went home.",
    "Everyone laughed and smiled.",
]


def synthetic_corpus(n: int = 4000) -> list[str]:
    import random

    rng = random.Random(0)
    out = []
    for _ in range(n):
        s = rng.choice(SUBJECTS)
        v = rng.choice(VERBS)
        o = rng.choice(OBJECTS)
        e = rng.choice(ENDINGS)
        out.append(f"One day, {s} {v} {o}. {e}")
    return out


def main() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus_file = tmp / "corpus.txt"

        texts = synthetic_corpus()
        n = write_corpus_sample(texts, corpus_file)
        print(f"corpus: {n:,} lines, {corpus_file.stat().st_size / 1e3:.0f} KB")

        # 512 is the smallest workable size: byte_fallback alone claims 256.
        sp_path = train_sentencepiece(corpus_file, tmp, vocab_size=512)
        sp = load_sp(sp_path)
        print(f"trained: {sp.GetPieceSize()} pieces\n")

        # --- 1. the regression this file exists for ------------------------
        for piece in (tok_cfg.im_start, tok_cfg.im_end):
            pid = sp.PieceToId(piece)
            bare = sp.EncodeAsIds(piece)
            print(f"  {piece!r}")
            print(f"    id                 {pid}")
            print(f"    bare encode        {[sp.IdToPiece(i) for i in bare]}")
            ctx = sp.EncodeAsIds(f"hello{piece}world")
            print(f"    in-context encode  {[sp.IdToPiece(i) for i in ctx]}")

        try:
            assert_special_tokens(sp)
            print("\n  PASS  assert_special_tokens")
        except ValueError as e:
            failures.append(f"assert_special_tokens: {e}")
            print(f"\n  FAIL  assert_special_tokens: {e}")

        # --- 2. round-trip exactness ---------------------------------------
        rt = roundtrip_report(sp, texts[:200])
        if rt["exact"]:
            print(f"  PASS  round-trip exact ({rt['chars_per_token']:.2f} chars/token)")
        else:
            failures.append(f"round-trip lossy: {rt['examples'][:1]}")
            print(f"  FAIL  round-trip lossy: {rt['examples'][:1]}")

        # Whitespace and unicode are what the identity-normalization settings
        # are protecting; assert they survive.
        for tricky in ["  double  spaces  ", "tabs\tand\nnewlines", "café naïve", "1997 vs 1998"]:
            back = sp.DecodeIds(sp.EncodeAsIds(tricky))
            if back != tricky:
                failures.append(f"round-trip changed {tricky!r} -> {back!r}")
                print(f"  FAIL  {tricky!r} -> {back!r}")
        else:
            print("  PASS  whitespace/unicode round-trip")

        # --- 3. special ids are where the config says --------------------
        st = vocab_stats(sp)
        for name, want in (
            ("unk_id", tok_cfg.unk_id), ("bos_id", tok_cfg.bos_id),
            ("eos_id", tok_cfg.eos_id), ("pad_id", tok_cfg.pad_id),
        ):
            if st[name] != want:
                failures.append(f"{name} is {st[name]}, expected {want}")
        print(f"  PASS  special ids  unk={st['unk_id']} bos={st['bos_id']} "
              f"eos={st['eos_id']} pad={st['pad_id']}")

        if st["byte_fallback_pieces"] != 256:
            failures.append(f"expected 256 byte-fallback pieces, got {st['byte_fallback_pieces']}")
        print(f"  PASS  byte fallback ({st['byte_fallback_pieces']} pieces)")

        # --- 4. the chat template tokenizes sanely -------------------------
        prompt = (
            f"{tok_cfg.im_start}user\nWrite a story.{tok_cfg.im_end}\n"
            f"{tok_cfg.im_start}assistant\n"
        )
        ids = sp.EncodeAsIds(prompt)
        n_start = ids.count(sp.PieceToId(tok_cfg.im_start))
        n_end = ids.count(sp.PieceToId(tok_cfg.im_end))
        if n_start != 2 or n_end != 1:
            failures.append(f"chat prompt has {n_start} im_start / {n_end} im_end, expected 2 / 1")
            print(f"  FAIL  chat template: {n_start} im_start, {n_end} im_end")
        else:
            print(f"  PASS  chat template ({len(ids)} tokens, 2 im_start, 1 im_end)")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  " + f)
        return 1
    print("ALL TOKENIZER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
