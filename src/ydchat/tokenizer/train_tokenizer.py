from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterator

import sentencepiece as spm


def iter_corpus_text(input_path: Path, jsonl_key: str) -> Iterator[str]:
    if input_path.is_dir():
        for txt_path in sorted(input_path.rglob("*.txt")):
            yield txt_path.read_text(encoding="utf-8", errors="ignore")
        return

    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(jsonl_key)
                if isinstance(text, str) and text:
                    yield text
        return

    if input_path.suffix.lower() == ".txt":
        yield input_path.read_text(encoding="utf-8", errors="ignore")
        return

    raise ValueError(f"Unsupported input path: {input_path}")


def train_sentencepiece(
    input_path: Path,
    model_prefix: str,
    vocab_size: int,
    jsonl_key: str,
    character_coverage: float,
    model_type: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        corpus_file = Path(tmp_dir) / "corpus.txt"
        with corpus_file.open("w", encoding="utf-8") as out:
            for text in iter_corpus_text(input_path, jsonl_key=jsonl_key):
                out.write(text.replace("\n", " ").strip())
                out.write("\n")

        Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=str(corpus_file),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            bos_id=1,
            eos_id=2,
            unk_id=0,
            pad_id=3,
            normalization_rule_name="nmt_nfkc",
            train_extremely_large_corpus=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer for YDChat")
    parser.add_argument("--input", required=True, help="Path to txt folder, txt file, or jsonl file")
    parser.add_argument("--model-prefix", required=True, help="Output prefix, e.g. artifacts/tokenizer/ydchat")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--jsonl-key", default="text")
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram", "word", "char"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_sentencepiece(
        input_path=Path(args.input),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        jsonl_key=args.jsonl_key,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
    )
    print(f"Tokenizer written to: {args.model_prefix}.model")


if __name__ == "__main__":
    main()
