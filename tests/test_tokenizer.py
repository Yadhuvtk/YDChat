from __future__ import annotations

from pathlib import Path

from ydchat.tokenizer.tokenizer import YDTokenizer
from ydchat.tokenizer.train_tokenizer import train_sentencepiece


def test_train_and_load_tokenizer(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    for i in range(20):
        (corpus_dir / f"sample_{i}.txt").write_text(
            "YDChat tokenizer test corpus with repeated words and structured text.\n" * 10,
            encoding="utf-8",
        )

    model_prefix = str(tmp_path / "ydtok")
    train_sentencepiece(
        input_path=corpus_dir,
        model_prefix=model_prefix,
        vocab_size=64,
        jsonl_key="text",
        character_coverage=1.0,
        model_type="bpe",
    )

    tok = YDTokenizer(f"{model_prefix}.model")
    ids = tok.encode("hello ydchat", add_bos=True, add_eos=True)
    assert len(ids) >= 3
    text = tok.decode(ids)
    assert isinstance(text, str)
    assert len(text) > 0
