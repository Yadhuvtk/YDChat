from __future__ import annotations

import argparse

from ydchat.infer.generate import generate_text, load_model_and_tokenizer, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat with YDChat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    model, tokenizer = load_model_and_tokenizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=device,
    )

    print("YDChat interactive mode. Type /exit to quit.")
    history = ""

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            break

        history += f"### User:\n{user_text}\n### Assistant:\n"
        generated = generate_text(
            model,
            tokenizer,
            prompt=history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )

        reply = generated[len(history) :]
        stop_tag = "### User:"
        if stop_tag in reply:
            reply = reply.split(stop_tag, 1)[0]
        reply = reply.strip()

        print(f"YDChat: {reply}")
        history += reply + "\n"


if __name__ == "__main__":
    main()
