from __future__ import annotations

import argparse

from ydchat.infer.generate import generate_text, load_model_and_tokenizer, resolve_device, set_seed


def format_instruction_prompt(instruction: str, input_text: str = "") -> str:
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n"
        "### Input:\n"
        f"{input_text.strip()}\n"
        "### Response:\n"
    )


def clean_reply(generated: str, prompt: str) -> str:
    reply = generated[len(prompt) :] if generated.startswith(prompt) else generated
    for stop_tag in ["### Instruction:", "### Input:", "### Response:", "### User:", "### Assistant:"]:
        if stop_tag in reply:
            reply = reply.split(stop_tag, 1)[0]
    return reply.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat with YDChat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["instruction", "chat"], default="instruction")
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

        if args.mode == "instruction":
            prompt = format_instruction_prompt(user_text, "")
        else:
            history += f"### User:\n{user_text}\n### Assistant:\n"
            prompt = history

        generated = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )

        reply = clean_reply(generated, prompt)
        print(f"YDChat: {reply}")

        if args.mode == "chat":
            history += reply + "\n"


if __name__ == "__main__":
    main()
