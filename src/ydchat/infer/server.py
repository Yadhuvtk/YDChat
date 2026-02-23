from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ydchat.infer.generate import generate_text, load_model_and_tokenizer, resolve_device


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1


class GenerateResponse(BaseModel):
    text: str


class InferenceService:
    def __init__(self, config: str, checkpoint: str, tokenizer: str, device: str) -> None:
        self.device = resolve_device(device)
        self.model, self.tokenizer = load_model_and_tokenizer(
            config_path=config,
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            device=self.device,
        )

    def generate(self, req: GenerateRequest) -> str:
        return generate_text(
            self.model,
            self.tokenizer,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            device=self.device,
        )


def create_app(config: str, checkpoint: str, tokenizer: str, device: str) -> FastAPI:
    service = InferenceService(config=config, checkpoint=checkpoint, tokenizer=tokenizer, device=device)
    app = FastAPI(title="YDChat Inference API")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        text = service.generate(req)
        return GenerateResponse(text=text)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve YDChat with FastAPI")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        config=args.config,
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
