"""Chat interface for KanpreyLM."""

import argparse
import sys
import torch

import kanprey
import kanprey.config as kanprey_config
import kanprey.kan_layers as kanprey_kan_layers
import kanprey.model as kanprey_model
import kanprey.moe_layers as kanprey_moe_layers
from kanprey.config import ModelConfig, TrainConfig
from kanprey.dataset import load_tokenizer, TEMPLATE
from kanprey.model import KANpreyLM, KATpreyLM, MLPEdgepreyLM, MLPTransformer, GRKANpreyLM, BasisKANpreyLM, SwiGLUTransformer
from kanprey.train import detect_device


IM_END = "<|im_end|>"


def _install_legacy_module_aliases():
    """
    Older checkpoints were pickled under the pre-rename package name `kanpy`.
    Register lightweight aliases before torch.load so those checkpoints remain
    readable without editing the pickle payload.
    """
    sys.modules.setdefault("kanpy", kanprey)
    sys.modules.setdefault("kanpy.config", kanprey_config)
    sys.modules.setdefault("kanpy.model", kanprey_model)
    sys.modules.setdefault("kanpy.kan_layers", kanprey_kan_layers)
    sys.modules.setdefault("kanpy.moe_layers", kanprey_moe_layers)


def load_model(checkpoint: str, device: torch.device):
    _install_legacy_module_aliases()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg", ModelConfig())
    model_type = ckpt.get("model_type", "kan")

    if model_type == "kat":
        model = KATpreyLM(model_cfg).to(device)
    elif model_type == "mlpedge":
        model = MLPEdgepreyLM(model_cfg).to(device)
    elif model_type == "mlp":
        model = MLPTransformer(model_cfg).to(device)
    elif model_type == "swiglu":
        model = SwiGLUTransformer(model_cfg).to(device)
    elif model_type == "grkan":
        model = GRKANpreyLM(model_cfg).to(device)
    elif model_type == "kan":
        model = KANpreyLM(model_cfg).to(device)
    elif model_type == "basis":
        model = BasisKANpreyLM(model_cfg).to(device)
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type!r}")
    state = ckpt["model"]
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if not k.endswith(".mask")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint} incompatible with model_type={model_type!r}: "
            f"missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, model_cfg


def chat_completion(
    prompt: str,
    model: KANpreyLM,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
) -> str:
    # Template already opens with <|im_start|> which serves as BOS — no extra token needed.
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(text).ids
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    # Decode only the new tokens
    new_ids = out[0, len(ids):].tolist()
    response = tokenizer.decode(new_ids)

    # Trim at <|im_end|>
    if IM_END in response:
        response = response[: response.index(IM_END)]
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with KanpreyLM")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    device = detect_device()
    tokenizer = load_tokenizer(args.tokenizer)
    model, cfg = load_model(args.checkpoint, device)
    print(f"KanpreyLM loaded  |  {sum(p.numel() for p in model.parameters()):,} params")

    def respond(prompt: str) -> str:
        return chat_completion(prompt, model, tokenizer, device,
                               max_new_tokens=args.max_tokens,
                               temperature=args.temperature,
                               top_k=args.top_k)

    if args.prompt:
        print(respond(args.prompt))
        return

    print("KAN-Guppy is ready. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        response = respond(user_input)
        print(f"Guppy: {response}\n")


if __name__ == "__main__":
    main()
