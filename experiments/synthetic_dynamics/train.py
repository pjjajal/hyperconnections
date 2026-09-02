"""Train one dynamic hyperconnection model on one synthetic task."""

import argparse
import contextlib
import json
from pathlib import Path

import torch

from hyperconnections import cghc, ghc, mhc

from .models import StreamDynamics, ZeroModule
from .tasks import SyntheticTask, relative_error


# Fixed experiment constants. Change these here, not through a sprawling CLI.
N_STREAMS = 4
FEATURE_DIM = 32
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EVAL_BATCHES = 2000
LOG_POINTS = 20
DT = 0.01
DT_MIN = 0.001
DT_MAX = 0.40


def autocast_ctx(device: str) -> contextlib.AbstractContextManager:
    device_type = device.split(":", maxsplit=1)[0]
    if device_type in {"cuda", "mps"}:
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_model(name: str, depth: int) -> StreamDynamics:
    common = {
        "n": N_STREAMS,
        "m": N_STREAMS,
        "input_dim": N_STREAMS * FEATURE_DIM,
        "embed_dim": N_STREAMS * FEATURE_DIM,
        "module": ZeroModule(),
    }
    if name == "cghc":
        layer = cghc.ContinuousGenHyperConnections(
            **common,
            generator_type="conservative_psd_diss",
            dt=DT,
            dt_min=DT_MIN,
            dt_max=DT_MAX,
            learn_dt=True,
            vec_dt=True,
            use_triton=False,
        )
    elif name == "mhc":
        layer = mhc.ManifoldHyperConnections(**common)
    elif name == "ghc":
        layer = ghc.GeneralizedHyperConnections(**common)
    else:
        raise ValueError(f"unknown model: {name}")
    return StreamDynamics(layer, N_STREAMS, FEATURE_DIM, depth)


@torch.no_grad()
def evaluate(
    model: StreamDynamics,
    task: SyntheticTask,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    for _ in range(EVAL_BATCHES):
        state, target, noise = task.sample(BATCH_SIZE, generator)
        state = state.to(device)
        target = target.to(device)
        noise = None if noise is None else noise.to(device)
        with autocast_ctx(str(device)):
            prediction = model(state, noise)
        for key, value in task.metrics(
            prediction.float(), target.float(), noise
        ).items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / EVAL_BATCHES for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", choices=["preservation", "rotation", "permutation", "filtering"]
    )
    parser.add_argument("model", choices=["cghc", "mhc", "ghc"])
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    task = SyntheticTask.create(
        args.task, N_STREAMS, FEATURE_DIM, args.depth, args.seed
    )
    model = build_model(args.model, args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_generator = torch.Generator().manual_seed(args.seed + 1)

    log_every = max(1, args.steps // LOG_POINTS)
    history = []
    running_loss = 0.0
    running_steps = 0
    model.train()
    for step in range(1, args.steps + 1):
        state, target, noise = task.sample(BATCH_SIZE, train_generator)
        state = state.to(device)
        target = target.to(device)
        noise = None if noise is None else noise.to(device)

        with autocast_ctx(str(device)):
            prediction = model(state, noise)
        loss = relative_error(prediction.float(), target.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_steps += 1
        if step == 1 or step % log_every == 0 or step == args.steps:
            evaluation = evaluate(
                model,
                task,
                device,
                torch.Generator().manual_seed(args.seed + 2),
            )
            entry = {
                "step": step,
                "train_loss": running_loss / running_steps,
                "val_loss": evaluation["error"],
                **{
                    f"val_{key}": value
                    for key, value in evaluation.items()
                    if key != "error"
                },
            }
            history.append(entry)
            values = " ".join(
                f"{key}={value:.6f}"
                for key, value in entry.items()
                if key != "step"
            )
            print(f"step={step} {values}")
            running_loss = 0.0
            running_steps = 0
            model.train()

    output = args.output or Path(
        f"experiments/synthetic_dynamics/results/"
        f"{args.task}_{args.model}_L{args.depth}_seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "task": args.task,
                "model": args.model,
                "depth": args.depth,
                "steps": args.steps,
                "seed": args.seed,
                "history": history,
                **evaluation,
            },
            indent=2,
        )
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
