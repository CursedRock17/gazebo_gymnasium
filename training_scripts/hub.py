# Copyright 2026 Lucas Wendland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish and fetch trained policies via the Hugging Face Hub.

A trained Stable-Baselines3 model is a single ``.zip``. ``push_to_hub`` uploads
that zip alongside an auto-generated **model card** (``README.md``) documenting
the algorithm, the exact hyperparameters, the environment, and the evaluated
result — so a shared model carries the recipe that produced it. ``pull_from_hub``
downloads the zip so ``deploy.py`` can load it like any local model.

This talks to ``huggingface_hub`` directly (not ``huggingface_sb3``): fewer
dependency constraints, and full control over the model-card contents. Auth is
whatever ``huggingface_hub`` already resolves — the ``HF_TOKEN`` env var, or a
token cached by ``hf auth login`` — so no credential handling lives here.
"""

from pathlib import Path

MODEL_FILENAME = "model.zip"


def _card(
    repo_id,
    agent,
    algo,
    env_id,
    n_agents,
    hyperparams,
    eval_result,
    video_filename=None,
    curve=None,
):
    """Build the model-card markdown (with a YAML metadata header)."""
    tags = [
        "reinforcement-learning",
        "stable-baselines3",
        "gazebo",
        "gymnasium",
        "robotics",
        agent,
    ]
    lines = [
        "---",
        "library_name: stable-baselines3",
        "tags:",
        *[f"  - {t}" for t in tags],
        "---",
        "",
        f"# {algo.upper()} · {agent} (Gazebo Gymnasium)",
        "",
        f"A **{algo.upper()}** policy for the `{agent}` environment of "
        "[Gazebo Gymnasium](https://github.com/CursedRock17/gazebo_gymnasium),"
        " trained on real Gazebo Harmonic physics.",
        "",
        "## Result",
        "",
        eval_result or "_Not evaluated at upload time._",
        "",
        "See the per-environment "
        "[solved bars](https://github.com/CursedRock17/gazebo_gymnasium/blob/"
        'main/docs/examples/README.md) for what "solved" means for this task.',
        "",
    ]
    if video_filename:
        lines += [
            "## Replay",
            "",
            f"![replay of the trained policy]({video_filename})",
            "",
        ]
    lines += [
        "## Training configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Environment | `{agent}`" + (f" (`{env_id}`)" if env_id else "") + " |",
        f"| Algorithm | {algo.upper()} |",
        f"| Agents in sim | {n_agents} |",
    ]
    for k, v in (hyperparams or {}).items():
        lines.append(f"| {k} | {v} |")
    if curve:
        lines += [
            "",
            "## Learning curve",
            "",
            "| Timesteps | Mean episode reward |",
            "| --- | --- |",
        ]
        # sample up to ~12 evenly spaced points so long sweeps stay readable
        step = max(1, len(curve) // 12)
        for t, r in curve[::step]:
            lines.append(f"| {t} | {round(float(r), 2)} |")
    lines += [
        "",
        "## Usage",
        "",
        "```bash",
        f"pixi run deploy --agent {agent} --from-hub {repo_id}",
        "```",
        "",
        "Or in Python:",
        "",
        "```python",
        "from huggingface_hub import hf_hub_download",
        "from stable_baselines3 import " + algo.upper(),
        f'path = hf_hub_download("{repo_id}", "{MODEL_FILENAME}")',
        f"model = {algo.upper()}.load(path)",
        "```",
        "",
    ]
    return "\n".join(lines)


def evaluate_model(model, env, n_eval_episodes=20):
    """Deterministically evaluate a model; return (result_string, metrics).

    The string goes in the model card's Result section; the metrics dict is
    merged into the card's hyperparameter table so the number is machine-
    readable too.
    """
    from stable_baselines3.common.evaluation import evaluate_policy

    mean, std = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=True, warn=False
    )
    result = (
        f"**Mean episode reward: {mean:.1f} ± {std:.1f}** over "
        f"{n_eval_episodes} deterministic episodes."
    )
    metrics = {
        "eval_mean_reward": round(float(mean), 2),
        "eval_std_reward": round(float(std), 2),
        "eval_episodes": n_eval_episodes,
    }
    return result, metrics


def record_replay(model, wrapped_env, spec, out_path, max_steps=400, fps=20):
    """Write a replay video of the policy; return the path, or None if N/A.

    Only camera (image-observation) environments have headless RGB frames to
    record — for those we read the raw (H, W, C) camera frames from the base
    in-process env while the *wrapped* env drives the policy (so we reuse the
    already-open env and don't trip the one-camera-per-process limit). State
    environments have no headless spectator view, so this returns None and the
    caller skips the video.
    """
    if spec.image_obs is None:
        return None
    base = wrapped_env
    while hasattr(base, "venv"):  # unwrap VecTranspose/VecFrameStack
        base = base.venv
    if not hasattr(base, "_latest_obs"):
        return None

    import numpy as np

    frames = []
    obs = wrapped_env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, _i = wrapped_env.step(action)
        frames.append(np.asarray(base._latest_obs[0]).copy())
        if bool(dones[0]):
            break
    if not frames:
        return None
    # Write an animated GIF with Pillow — a guaranteed dependency in the env, so
    # no extra video package (imageio/ffmpeg) is needed. GIF embeds directly in
    # a Hugging Face model card.
    from PIL import Image

    imgs = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    imgs[0].save(
        out_path, save_all=True, append_images=imgs[1:], duration=max(1, int(1000 / fps)), loop=0
    )
    return out_path


def push_to_hub(
    model_path,
    repo_id,
    *,
    agent,
    algo,
    env_id=None,
    n_agents=1,
    hyperparams=None,
    eval_result=None,
    video_path=None,
    curve=None,
    private=False,
):
    """Upload a trained model ``.zip`` and a generated model card.

    Parameters
    ----------
    model_path : str | Path
        Local ``.zip`` produced by ``model.save(...)``.
    repo_id : str
        Target ``user/name`` (or ``org/name``) on the Hub.
    agent, algo, env_id, n_agents, hyperparams, eval_result :
        Metadata for the model card. ``hyperparams`` is any dict; the whole
        thing is rendered into a table.
    video_path : str | Path | None
        Optional replay video to upload and embed in the card.
    curve : list[tuple[int, float]] | None
        Optional (timesteps, mean_reward) learning curve to tabulate.

    Returns the repo URL.
    """
    from huggingface_hub import HfApi

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo=MODEL_FILENAME, repo_id=repo_id)

    video_filename = None
    if video_path is not None and Path(video_path).exists():
        video_filename = Path(video_path).name
        api.upload_file(
            path_or_fileobj=str(video_path), path_in_repo=video_filename, repo_id=repo_id
        )

    card = _card(
        repo_id,
        agent,
        algo,
        env_id,
        n_agents,
        hyperparams,
        eval_result,
        video_filename=video_filename,
        curve=curve,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id
    )
    return f"https://huggingface.co/{repo_id}"


def pull_from_hub(repo_id, *, filename=MODEL_FILENAME):
    """Download a model ``.zip`` from the Hub; return the local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)
