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


def _card(repo_id, agent, algo, env_id, n_agents, hyperparams, eval_result):
    """Build the model-card markdown (with a YAML metadata header)."""
    tags = ["reinforcement-learning", "stable-baselines3", "gazebo",
            "gymnasium", "robotics", agent]
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
        "## Training configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Environment | `{agent}`" + (f" (`{env_id}`)" if env_id else "") +
        " |",
        f"| Algorithm | {algo.upper()} |",
        f"| Agents in sim | {n_agents} |",
    ]
    for k, v in (hyperparams or {}).items():
        lines.append(f"| {k} | {v} |")
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


def push_to_hub(model_path, repo_id, *, agent, algo, env_id=None,
                n_agents=1, hyperparams=None, eval_result=None,
                private=False):
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

    Returns the repo URL.
    """
    from huggingface_hub import HfApi

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private,
                    exist_ok=True)
    api.upload_file(path_or_fileobj=str(model_path),
                    path_in_repo=MODEL_FILENAME, repo_id=repo_id)
    card = _card(repo_id, agent, algo, env_id, n_agents, hyperparams,
                 eval_result)
    api.upload_file(path_or_fileobj=card.encode("utf-8"),
                    path_in_repo="README.md", repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def pull_from_hub(repo_id, *, filename=MODEL_FILENAME):
    """Download a model ``.zip`` from the Hub; return the local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename)
