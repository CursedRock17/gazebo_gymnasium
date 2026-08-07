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

"""Hugging Face Hub push/pull — model-card content and the upload contract.

Network is mocked: these run offline in CI. A real round-trip against a live
Hub account is verified out of band (documented in the PR / commit), not here.
"""

from pathlib import Path
import sys
from unittest import mock

import pytest

# training_scripts/ is a sibling of the bridge package, not importable by name.
HUB = Path(__file__).resolve().parents[2] / "training_scripts"
sys.path.insert(0, str(HUB))

hub = pytest.importorskip("hub", reason="training_scripts/hub.py")


def test_model_card_documents_the_recipe():
    card = hub._card(
        repo_id="alice/ppo-hopper", agent="hopper", algo="ppo",
        env_id="GazeboHopper-v0", n_agents=16,
        hyperparams={"timesteps": 400000, "learning_rate": 3e-4},
        eval_result="Mean episode reward: 272.")
    # YAML metadata header the Hub parses
    assert card.startswith("---\nlibrary_name: stable-baselines3")
    assert "  - hopper" in card
    # the recipe is present and human-readable
    assert "GazeboHopper-v0" in card
    assert "| timesteps | 400000 |" in card
    assert "| learning_rate | 0.0003 |" in card   # 3e-4 rendered by the table
    assert "Mean episode reward: 272." in card
    # copy-paste usage that actually matches our CLI
    assert "--from-hub alice/ppo-hopper" in card


def test_card_handles_missing_optional_fields():
    card = hub._card("u/r", "cartpole", "ppo", None, 4, None, None)
    assert "Not evaluated" in card
    assert "`cartpole`" in card          # env_id omitted, no "(``)" noise
    assert "(`" not in card.split("Environment")[1].split("|")[1]


def test_push_uploads_zip_and_card(tmp_path):
    model = tmp_path / "model.zip"
    model.write_bytes(b"PK\x03\x04 fake sb3 zip")

    fake_api = mock.MagicMock()
    with mock.patch("huggingface_hub.HfApi", return_value=fake_api):
        url = hub.push_to_hub(
            model, "alice/ppo-cartpole", agent="cartpole", algo="ppo",
            n_agents=4, hyperparams={"timesteps": 100000})

    assert url == "https://huggingface.co/alice/ppo-cartpole"
    fake_api.create_repo.assert_called_once()
    assert fake_api.create_repo.call_args.kwargs.get("exist_ok") is True
    # both the model zip and the README card are uploaded
    uploaded = [c.kwargs["path_in_repo"]
                for c in fake_api.upload_file.call_args_list]
    assert hub.MODEL_FILENAME in uploaded
    assert "README.md" in uploaded


def test_push_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hub.push_to_hub(tmp_path / "nope.zip", "a/b", agent="cartpole",
                        algo="ppo")


def test_pull_downloads_named_file():
    with mock.patch("huggingface_hub.hf_hub_download",
                    return_value="/cache/model.zip") as dl:
        got = hub.pull_from_hub("alice/ppo-cartpole")
    assert got == "/cache/model.zip"
    assert dl.call_args.kwargs["repo_id"] == "alice/ppo-cartpole"
    assert dl.call_args.kwargs["filename"] == hub.MODEL_FILENAME
