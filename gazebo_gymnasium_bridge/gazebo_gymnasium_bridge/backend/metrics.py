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
"""Shared `/env/metrics` publisher used by every env class.

Replaces the older per-env gz-transport `Float_V` publisher with a real
ROS 2 `gazebo_gymnasium_msgs/EnvMetrics` message — same six values, but
each one named so Foxglove panels can address them by field instead of by
array index.

Gracefully degrades when rclpy / the message package isn't importable
(e.g. the venv-only test path runs without ROS 2 sourced). In that case
`publish()` is a no-op so the env classes remain testable in isolation.
"""

from __future__ import annotations

try:
    import rclpy

    from gazebo_gymnasium_msgs.msg import EnvMetrics

    _HAVE_ROS = True
except Exception:  # noqa: BLE001 — any failure is "ROS not available", treat the same
    _HAVE_ROS = False


class MetricsPublisher:
    """Wraps an rclpy node + publisher for the `/env/metrics` topic.

    Instantiate once per env. Call `.publish(...)` each env.step(); call
    `.close()` when the env is being torn down (the destructor handles it
    too, but explicit `.close()` is the clean path).
    """

    TOPIC = "/env/metrics"

    def __init__(self, node_name: str = "gazebo_gym_env_metrics") -> None:
        self._enabled = _HAVE_ROS
        self._node = None
        self._pub = None
        if not self._enabled:
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node(node_name)
            self._pub = self._node.create_publisher(EnvMetrics, self.TOPIC, 10)
        except Exception as exc:  # noqa: BLE001
            # Failing to create a node should not kill the env. Surface the
            # failure once and continue silently — training still works.
            print(f"[MetricsPublisher] disabled (rclpy init failed): {exc}")
            self._enabled = False

    def publish(
        self,
        *,
        steps_per_sec: float,
        mean_step_ms: float,
        episode_reward: float,
        current_step: int,
        total_steps: int,
        current_episode: int,
    ) -> None:
        if not self._enabled or self._pub is None:
            return
        msg = EnvMetrics()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.steps_per_sec = float(steps_per_sec)
        msg.mean_step_ms = float(mean_step_ms)
        msg.episode_reward = float(episode_reward)
        msg.current_step = int(current_step)
        msg.total_steps = int(total_steps)
        msg.current_episode = int(current_episode)
        self._pub.publish(msg)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
            self._pub = None
            self._enabled = False

    def __del__(self) -> None:
        # Best-effort cleanup on GC. Wrapped in a broad try because rclpy
        # may have already been shut down by interpreter teardown.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
