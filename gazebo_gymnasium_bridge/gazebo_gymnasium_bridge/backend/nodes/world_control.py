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

# Import Gazebo Libraries
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node


class WorldController:
    """Thin client over the /world/<name>/control service.

    Requests retry with short per-attempt timeouts so startup races (where the service hasn't been
    registered yet) don't block 5+ seconds on the first call.
    """

    # Per-attempt timeout (ms) and number of retries. The first reset on
    # startup typically needs 1–3 retries before the service is up; in
    # steady state every request succeeds on attempt 1.
    DEFAULT_ATTEMPT_TIMEOUT_MS = 500
    DEFAULT_MAX_ATTEMPTS = 60  # = 30s total max wait on startup

    def __init__(self, world_name: str, steps_per_action):
        self.world_control_node = Node()
        self.steps_per_action = steps_per_action
        self.control_service_name = "/world/" + world_name + "/control"

    def _send(
        self,
        request: WorldControl,
        attempt_timeout_ms: int = DEFAULT_ATTEMPT_TIMEOUT_MS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> bool:
        """Send a WorldControl request, retrying until the service responds.

        Returns True on success, False if all attempts exhausted.
        """
        for _ in range(max_attempts):
            response = Boolean()
            result, response = self.world_control_node.request(
                self.control_service_name, request, WorldControl, Boolean, attempt_timeout_ms
            )
            if result:
                return True
        return False

    def step(self):
        """Advance the (paused) world by steps_per_action ticks via multi_step.

        After multi_step completes, the simulation is paused again.
        """
        control_request = WorldControl()
        control_request.multi_step = self.steps_per_action
        self._send(control_request)

    def reset(self, pause_after: bool = False):
        """Reset the entire world along with all of its entities.

        Args:
            pause_after: If True, send a follow-up `pause=True` request after
                the reset completes. Combining reset+pause into a single
                WorldControl message is unreliable — Gazebo may ignore the
                pause during the reset — so we do it as two separate calls.
        """
        control_request = WorldControl()
        control_request.reset.all = True
        self._send(control_request)

        if pause_after:
            pause_request = WorldControl()
            pause_request.pause = True
            self._send(pause_request)

    def unpause(self):
        """Unpause the world regardless of its state."""
        control_request = WorldControl()
        control_request.pause = False
        self._send(control_request)
