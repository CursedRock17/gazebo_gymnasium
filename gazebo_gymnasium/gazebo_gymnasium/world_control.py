_TIMEOUT_MS = 5000


class WorldController:
    """Drives a Gazebo world via the WorldControl service over gz.transport."""

    def __init__(self, world_name: str, steps_per_action: int = 10):
        # Imported here so the package can be imported without Gazebo installed
        # (allows unit tests and static analysis to work in any environment).
        from gz.transport13 import Node
        from gz.msgs10.world_control_pb2 import WorldControl
        from gz.msgs10.boolean_pb2 import Boolean

        self._WorldControl = WorldControl
        self._Boolean = Boolean
        self._node = Node()
        self._steps_per_action = steps_per_action
        self._service = f"/world/{world_name}/control"

    def step(self) -> bool:
        """Step the simulation by steps_per_action physics ticks, then pause."""
        req = self._WorldControl()
        req.multi_step = self._steps_per_action
        result, _ = self._node.request(
            self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
        )
        return result

    def reset(self) -> bool:
        """Reset all entities in the world to their initial state."""
        req = self._WorldControl()
        req.reset.all = True
        result, _ = self._node.request(
            self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
        )
        return result

    def pause(self) -> bool:
        req = self._WorldControl()
        req.pause = True
        result, _ = self._node.request(
            self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
        )
        return result

    def unpause(self) -> bool:
        req = self._WorldControl()
        req.pause = False
        result, _ = self._node.request(
            self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
        )
        return result
