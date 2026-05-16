# Import Gazebo Libraries
from gz.transport13 import Node
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean


class WorldController():
    def __init__(self, world_name: str, steps_per_action):
        self.world_control_node = Node()

        self.steps_per_action = steps_per_action
        # Utilize defaulted spawn service in Gazebo
        self.control_service_name = "/world/" + world_name + "/control"

    def step(self):
        """
        Rapidly step by the number of steps_per_action to speed up sim time.
        After multi_step completes, the simulation is paused again.
        """
        control_request = WorldControl()
        control_request.multi_step = self.steps_per_action
        control_response = Boolean()
        timeout = 5000
        result, control_response = self.world_control_node.request(
            self.control_service_name, control_request, WorldControl, Boolean, timeout)

    def reset(self):
        """
        Reset the entire world along with all of it's entities.
        """
        control_request = WorldControl()
        control_request.reset.all = True
        control_response = Boolean()
        timeout = 5000
        result, control_response = self.world_control_node.request(
            self.control_service_name, control_request, WorldControl, Boolean, timeout)

    def unpause(self):
        """
        Unpause the world regardless of its state.
        """
        control_request = WorldControl()
        control_request.pause = False
        control_response = Boolean()
        timeout = 5000
        result, control_response = self.world_control_node.request(
            self.control_service_name, control_request, WorldControl, Boolean, timeout)
