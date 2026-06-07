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

from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.transport13 import Node

from ..utils import math_helpers


class HandleSingleAgent(Node):

    # Spawn a Singular Entity into the World
    def __init__(self, world_name: str, entity_name: str, sdf_file: str):
        super().__init__()

        # Add information about the entity we want to spawn
        self.entity_name = entity_name
        self.entity_file = sdf_file

        # Utilize defaulted spawn service in Gazebo
        spawn_service = "/world/" + world_name + "/create"
        delete_service = "/world/" + world_name + "/remove"

        # Publisher acts like service client in gz-transport
        self.spawner_pub = super().advertise(spawn_service, EntityFactory)
        self.deleter_pub = super().advertise(delete_service, Entity)

    def spawn_entity(self, pose=None):
        """Spawn an entity in Gazebo via /world/<world_name>/create service.

        Args:
            pose (tuple): Optional (x, y, z, roll, pitch, yaw).
        """
        # Create entity request
        msg = EntityFactory()
        msg.name = self.entity_name
        with open(self.entity_file, "r") as f:
            msg.sdf = f.read()

        if pose:
            x, y, z, roll, pitch, yaw = pose
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            # Convert from rpy to quaternion
            qx, qy, qz, qw = math_helpers.euler_to_quat(roll, pitch, yaw)
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw

        print(f"[spawn_entity] Publishing entity '{self.entity_name}'")
        self.spawner_pub.publish(msg)

    def delete_entity(self):
        """Delete an entity in Gazebo via /world/<world>/remove service.

        Args:
            None
        """
        msg = Entity()
        msg.name = self.entity_name

        print(f"[delete_entity] Removing entity '{self.entity_name}' from world")
        self.deleter_pub.publish(msg)
