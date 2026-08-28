#!/usr/bin/env python3
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

r"""Bridge a live gz-transport Float_V topic to Foxglove and PlotJuggler.

`/rl/observations` (and any other `gz.msgs.Float_V` topic) carries no
schema of its own, just a flat array of floats, so neither Foxglove nor
PlotJuggler can subscribe to it directly. This script subscribes with the
real `gz.transport` Python bindings, labels each value using
`--n-agents`/`--fields`, and re-publishes every message two ways:

  - live, over the Foxglove WebSocket protocol (`ws://<host>:<port>`),
    which both Foxglove and PlotJuggler's "Foxglove Bridge" streaming
    source can connect to directly, no ROS involved
  - appended to a CSV file, for PlotJuggler's plain file loader or any
    offline analysis

See docs/reviewing_data.md for the full walkthrough.

    python scripts/gz_topic_to_foxglove.py --topic /rl/observations \\
        --n-agents 4 --fields cart_pos,cart_vel,pole_angle,pole_ang_vel

Needs `foxglove-websocket` (`python3 -m pip install foxglove-websocket`),
kept out of the pixi environment since this script is a docs/debugging
aid, not part of the trained pipeline.
"""

import argparse
import asyncio
import csv
import json
import time

from gz.msgs10.float_v_pb2 import Float_V
import gz.transport13 as gzt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--topic", default="/rl/observations", help="gz.msgs.Float_V topic to bridge")
    p.add_argument("--n-agents", type=int, default=4, help="agents batched into each message")
    p.add_argument(
        "--fields",
        default="cart_pos,cart_vel,pole_angle,pole_ang_vel",
        help="comma-separated field names for one agent's slice of the flat array",
    )
    p.add_argument("--host", default="0.0.0.0", help="Foxglove WebSocket bind address")
    p.add_argument("--port", type=int, default=8765, help="Foxglove WebSocket port")
    p.add_argument(
        "--csv", default=None, help="also append every message as a row to this CSV file"
    )
    return p.parse_args()


async def main():
    args = parse_args()
    fields = args.fields.split(",")
    n_fields = args.n_agents * len(fields)

    from foxglove_websocket.server import FoxgloveServer

    csv_writer = csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = ["timestamp"] + [f"agent{i}_{f}" for i in range(args.n_agents) for f in fields]
        csv_writer.writerow(header)
        csv_file.flush()

    loop = asyncio.get_event_loop()
    async with FoxgloveServer(
        args.host,
        args.port,
        "gazebo_gymnasium gz-transport bridge",
        capabilities=[],
        supported_encodings=["json"],
    ) as server:
        chan_id = await server.add_channel(
            {
                "topic": args.topic,
                "encoding": "json",
                "schemaName": "gazebo_gymnasium.BridgedObservations",
                "schema": json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            f"agent{i}_{f}": {"type": "number"}
                            for i in range(args.n_agents)
                            for f in fields
                        },
                    }
                ),
                "schemaEncoding": "jsonschema",
            }
        )
        print(
            f"Foxglove WebSocket server listening on ws://{args.host}:{args.port}, "
            f"channel {chan_id}"
        )
        print(
            f"subscribing to {args.topic} "
            f"({args.n_agents} agents x {len(fields)} fields = {n_fields})"
        )

        start = time.time()

        def on_message(msg: Float_V):
            data = list(msg.data)
            if len(data) != n_fields:
                return
            row = {
                f"agent{i}_{f}": data[i * len(fields) + j]
                for i in range(args.n_agents)
                for j, f in enumerate(fields)
            }
            payload = json.dumps(row).encode("utf-8")
            asyncio.run_coroutine_threadsafe(
                server.send_message(chan_id, time.time_ns(), payload), loop
            )
            if csv_writer is not None:
                csv_writer.writerow([time.time() - start] + [row[k] for k in row])
                csv_file.flush()

        node = gzt.Node()
        if not node.subscribe(Float_V, args.topic, on_message):
            raise SystemExit(f"failed to subscribe to {args.topic}")

        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
