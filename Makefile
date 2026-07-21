# gazebo_gymnasium — common commands. Wraps scripts/env.sh so you never paste
# environment lines. Two terminals: `make sim` in one, `make train` in another.
#
#   make sim   AGENT=cartpole N=16 HEADLESS=true
#   make train AGENT=cartpole N=16 ALGO=ppo TIMESTEPS=200000
#   make deploy AGENT=cartpole N=4
#   make test
#
# Override GAZEBO_GYM_HARMONIC_WS / GAZEBO_GYM_VENV as needed (see scripts/env.sh).

AGENT     ?= cartpole
N         ?= 4
HEADLESS  ?= true
ALGO      ?= ppo
TIMESTEPS ?= 200000

SHELL := /bin/bash

.PHONY: sim train deploy test env-check help

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n",$$1,$$2}'

sim: ## launch the gz world (server terminal)
	source scripts/env.sh server && \
	  ros2 launch gazebo_gymnasium_bringup $(AGENT)_multi.launch.py \
	    n_agents:=$(N) headless:=$(HEADLESS)

train: ## train a policy (client terminal)
	source scripts/env.sh client && \
	  python3 training_scripts/train.py --agent $(AGENT) --n_agents $(N) \
	    --algo $(ALGO) --timesteps $(TIMESTEPS)

deploy: ## evaluate a saved policy (client terminal)
	source scripts/env.sh client && \
	  python3 training_scripts/deploy.py --agent $(AGENT) --n_agents $(N) \
	    --algo $(ALGO)

test: ## run the functional test suite (client env)
	source scripts/env.sh client && \
	  python3 -m pytest gazebo_gymnasium_bridge/test/test_compile.py \
	    gazebo_gymnasium_bridge/test/test_agent_spec.py \
	    gazebo_gymnasium_bridge/test/test_multi_agent_env.py \
	    gazebo_gymnasium_bridge/test/test_library_integration.py -q

env-check: ## sanity-check that the client env imports the full stack
	source scripts/env.sh client && python3 -c \
	  "import gazebo_gymnasium_bridge.envs, stable_baselines3, gz.transport13; \
	   print('client env OK:', gazebo_gymnasium_bridge.envs.registered_specs())"
