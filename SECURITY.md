# Security Policy

This document follows [REP-2006](https://ros.org/reps/rep-2006.html) —
the ROS 2 Vulnerability Disclosure Policy — adapted for a
single-maintainer research package.

## Reporting a Vulnerability

Email **lwendlan@umd.edu** with vulnerability details. Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or proof-of-concept code if available.
- The version (git commit SHA or `package.xml` version) of
  `gazebo_gymnasium` you tested against.
- Any mitigations or workarounds you've identified.

Do not file public GitHub issues for security-sensitive reports — they
are visible to anyone watching the repo before a fix lands.

## Response SLA

REP-2006 commits ROS 2 core packages to a 2-business-day initial
response. This project is single-maintainer and uses a slightly
slower budget:

- **Initial acknowledgement**: within **5 business days** of receipt.
- **Risk assessment + remediation plan**: within **15 business days**
  for confirmed vulnerabilities.
- **Fix availability**: target **90 days** from initial report, per
  REP-2006's coordinated-disclosure window. Earlier if the fix is
  small; longer (with notification) if the issue requires upstream
  dependency changes.

If you don't receive an acknowledgement within 5 business days, the
email may have hit a spam filter. Ping with a second email or open a
*public* (non-security) GitHub issue saying "I emailed you about
something private 5+ days ago" without disclosing details.

## Disclosure Timeline

Per REP-2006: "we ask that you not publicly discuss the vulnerability
until a fix is published or at least 90 days have passed since the
initial report submission." Same convention applies here.

If the 90-day window will be exceeded (e.g., for a fix that requires
coordinated changes to gz-sim or Gymnasium upstream), we'll notify
the reporter with an updated timeline before the window closes.

## Safe Harbor

Mirroring REP-2006's language: **the maintainer will not engage in
legal action against individuals who act in good faith to identify,
report, and fix vulnerabilities in `gazebo_gymnasium`**, provided you:

- Make a good-faith effort to avoid privacy violations, data
  destruction, and service disruption while researching.
- Don't access more data or systems than is necessary to demonstrate
  the vulnerability.
- Give the project a reasonable time (per the SLA above) to address
  the issue before public disclosure.
- Comply with applicable laws.

## Scope

The disclosure policy covers code in this repository:

- `gazebo_gymnasium_bridge/` — env classes, helpers
- `gazebo_gymnasium_examples/` — SDF worlds, models, plugins, launch files
- `gazebo_gymnasium_msgs/` — message definitions
- `gazebo_gymnasium_reinforcement_learning/` — custom RL algorithms
- `training_scripts/` — example training entry points
- `scripts/` — generators, validators, lint helpers

Out of scope: vulnerabilities in upstream dependencies (ROS 2 core,
gz-sim, Gymnasium, Stable Baselines3, PyTorch). For those, please
report directly to the upstream maintainers:

- **ROS 2 core**: <security@openrobotics.org> (per REP-2006)
- **Gazebo**: <https://gazebosim.org/about> contact form
- **Gymnasium**: <https://github.com/Farama-Foundation/Gymnasium/security>
- **Stable Baselines3**: <https://github.com/DLR-RM/stable-baselines3/security>
- **PyTorch**: <https://pytorch.org/security/>

If you're unsure whether an issue is in scope, email anyway — better
to over-report and let me triage than to miss it.

## CVE Assignment

The maintainer will request CVE IDs through GitHub Security Advisories
for confirmed vulnerabilities. Reporters will be credited unless they
prefer anonymity (state this in the initial email).
