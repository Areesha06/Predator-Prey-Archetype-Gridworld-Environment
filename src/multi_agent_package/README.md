# 🐾 Multi-Agent Package

The `multi_agent_package` implements the Predator–Prey GridWorld environment.

It defines:

- Environment dynamics (physics)
- Observation plug-ins (perception)
- Reward plug-ins (incentives)
- Registry system (safe modular selection)
- Experiment scripts (orchestration)

This package does **not** contain learning algorithms.

Learning lives in the separate `baselines` module.

---

# 🧠 Conceptual Model

The system follows a strict layered structure:

```

Physics → Perception → Incentives → Learning

````

This package implements:

- **Physics** (Core)
- **Perception** (Observations)
- **Incentives** (Rewards)

Learning is external.

---

# 🏗 Architecture

## Structural View

```mermaid
flowchart TD

    Scripts --> Registry
    Registry --> Observations
    Registry --> Rewards
    Scripts --> Core
    Core --> Observations
    Core --> Rewards
````

* Scripts build the environment
* Registry selects plug-ins
* Core executes environment logic
* Observations and rewards extend behavior safely

Dependencies flow in one direction only.

---

## Execution Flow

```mermaid
sequenceDiagram

    participant Script
    participant Env
    participant Obs
    participant Rew
    participant Algo

    Script->>Env: Build environment
    Script->>Env: Attach Obs & Rew
    Script->>Algo: Instantiate algorithm(env)

    Algo->>Env: step(actions)
    Env->>Env: Update physics
    Env->>Rew: Apply reward logic
    Env->>Obs: Build observations
    Env-->>Algo: Return obs, reward
```

Algorithms only call:

```python
env.reset()
env.step(actions)
```

They never access internal state directly.

---

# 📂 Structure

```
multi_agent_package/
├── core/           # Environment dynamics
├── observations/   # Perception plug-ins
├── rewards/        # Incentive plug-ins
├── registry/       # Plug-in lookup
├── scripts/        # Experiment orchestration
└── README.md
```

---

# 🔒 Core

Defines:

* Grid construction
* Agent movement
* Capture rules
* Episode termination
* Rendering
* Seeding

Core logic is stable infrastructure and should not be modified for experiments.

---

# 👁 Observations

Define what each agent perceives.

Examples:

* Full state
* Local-only
* Radius-based partial observability

Observation builders must be pure and deterministic.

---

# 🎯 Rewards

Define incentive shaping.

Reward functions must:

* Be pure
* Not modify environment state
* Only compute scalar values

---

# 🔌 Registry

Provides name-based plug-in selection:

```python
get_observation_builder("local_only")
get_reward_function("predator_distance")
```

Enables YAML-driven configuration.

---

# ▶ Scripts

Responsible for:

* Loading configs
* Building agents
* Constructing environment
* Attaching wrappers
* Launching training

No learning logic exists here.

---

# 🧩 Extension Rules

You may safely extend:

* Observation plug-ins
* Reward plug-ins
* Experiment configurations

You should not modify:

* Core physics
* Capture rules
* Episode semantics

---

# 🔁 Reproducibility

The environment guarantees:

* Deterministic seeding
* Explicit state transitions
* Explicit reward computation
* Explicit observation construction

Identical configs produce identical trajectories.

---

# Summary

`multi_agent_package` is a modular, deterministic multi-agent environment.

It implements:

**Environment dynamics + perception + incentives.**

Learning is intentionally external.

```

---


