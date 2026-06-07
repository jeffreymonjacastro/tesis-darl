---
name: rl-book
description: "Knowledge base from Sutton and Barto's Reinforcement Learning: An Introduction for DARL methodology. Use for RL foundations, MDPs, value functions, Bellman equations, Monte Carlo, TD learning, function approximation, policy gradients, actor-critic, partial observability, reward design, and mapping RL concepts to DARL."
---

# RL Book

Use this skill when the thesis or code work needs foundations from Sutton and Barto's *Reinforcement Learning: An Introduction*. It follows the `book-to-skill` pattern: this `SKILL.md` is a compact router; load only the chapter needed for the current RL concept.

## Source Scope

- Source: `RL An Introduction.pdf`.
- Extracted with: `pdftotext` in `text` mode.
- Approximate source size: 548 pages, 261243 words, ~348K tokens.
- Generated: 2026-06-07.

## How To Use

1. Identify the RL concept in the user request.
2. Read only the matching chapter under `chapters/`.
3. Use `glossary.md` for terms, `patterns.md` for methodology-writing templates, and `cheatsheet.md` for fast routing.
4. When writing LaTeX formulas, follow the project rule: formula, `Donde:`, explanation of symbols, and conceptual explanation.

## Chapter Router

- `ch01-reinforcement-learning-problem.md` - agent, environment, state, action, reward, return, policy.
- `ch02-markov-decision-processes.md` - MDP, state transition, reward function, discount factor, Markov property.
- `ch03-value-functions-and-bellman-equations.md` - state value, action value, Bellman expectation, optimality.
- `ch04-dynamic-programming.md` - policy evaluation, policy improvement, value iteration, known models.
- `ch05-monte-carlo-and-temporal-difference-learning.md` - Monte Carlo return, TD error, bootstrapping, online learning.
- `ch06-function-approximation.md` - generalization, parameterized value functions, features, neural approximators.
- `ch07-policy-gradient-methods.md` - parameterized policy, stochastic actions, objective gradient, return maximization.
- `ch08-actor-critic-and-advantage.md` - actor, critic, advantage, variance reduction, PPO connection.
- `ch09-partial-observability.md` - hidden state, observations, belief, history, POMDP connection.
- `ch10-darl-rl-design.md` - DARL state, action, reward, episode, baselines, thesis wording.

## DARL Mapping

- **Agent:** maintenance controller.
- **Environment:** two-stage tabular pipeline plus stream/drift process.
- **Observation/state:** drift statistics, AUC drop, detector outputs, cost signals, recent action context.
- **Actions:** no update, update preprocessing, update predictive model, full update.
- **Reward:** performance recovery minus action cost.
- **Episode:** simulated pre-drift and post-drift sequence over dataset windows.

## Guardrails

- Use this skill for RL foundations, not for paper-specific drift claims.
- For PPO details, also consult `darl-papers` chapter on PPO.
- For POMDP literature support, also consult `darl-papers` chapter on partially observable stochastic domains.
- Do not merge this skill with `darl-papers`; the book is intentionally separate to avoid loading an entire RL reference for a single paper question.
