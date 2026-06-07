# Patterns

## Methodology Paragraph Pattern

1. Define the RL element in general terms.
2. Map it to DARL's maintenance environment.
3. State the practical implementation choice.
4. Explain why this choice supports selective updating under drift.

Example structure:

`In reinforcement learning, a policy maps states or observations to actions that maximize expected return. In DARL, the policy maps drift diagnostics and performance signals to maintenance actions. This allows the system to learn when a cheaper stage-specific update is preferable to full retraining.`

## Formula Explanation Pattern

When adding an RL formula to the thesis:

1. Present the equation.
2. Add `Donde:`.
3. Define every symbol.
4. Explain the maintenance interpretation in Spanish.

## Guardrail Pattern

- Use MDP when discussing standard RL notation.
- Use POMDP when discussing hidden drift causes.
- Use value/return when explaining delayed effects of update actions.
- Use actor-critic/advantage when explaining PPO implementation details.
