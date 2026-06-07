# Reinforcement Learning Problem

## Core Idea

Reinforcement learning studies an agent that interacts with an environment over time. At each step the agent receives an observation or state, chooses an action, obtains a reward, and transitions to a new situation. The objective is not to predict a label for a fixed dataset, but to learn a policy that chooses actions with high long-term return. For DARL, the agent is a maintenance controller and the environment is the deployed two-stage tabular pipeline under changing data conditions.

## DARL Interpretation

The DARL policy should interpret drift and performance signals as observations, then choose among maintenance actions: no update, preprocessing update, model update, or full update. The reward should measure the quality of the decision, combining performance recovery with update cost. This framing prevents the thesis from treating retraining as a one-shot classification rule.

## Formula Anchor

```text
G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

The return G_t accumulates future rewards with discount gamma. In DARL, future rewards matter because an update action can affect several later windows.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- agent
- environment
- state
- action
- reward
- return
- policy

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch01-reinforcement-learning-problem.md`
