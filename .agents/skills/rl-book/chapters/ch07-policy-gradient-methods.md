# Policy Gradient Methods

## Core Idea

Policy-gradient methods optimize the policy directly by adjusting parameters in the direction that increases expected return. Instead of deriving a policy from a learned value table, the policy itself is differentiable and can represent stochastic action choices. This is useful when actions are discrete but the state is high-dimensional and noisy.

## DARL Interpretation

DARL can model update choices as a stochastic policy over maintenance actions. Policy-gradient logic supports learning which action is appropriate under different combinations of drift severity, AUC drop, and compute budget. It also explains why advantage estimates are used to weight updates.

## Formula Anchor

```text
nabla J(theta) proportional E[nabla log pi_theta(a|s) A_pi(s,a)]
```

The gradient increases the probability of actions whose advantage is positive under the current policy.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- parameterized policy
- stochastic actions
- objective gradient
- return maximization

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch07-policy-gradient-methods.md`
