# Value Functions and Bellman Equations

## Core Idea

Value functions estimate how good it is to be in a state or to take an action in a state under a policy. The state-value function V captures expected return from a state; the action-value function Q captures expected return after taking a specific action and then following the policy. Bellman equations express these values recursively: current value equals expected immediate reward plus discounted future value.

## DARL Interpretation

For DARL, value functions clarify why the cheapest immediate action is not always best and why full retraining is not always best. A preprocessing update may have a cost now but recover future AUC; no update may save resources now but worsen later returns. This is the conceptual basis for learning a policy rather than using static thresholds.

## Formula Anchor

```text
v_pi(s) = E_pi[R_{t+1} + gamma v_pi(S_{t+1}) | S_t=s]
```

The Bellman expectation equation links current state value to immediate reward and expected future value under policy pi.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- state value
- action value
- Bellman expectation
- optimality

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch03-value-functions-and-bellman-equations.md`
