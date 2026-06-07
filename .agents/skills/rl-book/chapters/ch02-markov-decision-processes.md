# Markov Decision Processes

## Core Idea

An MDP formalizes sequential decision-making when the current state contains all information needed to predict the next transition and reward distribution. It is defined by states, actions, transition probabilities, rewards, and a discount factor. The Markov property is a modeling assumption: future dynamics depend on the present state and action, not the full history.

## DARL Interpretation

DARL can start from the MDP vocabulary but must be careful: drift causes are not fully observed, so the implemented state is an observation vector, not necessarily the true Markov state. Use MDP language for reward, action, return, and policy, then use POMDP language when explaining hidden drift causes.

## Formula Anchor

```text
p(s', r | s, a) = Pr(S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a)
```

The transition model defines how actions influence future states and rewards. DARL estimates this through simulated episodes rather than a known analytic transition table.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- MDP
- state transition
- reward function
- discount factor
- Markov property

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch02-markov-decision-processes.md`
