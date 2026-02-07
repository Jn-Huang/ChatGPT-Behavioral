# Comprehensive Review: Prisoner's Dilemma Game Implementation

## 1. Overview

This repository implements a behavioral study comparing how ChatGPT models (GPT-3.5-turbo and GPT-4) play the Prisoner's Dilemma game relative to human participants. The game is framed as a "Push/Pull" card game. The study covers data collection via the OpenAI API, automated response extraction, multi-round game simulation with scripted opponent strategies, and statistical analysis of cooperation rates and utility-theoretic preference estimation.

---

## 2. Game Design

### 2.1 Framing

The Prisoner's Dilemma is presented as a card game rather than using the traditional "cooperate/defect" framing. Each player holds two cards:

- **Push** (cooperative action)
- **Pull** (defective action)

### 2.2 Payoff Matrix

| Your Card | Opponent's Card | Your Payoff | Opponent's Payoff |
|-----------|----------------|-------------|-------------------|
| Push      | Push           | $400        | $400              |
| Push      | Pull           | $0          | $700              |
| Pull      | Push           | $700        | $0                |
| Pull      | Pull           | $300        | $300              |

This is a standard Prisoner's Dilemma structure:
- **Temptation (T)** = $700 (defect while other cooperates)
- **Reward (R)** = $400 (mutual cooperation)
- **Punishment (P)** = $300 (mutual defection)
- **Sucker (S)** = $0 (cooperate while other defects)
- Satisfies T > R > P > S and 2R > T + S

### 2.3 Game Variants

| Variant | Rounds | Opponent Strategy | Models Tested | Instances |
|---------|--------|-------------------|---------------|-----------|
| Two-round, opponent plays Push | 2 | Always Push (cooperate) | GPT-4, GPT-3.5-turbo | 30 each |
| Two-round, opponent plays Pull | 2 | Always Pull (defect) | GPT-4, GPT-3.5-turbo | 30 each |
| Five-round, mixed opponent | 5 | Pull, Pull, Push, Push (rounds 2-5) | GPT-4, GPT-3.5-turbo | 30 each |
| With occupation context | 2 or 5 | Various | GPT-4, GPT-3.5-turbo | 30 per occupation |

The five-round variant uses a specific opponent strategy sequence:
- Round 1: Model makes a blind choice
- Round 2: Opponent played Pull (defect)
- Round 3: Opponent played Pull (defect)
- Round 4: Opponent played Push (cooperate)
- Round 5: Opponent played Push (cooperate)

This tests whether the model reciprocates or maintains a fixed strategy.

---

## 3. Prompts Used

### 3.1 Game Introduction
```
Hi, let's play a fun card game.
```
System message: `"You are a helpful assistant."`

### 3.2 First Round Prompt
```
You're paired with another player, each having a $400 'Push' card and a $300 'Pull'
card. Your payoff depends on both players' card choices. Here are the scenarios:

* Both play 'Push': Each earns $400
* You play 'Push', the other player plays 'Pull': You earn $0, the other player earns $700
* You play 'Pull', the other player plays 'Push': You earn $700, the other player earns $0
* Both play 'Pull': Each earns $300

In 5 total rounds, which card would you play in the first round? Please give a
concrete choice and highlight it with [] (i.e., [Push] or [Pull]).
```

### 3.3 Subsequent Round Prompt (Template)
```
In the {ordinal} round, you played '{your_card}' and the other player played
'{opponent_card}' card. You earned ${your_payoff} and the other player earned
${opponent_payoff}. Which card would you play in the next round? Please give a
concrete choice and highlight it with [] (e.g., [Choice]).
```

### 3.4 Response Extraction Prompt
A secondary ChatGPT-3 (turbo) call parses the model's free-text response:
```
Below is a response of a game player, who proposed to play a card to another player.
What's the card (Push or Pull) this player proposed to play? Please output only the
card. If there is no answer or the answer is not clear, please output None.

Example:

Input: "Based on the payoffs, if I play [Push] and my opponent also plays [Push],
I'll earn $400. However, if I play [Pull] and my opponent plays [Push], I'll earn
$700. The same is true if I play [Push] and my opponent plays [Pull].

Since the payoffs are higher in the case where I play [Pull], I would choose to
play [Pull] in the first round."

Output: Pull

Input: "{model_response}"

Output:
```

---

## 4. Data Collection Pipeline

### 4.1 API Configuration

| Parameter | Value |
|-----------|-------|
| GPT-4 model | `gpt-4-0314` |
| GPT-3.5-turbo model | `gpt-3.5-turbo-0301` |
| System message | `"You are a helpful assistant."` |
| Timeout per call | 30 seconds |
| Instances per condition | 30 |

### 4.2 Collection Flow (`run-openai.ipynb`)

The `run_PD()` function implements the game loop:

1. Initialize conversation with system message
2. Send game introduction (`prompt_init`)
3. Send rules and first-round prompt (`prompt_1`)
4. Extract card choice via secondary LLM call
5. Validate extraction (must be exactly "Push" or "Pull")
6. For rounds 2-5: compute payoffs, send round result + next prompt, extract choice
7. Store full message history, raw API responses, and extracted choices

```python
payoff_dict = {
    ('Push', 'Push'): 400,
    ('Push', 'Pull'): 0,
    ('Pull', 'Push'): 700,
    ('Pull', 'Pull'): 300
}
```

### 4.3 Record Format (JSON)

Each experiment produces a JSON file with:
```json
{
  "messages": [
    [  // session 1
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hi, let's play a fun card game."},
      {"role": "assistant", "content": "..."},
      ...
    ],
    ...  // 30 sessions
  ],
  "responses": [[...], ...],  // raw API response objects
  "choices": [
    ["Push", "Pull"],           // 2-round game
    ["Push", "Pull", "Push", "Push", "Push"],  // 5-round game
    ...
  ]
}
```

### 4.4 Record Files

| File | Model | Variant | Size |
|------|-------|---------|------|
| `PD_gpt4_two_rounds_push_*.json` | GPT-4 | 2-round, opp=Push | 125 KB |
| `PD_gpt4_two_rounds_pull_*.json` | GPT-4 | 2-round, opp=Pull | 137 KB |
| `PD_turbo_two_rounds_push_*.json` | GPT-3.5 | 2-round, opp=Push | 128 KB |
| `PD_turbo_two_rounds_pull_*.json` | GPT-3.5 | 2-round, opp=Pull | 141 KB |
| `PD_gpt4_five_rounds_pull_*.json` | GPT-4 | 5-round, mixed | 277 KB |
| `PD_turbo_five_rounds_pull_*.json` | GPT-3.5 | 5-round, mixed | 276 KB |
| `*_occupations_described_*.json` | Both | With occupation priming | 1-2 MB each |

---

## 5. Data Processing

### 5.1 Human Data (`data/push_pull.csv`)

**Source:** ~293,638 rows from a behavioral experiment platform.

**Filtering steps** (in `plot.ipynb`):
1. Filter to `gameType == 'push_pull'`
2. Filter to `Role == 'player'`
3. Filter to valid moves (`move == 0` or `move == 1`), where 0 = Push (defect in original encoding), 1 = Pull (cooperate)
4. Filter to `groupSize == 2` (pairwise games only)

**Derived columns:**
- `playID`: Auto-incremented session identifier (resets at each Round 1)
- `otherMove`: Extracted from `roundResult` array (opponent's action in that round)

**Five-round filtering:** For comparable analysis, only human sessions matching the same opponent strategy sequence (Pull, Pull, Push, Push in rounds 2-5) are retained via intersection of `playID` sets.

### 5.2 Model Data Extraction

The `extract.ipynb` notebook contains the `extract_chatgpt()` function that uses GPT-3.5-turbo as a parser to convert free-text responses into structured "Push" or "Pull" labels. Validated answers must be exactly one of these two strings; otherwise the session is retried.

### 5.3 Encoding Convention

**Important:** The human data and model data use different conventions:
- **Human data:** `move=0` = Push (defect), `move=1` = Pull (cooperate)
- **Model data:** "Push" = cooperate, "Pull" = defect

In the game as presented to models, **Push is the cooperative action** (mutual Push = $400 each). In the human platform data, `move=1` (Pull) corresponds to cooperation. The analysis code accounts for this by mapping appropriately when computing cooperation rates.

For the first-round analysis used in the paper:
- Human cooperation count: `n_coo_human = 36,269` (coded as `move=1` in human data)
- Human defection count: `n_def_human = 44,114` (coded as `move=0` in human data)

---

## 6. Statistics and Results

### 6.1 First-Round Cooperation Rates

| Agent | Cooperate | Defect | Cooperation Rate |
|-------|-----------|--------|-----------------|
| **Human** | 36,269 | 44,114 | **45.1%** |
| **GPT-4** | 55 (29+0+0+26) | 5 (0+1+1+3) | **91.7%** |
| **GPT-3.5-turbo** | 46 (21+3+7+15) | 14 (3+3+4+4) | **76.7%** |

GPT-4 and GPT-3.5 first-round counts combine data from all two-round variants (push + pull opponent conditions, aggregating both round-1 and round-2 choices).

### 6.2 Two-Round Game: Detailed Transition Counts

#### GPT-4, Opponent = Push (30 instances)
| Round 1 | Round 2 | Count |
|---------|---------|-------|
| Push | Push | 29 |
| Pull | Pull | 1 |

Overall cooperation (Push) rate: **96.7%**. Nearly deterministic cooperative behavior when the opponent cooperates.

#### GPT-4, Opponent = Pull (30 instances)
| Round 1 | Round 2 | Count |
|---------|---------|-------|
| Push | Pull | 26 |
| Pull | Pull | 3 |
| Pull | Push | 1 |

GPT-4 starts cooperative (26/30 Push in round 1) but shifts to defection after seeing the opponent defect. This is a **tit-for-tat-like** pattern.

#### GPT-3.5-turbo, Opponent = Push (30 instances)
| Round 1 | Round 2 | Count |
|---------|---------|-------|
| Push | Push | 21 |
| Push | Pull | 3 |
| Pull | Push | 3 |
| Pull | Pull | 3 |

Overall cooperation rate: **80.0%**. More variable than GPT-4.

#### GPT-3.5-turbo, Opponent = Pull (30 instances)
| Round 1 | Round 2 | Count |
|---------|---------|-------|
| Push | Pull | 15 |
| Push | Push | 7 |
| Pull | Pull | 4 |
| Pull | Push | 4 |

Overall cooperation rate: **55.0%**. More scattered, less consistent tit-for-tat.

### 6.3 Five-Round Game: Round-by-Round Results

#### GPT-4 Five-Round (opponent: Pull, Pull, Push, Push)
| Round | Push (coop) | Pull (defect) | Coop Rate |
|-------|-------------|---------------|-----------|
| 1 | 29 | 1 | 96.7% |
| 2 (opp=Pull) | 1 | 29 | 3.3% |
| 3 (opp=Pull) | 25 | 5 | 83.3% |
| 4 (opp=Push) | 21 | 9 | 70.0% |
| 5 (opp=Push) | 25 | 5 | 83.3% |

GPT-4 strongly reciprocates: after seeing opponent Pull in round 1, it switches almost entirely to Pull in round 2, then partially recovers when the opponent strategy changes.

#### GPT-3.5-turbo Five-Round (opponent: Pull, Pull, Push, Push)
| Round | Push (coop) | Pull (defect) | Coop Rate |
|-------|-------------|---------------|-----------|
| 1 | 21 | 9 | 70.0% |
| 2 (opp=Pull) | 7 | 23 | 23.3% |
| 3 (opp=Pull) | 14 | 16 | 46.7% |
| 4 (opp=Push) | 15 | 15 | 50.0% |
| 5 (opp=Push) | 14 | 16 | 46.7% |

GPT-3.5 shows less clear reciprocation patterns and higher overall variability.

### 6.4 Sankey Diagram Visualization

The `plot-sankey.ipynb` notebook creates Sankey flow diagrams showing round-to-round transitions. Nodes represent "Cooperate" (green) and "Defect" (red) per round. The flow widths represent transition frequencies.

Key Sankey values used in the notebook:

**Two-round, Opponent=Push:**
- Human: [12035, 3173, 4963, 8440] (Push->Push, Push->Pull, Pull->Push, Pull->Pull)
- GPT-4: [29, 0, 0, 1]
- GPT-3.5: [21, 3, 3, 3]

**Two-round, Opponent=Pull:**
- Human: [3821, 7686, 3635, 10584]
- GPT-4: [0, 26, 1, 3]
- GPT-3.5: [7, 15, 4, 4]

**Five-round (GPT-4):** [0, 29, 1, 0, 0, 1, 25, 4, 17, 8, 4, 1, 17, 4, 8, 1]
**Five-round (GPT-3.5):** [6, 15, 1, 8, 5, 2, 9, 14, 10, 4, 5, 11, 12, 3, 2, 13]

---

## 7. Analytical Framework

### 7.1 CES Utility Function

The analysis uses a Constant Elasticity of Substitution (CES) utility function to model player preferences between self-interest and altruism:

```
U(b, S, P, r) = (b * S^r + (1-b) * P^r)^(1/r)
```

Where:
- `b` = selfishness parameter (0 = fully altruistic, 1 = fully selfish)
- `S` = player's own payoff
- `P` = partner's payoff
- `r` = substitution elasticity parameter (tested at r=0.5 and r=1.0)

When `r=1`, the utility is a simple linear weighted sum. When `r<1`, there is diminishing marginal utility (favoring more equal distributions).

### 7.2 Expected Payoff Calculation for PD

For the Prisoner's Dilemma, expected payoffs depend on the probability of cooperation (`coo_prob`) and the empirical base rate of human cooperation:

```python
def expected_payoff(coo_prob):
    payoff_player = (
        coo_prob * r_coo * 400 +      # both cooperate
        coo_prob * r_def * 0 +         # I cooperate, they defect
        (1-coo_prob) * r_coo * 700 +   # I defect, they cooperate
        (1-coo_prob) * r_def * 300     # both defect
    )
    payoff_partner = (
        coo_prob * r_coo * 400 +       # both cooperate
        coo_prob * r_def * 700 +       # I cooperate, they defect
        (1-coo_prob) * r_coo * 0 +     # I defect, they cooperate
        (1-coo_prob) * r_def * 300     # both defect
    )
    return payoff_player, payoff_player + payoff_partner, payoff_partner
```

The opponent's strategy distribution is assumed to match the human base rate (`r_coo_human = 0.451`, `r_def_human = 0.549`).

### 7.3 Optimization Squared Error (SE)

The SE metric measures how far each player's actual utility is from the theoretical maximum under a given CES parameterization:

```
SE = (1 - actual_utility / theoretical_max)^2
```

This is computed for each player across a sweep of `b` values from 0 to 1 (11 points). The resulting SE curves show which value of `b` best explains each agent's behavior:
- **Humans** tend to minimize SE at intermediate `b` values (~0.5-0.7)
- **GPT-4** minimizes SE at lower `b` values (more altruistic/cooperative)
- **GPT-3.5** falls between humans and GPT-4

### 7.4 Cluster-Robust Standard Errors

To account for within-subject correlation in repeated measures:

```python
import statsmodels.api as sm
clustered_results = sm.OLS(SE_values, np.ones(len(SE_values))).fit(
    cov_type='cluster', cov_kwds={'groups': game_labels}
)
MSE = clustered_results.predict(1)[0]
clustered_SE = clustered_results.bse[0]
corrected_CI = stats.norm.interval(0.95, loc=MSE, scale=clustered_SE)
```

### 7.5 Multinomial Logistic Regression (Beta Estimation)

The `estimate_beta()` function fits a single preference parameter `b` using multinomial logistic regression:

```python
def estimate_beta(S, P, k, r=1.0):
    # S = player payoffs for each action
    # P = partner payoffs for each action
    # k = observed action indices (0=cooperate, 1=defect)

    def multinomial_log_likelihood(beta, S, P, k):
        logistic_scores = CES(beta, S, P, r=r)
        exp_scores = np.exp(logistic_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        log_likelihood = np.sum(np.log(probs[:, k]))
        return -log_likelihood  # minimize negative log-likelihood

    result = minimize_scalar(multinomial_log_likelihood, bounds=(0, 1),
                             args=(S, P, k), method='bounded')
    beta_hat = result.x
    # Standard error via numerical Hessian
```

For the PD, the payoff vectors used in beta estimation are:
```python
S = [r_coo * 400 + r_def * 0,   r_coo * 700 + r_def * 300]  # player payoffs for [coop, defect]
P = [r_coo * 400 + r_def * 700, r_coo * 0   + r_def * 300]  # partner payoffs for [coop, defect]
```

### 7.6 Statistical Testing

The `mu-test.ipynb` notebook uses the **Wilcoxon rank-sum test** (`scipy.stats.ranksums`) for comparing distributions between human and model behavior. This is a non-parametric test appropriate for comparing independent samples without assuming normal distributions.

---

## 8. Occupation Priming Experiments

The study also tests whether giving the model an occupational persona changes behavior:

```python
run_occupations(
    run_func=run_PD,
    extract_func=lambda x: x['choices'],
    n_instances=30,
    print_except=True,
)
```

The occupation prompt format is: `"You are a {occupation}."` prepended to the system message.

Record files for occupation variants are substantially larger (1-2 MB vs ~130 KB) due to running 30 instances for each of multiple occupations. These are saved as:
- `PD_gpt4_two_rounds_push_occupations_described_*.json`
- `PD_gpt4_two_rounds_pull_occupations_described_*.json`
- `PD_gpt4_five_rounds_pull_occupations_described_*.json`
- `PD_turbo_two_rounds_push_occupations_described_*.json`
- `PD_turbo_two_rounds_pull_occupations_described_*.json`
- `PD_turbo_five_rounds_occupations_described_*.json`

---

## 9. Key Findings Summary

| Finding | Detail |
|---------|--------|
| GPT-4 is strongly cooperative | 91.7% first-round cooperation vs 45.1% for humans |
| GPT-4 shows tit-for-tat behavior | Near-complete switch to defection after opponent defects (96.7% in round 2 of two-round pull variant) |
| GPT-3.5 is moderately cooperative | 76.7% first-round cooperation, but much noisier |
| GPT-3.5 is less strategic | Weaker reciprocation patterns; more stochastic behavior |
| Humans are mixed-strategy | ~45% cooperation, substantial heterogeneity |
| GPT models are more altruistic | Lower optimal `b` parameter in CES framework |
| Five-round games show adaptation | Both models adjust behavior based on opponent history, but GPT-4 does so more sharply |

---

## 10. Potential Issues and Observations

1. **Small sample size for models:** Only 30 instances per condition (vs. ~80,000 human data points). Statistical power for model comparisons is limited.

2. **Deterministic opponent strategy:** The opponent's moves are scripted (always Push or always Pull), not drawn from a population. This is a common experimental design choice but means the model never faces a truly adaptive opponent.

3. **Extraction reliability:** Using GPT-3.5-turbo to parse GPT-4's responses introduces a potential source of error. The validation step (requiring exact "Push" or "Pull") mitigates but does not eliminate this risk.

4. **Temperature not specified:** The notebooks do not explicitly set the temperature parameter for API calls, which defaults to 1.0. This explains some of the stochasticity in responses. Lower temperature could yield more deterministic behavior.

5. **Encoding mismatch between Push/Pull semantics:** In the game framing, "Push" is cooperative ($400 mutual) while "Pull" is selfish ($700 unilateral). However, the human data encodes `move=0` as Push and `move=1` as Pull with different cooperative semantics. The analysis code handles this, but the mapping requires careful attention.

6. **API model versions are dated:** The study uses `gpt-4-0314` and `gpt-3.5-turbo-0301` (March 2023 snapshots). Results may not generalize to newer model versions.

7. **Five-round opponent strategy is asymmetric:** The opponent plays Pull twice then Push twice (rounds 2-5). This tests punishment-then-forgiveness but not the reverse sequence, which could reveal different dynamics.
