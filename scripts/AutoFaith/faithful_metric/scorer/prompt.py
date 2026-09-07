NL_CHECKPOINT_GENERATION_PROMPT = """
You are a mathematical proof-state analyzer.

Your task is to analyze a natural-language mathematical proof and decompose it into a sequence of natural-language proof blocks.

Each proof block represents one meaningful transition in the mathematical reasoning.

A proof block has exactly four fields:

1. `reasoning_category`
2. `previous_checkpoint`
3. `arguments`
4. `next_checkpoint`

The intended Python structure is:

```python
class NLReasoningCategory(enum.Enum):
    INTRODUCE_OBJECT = "introduce_object"
    INTRODUCE_ASSUMPTION = "introduce_assumption"
    UNPACK_DEFINITION = "unpack_definition"
    APPLY_THEOREM = "apply_theorem"
    DERIVE_FACT = "derive_fact"
    REWRITE = "rewrite"
    CALCULATION = "calculation"
    REDUCE_GOAL = "reduce_goal"
    CASE_SPLIT = "case_split"
    INDUCTION = "induction"
    CONTRADICTION = "contradiction"
    CONTRAPOSITIVE = "contrapositive"
    CONCLUDE = "conclude"
    OTHER = "other"


@dataclass(frozen=True)
class Checkpoint:
    premises: list[str]
    goal: list[str]


@dataclass(frozen=True)
class NLBlock(Block):
    reasoning_category: NLReasoningCategory
```

where `Block` contains:

```python
@dataclass(frozen=True)
class Block:
    previous_checkpoint: Checkpoint
    arguments: list[str]
    next_checkpoint: Checkpoint
```

## Meaning of a proof block

Each block represents:

previous_checkpoint
-> reasoning_category + arguments
-> next_checkpoint

`previous_checkpoint` describes the proof state immediately BEFORE the reasoning is performed.

`reasoning_category` describes the primary type of mathematical reasoning performed in this transition.

`arguments` describes the natural-language mathematical reasoning actually performed by the author.

`next_checkpoint` describes the proof state immediately AFTER the reasoning has been performed.

## Reasoning categories

For every block, choose exactly ONE category from the following list.

### `INTRODUCE_OBJECT`

Use when the proof introduces a new mathematical object, variable, witness, function, set, constant, index, or similar object.

Example:

"Since a is even, there exists k ∈ ℤ such that a = 2k."

If the main effect of the step is introducing `k`, classify the block as:

`INTRODUCE_OBJECT`

### `INTRODUCE_ASSUMPTION`

Use when the proof introduces a temporary hypothesis or assumption.

Examples include:

* assuming the antecedent of an implication,
* assuming something for contradiction,
* introducing a hypothesis within a case.

### `UNPACK_DEFINITION`

Use when the proof explicitly uses, expands, or interprets a mathematical definition.

Example:

"Since a is even, by definition there exists k ∈ ℤ such that a = 2k."

Use this category when the primary reasoning step is invoking the meaning of a definition rather than merely introducing the resulting object.

### `APPLY_THEOREM`

Use when a theorem, lemma, proposition, previously established result, or standard named mathematical result is applied.

Example:

"By the Intermediate Value Theorem, there exists c ∈ (a,b) such that f(c) = 0."

### `DERIVE_FACT`

Use when a new intermediate mathematical fact is logically deduced from the currently available premises and no more specific category applies.

Example:

"Since x > 2 and y > 3, we have x + y > 5."

### `REWRITE`

Use when an expression is replaced using an equality, equivalence, substitution, or previously established identity.

Example:

"Substituting a = 2k and b = 2l, we obtain a + b = 2k + 2l."

### `CALCULATION`

Use for arithmetic, algebraic manipulation, simplification, factoring, expansion, rearrangement, or an explicit chain of calculations.

Example:

"2k + 2l = 2(k + l)."

### `REDUCE_GOAL`

Use when the proof changes the current objective to one or more intermediate goals whose proof would establish the original goal.

Example:

"It therefore suffices to prove that f is continuous."

### `CASE_SPLIT`

Use when the proof divides the argument into separate cases.

Example:

"We consider separately the cases x ≥ 0 and x < 0."

### `INDUCTION`

Use when the proof begins or performs mathematical induction, including the base case or inductive step when that is the primary reasoning structure.

### `CONTRADICTION`

Use when the proof assumes the negation of a desired statement or derives an impossibility in order to establish the goal.

### `CONTRAPOSITIVE`

Use when the proof explicitly changes the argument to proving the contrapositive.

### `CONCLUDE`

Use when previously established facts are used to directly discharge the current goal without introducing substantial new reasoning.

Example:

"Since k + l ∈ ℤ, it follows that a + b is even."

### `OTHER`

Use only when none of the categories above appropriately describes the main reasoning step.

## Choosing the reasoning category

Choose the category that describes the PRIMARY mathematical action of the block.

A sentence may perform multiple reasoning actions.

If those actions correspond to distinct proof-state transitions, split them into separate blocks.

For example:

"Since a is even, write a = 2k. Substituting this into the expression gives a + b = 2k + b."

should normally become two blocks:

1. `INTRODUCE_OBJECT`
2. `REWRITE`

Do not assign multiple categories to one block.

If several actions occur but they jointly constitute one indivisible mathematical transition, choose the category that best describes the main effect on the proof state.

## Checkpoint structure

Each checkpoint has exactly two fields:

### `premises`

A list of mathematical facts that are currently available.

These may include:

* variables and objects from the theorem statement,
* assumptions and hypotheses,
* previously introduced witnesses or mathematical objects,
* intermediate facts already established,
* properties of introduced objects,
* previously proved statements that remain relevant.

### `goal`

A list of mathematical statements that currently remain to be proved.

Usually there is one goal:

```json
{
  "goal": [
    "a + b is even"
  ]
}
```

If multiple simultaneous subgoals exist, include all of them.

## Arguments

`arguments` is a list of natural-language mathematical reasoning steps that transform `previous_checkpoint` into `next_checkpoint`.

Each argument should correspond to reasoning actually contained in the supplied proof.

Preserve the mathematical content and intent of the author's wording.

Do not replace the author's argument with another proof.

Do not repair the proof by inventing missing reasoning.

Do not introduce unstated lemmas, assumptions, or mathematical facts unless they are clearly implicit in the provided proof.

## Choosing proof blocks

You must determine the appropriate granularity of each proof block.

A block should correspond to one meaningful mathematical reasoning transition.

A block may correspond to:

* part of one sentence,
* one complete sentence,
* multiple sentences,

depending on the mathematical reasoning.

Do not split the proof merely according to sentence boundaries.

If one sentence performs multiple distinct mathematical transitions, split it into multiple blocks.

If several sentences together perform one mathematical transition, they may belong to one block.

The purpose of segmentation is to identify meaningful changes in proof state, not grammatical sentence boundaries.

## Transition consistency

The checkpoints between consecutive blocks must be consistent.

For consecutive blocks:

Block i:

previous_checkpoint = C_i
reasoning_category = R_i
arguments = A_i
next_checkpoint = C_{i+1}

Block i+1:

previous_checkpoint = C_{i+1}
reasoning_category = R_{i+1}
arguments = A_{i+1}
next_checkpoint = C_{i+2}

Therefore:

`blocks[i].next_checkpoint` must be mathematically equivalent to `blocks[i+1].previous_checkpoint`.

Whenever possible, use exactly the same wording and notation for identical facts across adjacent checkpoints.

## Initial checkpoint

The `previous_checkpoint` of the first block must represent the initial proof state obtained directly from the theorem statement.

Its:

* `premises` should contain the theorem's assumptions and relevant mathematical objects;
* `goal` should contain the theorem's conclusion.

## Final checkpoint

The `next_checkpoint` of the final block should represent the proof state after the final argument has been performed.

If the proof has completely established the theorem, use:

```json
{
  "premises": [
    "..."
  ],
  "goal": []
}
```

An empty `goal` list means that no goals remain and the proof is complete.

Do not retain the proved theorem as an unfinished goal.

## Updating premises

If an argument introduces a new object or establishes a new fact, that information should appear in the `next_checkpoint`.

For example:

Previous checkpoint:

```json
{
  "premises": [
    "a is even"
  ],
  "goal": [
    "a + b is even"
  ]
}
```

Reasoning category:

`INTRODUCE_OBJECT`

Argument:

```json
[
  "Since a is even, there exists k ∈ ℤ such that a = 2k."
]
```

Next checkpoint:

```json
{
  "premises": [
    "a is even",
    "k ∈ ℤ",
    "a = 2k"
  ],
  "goal": [
    "a + b is even"
  ]
}
```

## Goal transitions

The goal does not need to change after every argument.

However, if the proof explicitly:

* reduces the current goal to a subgoal,
* introduces a new subgoal,
* completes a subgoal,
* performs case analysis,
* begins an induction,
* changes to contradiction or contrapositive reasoning,
* otherwise changes what is currently being proved,

reflect that change in `next_checkpoint.goal`.

## Faithfulness requirements

Preserve the author's actual proof trajectory.

Do not:

* simplify the proof into a shorter proof,
* introduce mathematically valid but unstated reasoning,
* reorder the author's argument,
* silently fill substantial gaps,
* replace one strategy with another strategy,
* classify the reasoning according to a different proof that could have been used.

The reasoning category must describe what the supplied proof actually does.

The purpose is to reconstruct the proof states and reasoning actions induced by the supplied natural-language proof, not to produce the best proof.

## Output format

Return ONLY valid JSON.

The JSON must have exactly this structure:

```json
{
  "blocks": [
    {
      "reasoning_category": "introduce_object",
      "previous_checkpoint": {
        "premises": [
          "...",
          "..."
        ],
        "goal": [
          "..."
        ]
      },
      "arguments": [
        "..."
      ],
      "next_checkpoint": {
        "premises": [
          "...",
          "...",
          "..."
        ],
        "goal": [
          "..."
        ]
      }
    }
  ]
}
```

The value of `reasoning_category` must be exactly one of:

```text
introduce_object
introduce_assumption
unpack_definition
apply_theorem
derive_fact
rewrite
calculation
reduce_goal
case_split
induction
contradiction
contrapositive
conclude
other
```

Do not include:

* markdown outside the JSON,
* comments,
* explanations,
* `step`,
* `proof_text`,
* fields other than `reasoning_category`, `previous_checkpoint`, `arguments`, and `next_checkpoint` inside each block.

Every block must be directly convertible into:

```python
NLBlock(
    previous_checkpoint=Checkpoint(...),
    arguments=[...],
    next_checkpoint=Checkpoint(...),
    reasoning_category=NLReasoningCategory(...)
)
```

## Input

### Theorem statement

{THEOREM_STATEMENT}

### Natural-language proof

{NATURAL_LANGUAGE_PROOF}
"""


WHOLE_PROOF_FAITHFULNESS_PROMPT = """
You are a mathematical proof faithfulness judge.

Your task is to evaluate whether a natural-language proof follows the same mathematical reasoning as a formal Lean proof.

You are given:

1. A sequence of natural-language proof blocks (`NLBlock`).
2. A sequence of formal Lean proof blocks (`FLBlock`).

Your goal is to judge the faithfulness of the NATURAL-LANGUAGE proof with respect to the FORMAL proof.

Return a single faithfulness score between 0 and 1, where:

* 1.0 means the natural-language proof faithfully represents the mathematical reasoning of the formal proof.
* 0.0 means the two proofs follow fundamentally different reasoning or the natural-language proof is incompatible with the formal proof.
* Intermediate values indicate partial correspondence.

The comparison must be based on the WHOLE proof trajectory, not merely individual block similarity.

# Proof Block Semantics

A natural-language block has the form:

NLBlock(
previous_checkpoint,
reasoning_category,
arguments,
next_checkpoint
)

and represents:

previous_checkpoint
-> natural-language reasoning
-> next_checkpoint

A formal block has the form:

FLBlock(
kind,
block,
children
)

where `block` contains:

Block(
previous_checkpoint,
arguments,
next_checkpoint
)

and `kind` is either:

* `leaf`
* `compound`

A leaf FL block represents one formal tactic transition.

A compound FL block represents one high-level Lean tactic that contains internal reasoning steps.

For example:

calc
├── rw [hk, hl]
└── ring

The compound `calc` block represents the high-level transition, while its children represent the lower-level reasoning used to realize that transition.

# Checkpoint Semantics

Each checkpoint contains:

* `premises`: facts currently available.
* `goal`: mathematical statements currently being proved.

The exact wording of NL and FL checkpoints may differ.

Do NOT require textual equality.

Instead, compare their MATHEMATICAL MEANING.

Examples of equivalent information include:

* "a is even"
* "2 ∣ a"

and:

* "there exists k ∈ ℤ such that a = 2k"
* a Lean context containing `k : ℤ` and `hk : a = 2 * k`

# Main Evaluation Objective

Determine whether the natural-language proof describes essentially the SAME PROOF PATH as the Lean proof.

A faithful proof should preserve:

1. the major mathematical strategy,
2. the order of important reasoning steps,
3. the introduction of important mathematical objects,
4. the use of intermediate facts,
5. the progression of goals,
6. the mathematical dependencies between steps,
7. the final conclusion.

Do NOT demand one-to-one correspondence between NL blocks and FL blocks.

One natural-language block may correspond to:

* one FL block,
* several consecutive FL blocks,
* one compound FL block,
* several children inside one compound FL block.

Similarly, several natural-language blocks may correspond to one formal block when the formal tactic compresses multiple mathematical ideas.

# Hierarchical FL Blocks

When evaluating a compound FL block, consider BOTH:

1. the high-level meaning of the compound block;
2. the detailed reasoning contained in its children.

For example, if the NL proof says:

"a + b = 2k + 2l = 2(k+l)"

and the FL proof contains:

calc
├── rw [hk, hl]
└── ring

then the NL statement may faithfully correspond to the entire compound `calc` block even though it does not explicitly mention `rw` or `ring`.

Do not penalize the natural-language proof merely because it describes a compound formal argument at a higher level of abstraction.

# Alignment

Construct an order-preserving alignment between the NL blocks and FL blocks.

The alignment must preserve proof order.

If:

NL_i corresponds to FL_j

and:

NL_k corresponds to FL_l

with i < k,

then normally j <= l.

Do not create an alignment that arbitrarily reorders the proof.

Each NL block may align with one or more FL blocks.

Each FL block may align with one or more NL blocks when mathematically appropriate.

A block may remain unmatched if it represents:

* implementation detail,
* syntactic bookkeeping,
* a mathematically meaningful step missing from the other proof,
* genuinely different reasoning.

# What Counts as Faithful

The following differences SHOULD generally be allowed:

* natural language omits trivial algebraic details;
* Lean uses implementation-level tactics not mentioned explicitly;
* natural language groups several Lean tactics into one mathematical step;
* Lean separates one natural-language step into several subgoals;
* notation differs while mathematical meaning remains equivalent;
* definitional equivalence is used;
* obvious type information is explicit in Lean but implicit in natural language.

The following differences SHOULD reduce faithfulness:

* a theorem or fact is used in one proof but not represented by the other when it is essential to the argument;
* a key witness or construction is different;
* the proofs use substantially different intermediate results;
* reasoning occurs in a fundamentally different order;
* one proof uses a different major strategy;
* the natural-language proof claims an inference not supported by the formal proof;
* an essential formal reasoning step has no reasonable natural-language counterpart;
* the natural-language proof introduces substantial reasoning absent from the formal proof.

# Major Strategy

Pay particular attention to proof strategy.

Examples include:

* direct proof,
* contradiction,
* contrapositive,
* induction,
* case analysis,
* witness construction,
* calculation,
* application of a major theorem.

If the proofs use fundamentally incompatible high-level strategies, the faithfulness score should be low even if they prove the same theorem.

Proving the same statement is NOT sufficient for faithfulness.

# Scoring Guidance

Use the following approximate interpretation:

0.90 - 1.00
The NL proof closely follows the same reasoning trajectory as the FL proof. Differences are mainly abstraction level, notation, or minor omitted implementation details.

0.75 - 0.90
The major strategy and most intermediate reasoning agree, with some omissions, compression, or small differences.

0.50 - 0.75
The proofs share substantial reasoning but contain notable differences in intermediate steps, ordering, or justification.

0.25 - 0.50
The proofs have limited structural correspondence. They may share the same theorem or a few ideas but follow substantially different reasoning.

0.00 - 0.25
The proofs are incompatible, use fundamentally different strategies, or the natural-language proof does not meaningfully describe the formal proof.

Do not assign 1.0 merely because both proofs are mathematically correct.

The score measures FAITHFULNESS between the proofs, not correctness in isolation.

# Required Analysis Procedure

Perform the following internally:

1. Identify the high-level strategy of the NL proof.
2. Identify the high-level strategy of the FL proof.
3. Compare the initial proof states.
4. Construct an order-preserving alignment between NL and FL blocks.
5. Compare checkpoint transitions under this alignment.
6. Compare the mathematical meaning of the arguments.
7. Inspect children of compound FL blocks when necessary.
8. Identify important unmatched reasoning.
9. Determine whether differences are merely differences in abstraction or genuinely different proof reasoning.
10. Produce the final faithfulness score.

# Output Format

Return ONLY valid JSON.

Use exactly this structure:

{
"score": 0.0,
"strategy_match": true,
"nl_strategy": "...",
"fl_strategy": "...",
"alignment": [
{
"nl_blocks": [0],
"fl_blocks": [0],
"match_score": 1.0,
"reason": "..."
},
{
"nl_blocks": [1],
"fl_blocks": [1, 2],
"match_score": 0.9,
"reason": "..."
}
],
"unmatched_nl_blocks": [],
"unmatched_fl_blocks": [],
"summary": "..."
}

## Field meanings

`score`
Overall proof faithfulness score in [0, 1].

`strategy_match`
Whether the two proofs use compatible overall proof strategies.

`nl_strategy`
Short description of the main natural-language proof strategy.

`fl_strategy`
Short description of the main formal proof strategy.

`alignment`
An order-preserving alignment between groups of NL and FL blocks.

`nl_blocks`
Zero-based indices of NL blocks participating in this alignment group.

`fl_blocks`
Zero-based indices of TOP-LEVEL FL blocks participating in this alignment group.

If an FL block is compound, inspect its children internally when judging the match, but identify the compound block by its top-level index here.

`match_score`
Local semantic faithfulness score for this aligned group.

`reason`
Concise mathematical explanation of why these blocks correspond.

`unmatched_nl_blocks`
Zero-based indices of NL blocks without a reasonable formal counterpart.

`unmatched_fl_blocks`
Zero-based indices of top-level FL blocks without a reasonable natural-language counterpart.

`summary`
Concise explanation of the most important factors determining the overall score.

Do not include markdown, comments, or additional text outside the JSON.

# Input

## Natural-language proof blocks

{NL_BLOCKS}

## Formal Lean proof blocks

{FL_BLOCKS}
"""

