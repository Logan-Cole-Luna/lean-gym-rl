"""The reward subcomponents, one module per term.

A TERM IS NOT AN ARM. Each module here answers one question about a rollout and
turns it into a number. Nothing here decides an OUTCOME -- that is
`reward/reward.py` -- and nothing here decides which terms a run pays for --
that is the table in `reward/arms.py`.

Two shapes sit side by side: `brevity` and `faithfulness` adjust the top row,
while `gated` and `typecheck` replace the whole column of values. Both are
things an arm composes, which is why they share a package.
"""
