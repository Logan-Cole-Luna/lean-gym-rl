
'''
This module will define the Checkpoint and Proof Block classes, which will be used to store the state of the proof at various points in time. 
The Checkpoint class will store the premises and goal of the proof, while the Proof Block class will store the strategy and arguments used to reach that point in the proof.
'''

from dataclasses import dataclass
import enum
from typing import List
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Checkpoint:
    premises : list[str]
    goal : list[str]

    def __str__(self) -> str:
            premises = "\n".join(f"    {p}" for p in self.premises) or "    <none>"
            goals = "\n".join(f"    ⊢ {g}" for g in self.goal) or "    <no goals>"
            return f"Premises:\n{premises}\nGoals:\n{goals}"
   

@dataclass(frozen=True)
class Block:
    previous_checkpoint: Checkpoint
    arguments: list[str]
    next_checkpoint: Checkpoint

    def __str__(self) -> str:
        args = "\n".join(self.arguments)
        return f"[Before]\n{self.previous_checkpoint}\n\n[Tactic]\n{args}\n\n[After]\n{self.next_checkpoint}"

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
class NLBlock (Block):
    reasoning_category: NLReasoningCategory | None = None
    def to_dict(self) -> dict:
        return {
            "previous_checkpoint": {
                "premises": self.previous_checkpoint.premises,
                "goal": self.previous_checkpoint.goal,
            },
            "arguments": self.arguments,
            "next_checkpoint": {
                "premises": self.next_checkpoint.premises,
                "goal": self.next_checkpoint.goal,
            },
            "reasoning_category": (
                self.reasoning_category.value
                if self.reasoning_category is not None
                else None
            ),
        }


class FLTacticCategory(enum.Enum):
    LEAF = "leaf"
    COMPOUND = "compound"


@dataclass(frozen=True)
class FLBlock:
    kind: FLTacticCategory
    block: Block
    children: list["FLBlock"] = field(default_factory=list)

    def __post_init__(self):
        if self.kind == FLTacticCategory.LEAF and self.children:
            raise ValueError("A LEAF block cannot have children.")
        if self.kind == FLTacticCategory.COMPOUND and not self.children:
            raise ValueError("A COMPOUND block must have children.")

    def __str__(self) -> str:
        return self._str_with_depth(0)

    def _str_with_depth(self, depth: int) -> str:
        indent = "    " * depth
        tactic = self.block.arguments[0].strip()
        header = f"{indent}[{self.kind.value.upper()}] {tactic}"

        if not self.children:
            return header

        children = "\n".join(child._str_with_depth(depth + 1) for child in self.children)
        return f"{header}\n{children}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "block": {
                "previous_checkpoint": {
                    "premises": self.block.previous_checkpoint.premises,
                    "goal": self.block.previous_checkpoint.goal,
                },
                "arguments": self.block.arguments,
                "next_checkpoint": {
                    "premises": self.block.next_checkpoint.premises,
                    "goal": self.block.next_checkpoint.goal,
                },
            },
            "children": [child.to_dict() for child in self.children],
        }


