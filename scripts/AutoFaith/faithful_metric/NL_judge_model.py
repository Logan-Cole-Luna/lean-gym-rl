from __future__ import annotations

import sys

from pathlib import Path

from pydantic import BaseModel, ConfigDict


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from general_judge_model import GeneralJudgeModel
    from judge_config import JudgeConfig
    from scorer.checkpoints import Checkpoint, NLBlock, NLReasoningCategory
    from scorer.prompt import NL_CHECKPOINT_GENERATION_PROMPT
else:
    from .general_judge_model import GeneralJudgeModel
    from .judge_config import JudgeConfig
    from .scorer.checkpoints import Checkpoint, NLBlock, NLReasoningCategory
    from .scorer.prompt import NL_CHECKPOINT_GENERATION_PROMPT


# ============================================================
# Output classes -- UNCHANGED
# ============================================================

class CheckpointOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    premises: list[str]
    goal: list[str]


class NLBlockOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning_category: NLReasoningCategory
    previous_checkpoint: CheckpointOutput
    arguments: list[str]
    next_checkpoint: CheckpointOutput


class NLBlocksOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocks: list[NLBlockOutput]


# ============================================================
# NL-specific Judge Model
# ============================================================

class JudgeModel(GeneralJudgeModel):

    def __init__(self, config: JudgeConfig):
        super().__init__(config)

    def generate_nl_blocks(
        self,
        theorem_statement: str,
        natural_language_proof: str,
        feedback: str | None = None,
    ) -> list[NLBlock]:

        prompt = self._build_prompt(theorem_statement, natural_language_proof)

        if feedback is not None:
            prompt += f"\n\nFeedback: {feedback}"

        output = self.run_structured(
            prompt=prompt,
            output_model=NLBlocksOutput,
            schema_name="nl_proof_blocks",
        )

        blocks = self._convert_to_nl_blocks(output)
        self._validate_transition_consistency(blocks)

        return blocks

    @staticmethod
    def _build_prompt(theorem_statement: str, natural_language_proof: str) -> str:
        return NL_CHECKPOINT_GENERATION_PROMPT.replace(
            "{THEOREM_STATEMENT}",
            theorem_statement,
        ).replace(
            "{NATURAL_LANGUAGE_PROOF}",
            natural_language_proof,
        )

    @staticmethod
    def _convert_checkpoint(checkpoint: CheckpointOutput) -> Checkpoint:
        return Checkpoint(
            premises=checkpoint.premises,
            goal=checkpoint.goal,
        )

    @classmethod
    def _convert_to_nl_blocks(cls, output: NLBlocksOutput) -> list[NLBlock]:
        return [
            NLBlock(
                previous_checkpoint=cls._convert_checkpoint(block.previous_checkpoint),
                arguments=block.arguments,
                next_checkpoint=cls._convert_checkpoint(block.next_checkpoint),
                reasoning_category=block.reasoning_category,
            )
            for block in output.blocks
        ]

    @staticmethod
    def _validate_transition_consistency(blocks: list[NLBlock]) -> None:
        for i in range(len(blocks) - 1):
            if blocks[i].next_checkpoint != blocks[i + 1].previous_checkpoint:
                raise ValueError(
                    f"Inconsistent proof blocks: "
                    f"block {i}.next_checkpoint != "
                    f"block {i + 1}.previous_checkpoint"
                )

    @staticmethod
    def validate_nl_blocks(blocks: list[NLBlock]) -> tuple[str | None, bool]:
        for i in range(len(blocks) - 1):
            if blocks[i].next_checkpoint != blocks[i + 1].previous_checkpoint:
                feedback = (
                    f"Inconsistent checkpoints between block {i} and block {i + 1}:\n"
                    f"next_checkpoint = {blocks[i].next_checkpoint}\n"
                    f"previous_checkpoint = {blocks[i + 1].previous_checkpoint}"
                )

                return feedback, False

        return None, True