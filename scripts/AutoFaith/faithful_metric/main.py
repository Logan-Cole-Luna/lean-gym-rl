from __future__ import annotations

import json
import sys
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from dataclasses import asdict

if __package__ in (None, ""):
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from general_judge_model import GeneralJudgeModel
        from judge_config import JudgeConfig, ModelProvider, APIKeys, GenerationConfig
        from scorer.checkpoints import NLBlock, FLBlock, FLTacticCategory
        from scorer.prompt import WHOLE_PROOF_FAITHFULNESS_PROMPT
        from FL_checkpoints import LeanREPL
        from NL_judge_model import JudgeModel
    except ImportError:
        from .general_judge_model import GeneralJudgeModel
        from .judge_config import JudgeConfig, ModelProvider, APIKeys, GenerationConfig
        from .scorer.checkpoints import NLBlock, FLBlock, FLTacticCategory
        from .scorer.prompt import WHOLE_PROOF_FAITHFULNESS_PROMPT
        from .FL_checkpoints import LeanREPL
        from .NL_judge_model import JudgeModel
else:
    from .general_judge_model import GeneralJudgeModel
    from .judge_config import JudgeConfig, ModelProvider, APIKeys, GenerationConfig
    from .scorer.checkpoints import NLBlock, FLBlock, FLTacticCategory
    from .scorer.prompt import WHOLE_PROOF_FAITHFULNESS_PROMPT
    from .FL_checkpoints import LeanREPL
    from .NL_judge_model import JudgeModel


class AlignmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nl_blocks: list[int]
    fl_blocks: list[int]
    match_score: float = Field(ge=0, le=1)
    reason: str


class WholeProofJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    NL_proof : str | None
    FL_proof : str | None
    score: float = Field(ge=0, le=1)
    strategy_match: bool
    nl_strategy: str
    fl_strategy: str
    alignment: list[AlignmentResult]
    unmatched_nl_blocks: list[int]
    unmatched_fl_blocks: list[int]
    summary: str

class WholeProofJudge:
    def __init__(self, config: JudgeConfig):
        self.judge_model = GeneralJudgeModel(config)

    def judge(self, nl_blocks: list[NLBlock], fl_blocks: list[FLBlock]) -> WholeProofJudgement:
        self._validate_nl(nl_blocks)
        self._validate_fl(fl_blocks)

        if not nl_blocks or not fl_blocks:
            return self._empty(nl_blocks, fl_blocks)

        result = self.judge_model.run_structured(
            self._build_prompt(nl_blocks, fl_blocks),
            WholeProofJudgement,
            "whole_proof_judgement",
        )

        self._validate_result(result, nl_blocks, fl_blocks)
        return result

    def judge_and_save(self, nl_blocks: list[NLBlock], fl_blocks: list[FLBlock], output_path: Path) -> WholeProofJudgement:
        result = self.judge(nl_blocks, fl_blocks)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _build_prompt(nl_blocks: list[NLBlock], fl_blocks: list[FLBlock]) -> str:
        nl = json.dumps([b.to_dict() for b in nl_blocks], indent=2, ensure_ascii=False)
        fl = json.dumps([b.to_dict() for b in fl_blocks], indent=2, ensure_ascii=False)
        return WHOLE_PROOF_FAITHFULNESS_PROMPT.replace("{NL_BLOCKS}", nl).replace("{FL_BLOCKS}", fl)

    @staticmethod
    def _validate_nl(blocks: list[NLBlock]) -> None:
        for i in range(len(blocks) - 1):
            if blocks[i].next_checkpoint != blocks[i + 1].previous_checkpoint:
                raise ValueError(f"Inconsistent NL blocks between {i} and {i + 1}.")

    @classmethod
    def _validate_fl(cls, blocks: list[FLBlock]) -> None:
        for block in blocks:
            if block.kind == FLTacticCategory.LEAF and block.children:
                raise ValueError("LEAF FLBlock cannot contain children.")
            if block.kind == FLTacticCategory.COMPOUND and not block.children:
                raise ValueError("COMPOUND FLBlock must contain children.")
            cls._validate_fl(block.children)

    @staticmethod
    def _validate_result(result: WholeProofJudgement, nl_blocks: list[NLBlock], fl_blocks: list[FLBlock]) -> None:
        n, m = len(nl_blocks), len(fl_blocks)

        for a in result.alignment:
            if any(i < 0 or i >= n for i in a.nl_blocks):
                raise ValueError("Invalid NL block index.")
            if any(i < 0 or i >= m for i in a.fl_blocks):
                raise ValueError("Invalid FL block index.")

        if any(i < 0 or i >= n for i in result.unmatched_nl_blocks):
            raise ValueError("Invalid unmatched NL block index.")

        if any(i < 0 or i >= m for i in result.unmatched_fl_blocks):
            raise ValueError("Invalid unmatched FL block index.")

    @staticmethod
    def _empty(nl_blocks: list[NLBlock], fl_blocks: list[FLBlock]) -> WholeProofJudgement:
        return WholeProofJudgement(
            NL_proof = None if not nl_blocks else "NL proof exists but FL proof is empty.",
            FL_proof = None if not fl_blocks else "FL proof exists but NL proof is empty.",
            score=0.0,
            strategy_match=False,
            nl_strategy="",
            fl_strategy="",
            alignment=[],
            unmatched_nl_blocks=list(range(len(nl_blocks))),
            unmatched_fl_blocks=list(range(len(fl_blocks))),
            summary="One or both proof block lists are empty.",
        )

if __name__ == "__main__":
    # Example usage
    nl_blocks = []  # Replace with actual NLBlock instances
    fl_blocks = []  # Replace with actual FLBlock instances

    config = JudgeConfig(
        model_name="gpt-5.2",
        provider=ModelProvider.OPENAI,
        api_keys=APIKeys(openai_api_key=""),
        generation=GenerationConfig(
            temperature=0.0,
            max_tokens=4096,
            top_p=1.0,
        ),
    )

    theorem_statement = r"""
    \begin{theorem}
    Let $a,b \in \mathbb{Z}$. If $a$ and $b$ are even, then $a+b$ is even.
    \end{theorem}
    """

    natural_language_proof = r"""
    \begin{proof}
    Since $a$ is even, there exists an integer $k$ such that
    \[
    a = 2k.
    \]
    Similarly, since $b$ is even, there exists an integer $\ell$ such that
    \[
    b = 2\ell.
    \]
    Therefore,
    \[
    a+b = 2k + 2\ell = 2(k+\ell).
    \]
    Because $k+\ell$ is an integer, $a+b$ is divisible by $2$.
    Hence $a+b$ is even.
    \end{proof}
    """    

    

    judge = WholeProofJudge(config)
    NL_judge = JudgeModel(config)
    nl_blocks = NL_judge.generate_nl_blocks(
            theorem_statement,
            natural_language_proof,
    )

    project_path = Path("/Users/vietbachhoang/AutoFaith/faithful_metric/FLproof")
    proof_file = project_path / "FLProof/proof.lean"
    output_file = project_path / "fl_blocks.json"

    repl = LeanREPL(project_path)

    try:
        fl_blocks = repl.extract_fl_blocks(proof_file)

        for block in fl_blocks:
            print(block)

        repl.save_json(fl_blocks, output_file)
        print(f"\nSaved to {output_file}")

    finally:
        repl.close()


    result = judge.judge_and_save(nl_blocks, fl_blocks, Path("whole_proof_judgement.json"))
    print(result.score)


# def judge_nl_blocks_fl_blocks_dynamic(nl_blocks: list[NLBlock], fl_blocks: list[FLBlock], num_of_checkpoints: int = 0) -> float:
#     """
#     Judge the faithfulness of the natural language proof blocks against the formal proof blocks.

#     Args:
#         nl_blocks (list[NLBlock]): The natural language proof blocks.
#         fl_blocks (list[FLBlock]): The formal proof blocks.

#     Returns:
#         score ([0, 1]): A score indicating the faithfulness of the natural language blocks to the formal proof blocks.
#     """
#     judge._validate_transition_consistency(nl_blocks)
#     nl_blocks_len = len(nl_blocks)
#     fl_blocks_len = len(fl_blocks)

#     if nl_blocks_len == 0 or fl_blocks_len == 0:
#         return 0




    

