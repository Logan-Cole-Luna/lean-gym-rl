import enum
import json
import subprocess

from dataclasses import dataclass, field
from pathlib import Path

try:
    from .scorer.checkpoints import Checkpoint, FLTacticCategory, FLBlock, Block
except ImportError:
    from scorer.checkpoints import Checkpoint, FLTacticCategory, FLBlock, Block

@dataclass
class RawTactic:
    tactic: str
    proof_state: int
    goals: str | list[str]
    line: int
    column: int
    end_line: int
    end_column: int
    children: list["RawTactic"] = field(default_factory=list)

    @property
    def start(self) -> tuple[int, int]:
        return self.line, self.column

    @property
    def end(self) -> tuple[int, int]:
        return self.end_line, self.end_column


class LeanREPL:
    def __init__(self, project_path: Path):
        self.project_path = project_path.resolve()
        self.process = subprocess.Popen(
            ["lake", "exe", "repl"],
            cwd=self.project_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def send(self, request: dict) -> dict:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Lean REPL is not running.")

        self.process.stdin.write(json.dumps(request) + "\n\n")
        self.process.stdin.flush()

        response = ""

        while True:
            line = self.process.stdout.readline()

            if line == "":
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"Lean REPL terminated unexpectedly.\n{stderr}")

            response += line

            try:
                return json.loads(response)
            except json.JSONDecodeError:
                continue

    def extract_fl_blocks(self, file_path: Path) -> list[FLBlock]:
        relative_path = file_path.resolve().relative_to(self.project_path)
        file_result = self.send({"path": str(relative_path), "allTactics": True})

        if "tactics" not in file_result:
            raise RuntimeError(f"No tactics returned by REPL:\n{file_result}")

        raw_tactics = self._parse_raw_tactics(file_result["tactics"])
        roots = self._build_tactic_tree(raw_tactics)

        result = []

        for root in sorted(roots, key=lambda x: x.start):
            result.extend(self._convert_raw_tactic(root))

        return result

    @staticmethod
    def _parse_raw_tactics(tactics: list[dict]) -> list[RawTactic]:
        return [
            RawTactic(
                tactic=t["tactic"],
                proof_state=t["proofState"],
                goals=t["goals"],
                line=t["pos"]["line"],
                column=t["pos"]["column"],
                end_line=t["endPos"]["line"],
                end_column=t["endPos"]["column"],
            )
            for t in tactics
        ]

    @staticmethod
    def _contains(parent: RawTactic, child: RawTactic) -> bool:
        if parent.start == child.start and parent.end == child.end:
            return False
        return parent.start <= child.start and child.end <= parent.end

    @classmethod
    def _build_tactic_tree(cls, tactics: list[RawTactic]) -> list[RawTactic]:
        roots = []

        for child in tactics:
            parents = [p for p in tactics if p is not child and cls._contains(p, child)]

            if not parents:
                roots.append(child)
                continue

            parent = min(parents, key=cls._span_size)
            parent.children.append(child)

        for tactic in tactics:
            tactic.children.sort(key=lambda x: x.start)

        return roots

    @staticmethod
    def _span_size(tactic: RawTactic) -> tuple[int, int]:
        return tactic.end_line - tactic.line, tactic.end_column - tactic.column

    @staticmethod
    def _is_internal(tactic: RawTactic) -> bool:
        return not tactic.tactic.strip() or tactic.goals == "no goals"

    def _convert_raw_tactic(self, tactic: RawTactic) -> list[FLBlock]:
        children = []
        for child in sorted(tactic.children, key=lambda x: x.start):
            children.extend(self._convert_raw_tactic(child))
        if self._is_internal(tactic):
            return children
        before = self._goals_to_checkpoint(tactic.goals)
        after_result = self.send({"tactic": tactic.tactic, "proofState": tactic.proof_state})
        after = self._goals_to_checkpoint(after_result.get("goals", []))
        block = Block(
            previous_checkpoint=before,
            arguments=[tactic.tactic.strip()],
            next_checkpoint=after,
        )
        kind = FLTacticCategory.COMPOUND if children else FLTacticCategory.LEAF
        return [FLBlock(kind=kind, block=block, children=children)]

    def _goals_to_checkpoint(self, goals: str | list[str]) -> Checkpoint:
        if not goals or goals == "no goals":
            return Checkpoint([], [])
        if isinstance(goals, str):
            goals = [goals]
        premises, targets = [], []
        for state in goals:
            current_premises, target = self._parse_single_goal(state)
            for premise in current_premises:
                if premise not in premises:
                    premises.append(premise)
            if target:
                targets.append(target)
        return Checkpoint(premises, targets)

    @staticmethod
    def _parse_single_goal(state: str) -> tuple[list[str], str]:
        state = state.strip()
        if state == "no goals":
            return [], []
        if "⊢" not in state:
            return [], state
        context, target = state.rsplit("⊢", 1)
        premises = [
            line.strip()
            for line in context.splitlines()
            if line.strip() and not line.strip().startswith("case ")
        ]

        return premises, target.strip()

    @staticmethod
    def save_json(blocks: list[FLBlock], output_path: Path) -> None:
        data = {"blocks": [block.to_dict() for block in blocks]}
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


def main() -> None:
    project_path = Path("/Users/vietbachhoang/AutoFaith/faithful_metric/FLproof")
    proof_file = project_path / "FLProof/proof.lean"
    output_file = project_path / "fl_blocks.json"

    repl = LeanREPL(project_path)

    try:
        blocks = repl.extract_fl_blocks(proof_file)

        for block in blocks:
            print(block)

        repl.save_json(blocks, output_file)
        print(f"\nSaved to {output_file}")

    finally:
        repl.close()


if __name__ == "__main__":
    main()