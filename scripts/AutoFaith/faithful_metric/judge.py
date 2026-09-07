import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from judge_config import JudgeConfig, ModelProvider, APIKeys, GenerationConfig
    from NL_judge_model import JudgeModel
else:
    from .judge_config import JudgeConfig, ModelProvider, APIKeys, GenerationConfig
    from .NL_judge_model import JudgeModel


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

judge = JudgeModel(config)


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


blocks = judge.generate_nl_blocks(
    theorem_statement,
    natural_language_proof,
)

while judge.validate_nl_blocks(blocks)[1] == False:
    print("Invalid NL blocks detected. Regenerating...")
    blocks = judge.generate_nl_blocks(
        theorem_statement,
        natural_language_proof,
        feedback = "In this version\n" + blocks + "Here is the feedback\n" +  judge.validate_nl_blocks(blocks)[0]
    )


def serialize(obj):
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {key: serialize(value) for key, value in asdict(obj).items()}
    if isinstance(obj, list):
        return [serialize(value) for value in obj]
    if isinstance(obj, dict):
        return {key: serialize(value) for key, value in obj.items()}
    return obj


output = {
    "theorem_statement": theorem_statement.strip(),
    "natural_language_proof": natural_language_proof.strip(),
    "blocks": [serialize(block) for block in blocks],
}


output_path = Path(__file__).resolve().parent / "nl_blocks.json"

output_path.write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Saved NL blocks to: {output_path}")
