import os
from pathlib import Path
import asyncio
import sys

try:
	from pantograph.server import Server
except ModuleNotFoundError:
	print("Missing dependency: pantograph is not installed in this environment.")
	print("Install it inside your active environment with:")
	print("  pip install git+https://github.com/stanford-centaur/PyPantograph")
	print("Or install project dependencies (if using pip):")
	print("  pip install -e .")
	sys.exit(1)


project_path = Path(os.getcwd()).parent.resolve() / 'Example'
print(f"$PWD: {project_path}")


async def main():
	server = await Server.create(imports=['Example'], project_path=project_path)
	units = await server.tactic_invocations_async(project_path / "Example.lean")
	print(units)


if __name__ == "__main__":
	asyncio.run(main())

