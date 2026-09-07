import os
from pathlib import Path
import asyncio
import sys

try:
	from pantograph.server import Server
except ModuleNotFoundError:
	try:
		from Pantograph.server import Server
	except ModuleNotFoundError:
		print("Could not import 'pantograph.server' or 'Pantograph.server'.")
		print("If you've installed PyPantograph, ensure it's in the active environment and importable.")
		print("Install with:")
		print("  pip install git+https://github.com/stanford-centaur/PyPantograph")
		print("Then run this script in that environment.")
		sys.exit(1)


project_path = Path(os.getcwd())/ 'src/Example'
print(f"$PWD: {project_path}")


async def main():
	server = await Server.create(imports=['Example'], project_path=project_path)
	units = await server.tactic_invocations_async(project_path / "Example.lean")
	print(units)


if __name__ == "__main__":
	asyncio.run(main())

