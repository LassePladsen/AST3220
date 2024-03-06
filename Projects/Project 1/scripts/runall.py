import subprocess
import os

if __name__ == "__main__":
    print("Running all project 1 scripts...")

    # Run all scripts
    for nr in range(9, 15):
        print(f"\nProblem {nr}:")
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "scripts",
                f"problem{nr}.py",
            )
        )
        subprocess.run(["python", path])