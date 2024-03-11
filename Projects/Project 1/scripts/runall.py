import subprocess
import os

if __name__ == "__main__":
    print("Running all project 1 scripts...")

    # Run all scripts problem 9 to 14
    for nr in range(9, 15):
        print(f"\n\nProblem {nr}:")
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "scripts",
                f"problem{nr}.py",
            )
        )
        subprocess.run(["python", path])
