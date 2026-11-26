"""Check the fixed arithmetic prompt on the active Qwen gateway."""
import json
import subprocess
import sys


def main():
    result = subprocess.run([sys.executable, "-m", "kgl", "infer", "--json", "What is 2 + 2? Reply with only the digit."],
                            text=True, stdout=subprocess.PIPE, check=True)
    answer = json.loads(result.stdout)["text"].strip()
    print(f"Answer: {answer!r}")
    return 0 if answer == "4" else 1


if __name__ == "__main__":
    raise SystemExit(main())
