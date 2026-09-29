# nuitka-project: --standalone
# nuitka-project: --onefile

import numpy as np


def main():
    arr = np.array([1, 2, 3, 4, 5])
    print(f"NumPy Mean: {arr.mean()}")


if __name__ == "__main__":
    main()
