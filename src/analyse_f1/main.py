"""Execute the main `analyse-f1` pipeline."""

from analyse_f1 import __version__


def main() -> None:
    """Execute the main `analyse-f1` pipeline."""
    print(
        r"""    _                _                  _____ _
   / \   _ __   __ _| |_   _ ___  ___  |  ___/ |
  / _ \ | '_ \ / _` | | | | / __|/ _ \ | |_  | |
 / ___ \| | | | (_| | | |_| \__ \  __/ |  _| | |
/_/   \_\_| |_|\__,_|_|\__, |___/\___| |_|   |_|
                       |___/                            Version: {}
    """.format(__version__)
    )


if __name__ == "__main__":
    main()
