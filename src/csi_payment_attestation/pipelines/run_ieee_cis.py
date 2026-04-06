import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IEEE-CIS + CSI experiment pipeline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(f"Running IEEE-CIS pipeline with config: {args.config}")


if __name__ == "__main__":
    main()
