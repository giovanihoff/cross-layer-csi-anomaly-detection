import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sparkov + CSI experiment pipeline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(f"Running Sparkov pipeline with config: {args.config}")


if __name__ == "__main__":
    main()
