from __future__ import annotations

import argparse

from .pipeline import BootstrapPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap inicial dos datasets de fraude.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Baixa/localiza, perfila e prepara os datasets.")
    bootstrap_parser.add_argument(
        "--datasets",
        default="all",
        help="Lista separada por virgula dentre: ieee_cis,sparkov,ecommerce ou all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bootstrap":
        dataset_keys = None if args.datasets == "all" else [item.strip() for item in args.datasets.split(",") if item.strip()]
        results = BootstrapPipeline(dataset_keys=dataset_keys).run()
        for result in results:
            print(
                f"{result.display_name}: perfis={len(result.profiles)} "
                f"train={result.prepared.train_path} test={result.prepared.test_path}"
            )


if __name__ == "__main__":
    main()

