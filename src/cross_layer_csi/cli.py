from __future__ import annotations

import argparse

from .pipelines.csi import CSIPreprocessingPipeline
from .pipelines.tabular import TabularBootstrapPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-layer Tx+CSI project bootstrap and preprocessing utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tabular_parser = subparsers.add_parser(
        "tabular-bootstrap",
        help="Locate/download the tabular fraud datasets and generate prepared artifacts.",
    )
    tabular_parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated subset among: ieee_cis,sparkov,ecommerce or all.",
    )

    csi_parser = subparsers.add_parser(
        "csi-preprocess",
        help="Run the CSI amplitude conversion, filtering, smoothing, and harmonization flow.",
    )
    csi_parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt to download the CSI sources before preprocessing them.",
    )
    csi_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable diagnostic plots during CSI preprocessing.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "tabular-bootstrap":
        dataset_keys = None if args.datasets == "all" else [item.strip() for item in args.datasets.split(",") if item.strip()]
        results = TabularBootstrapPipeline(dataset_keys=dataset_keys).run()
        for result in results:
            print(
                f"{result.display_name}: perfis={len(result.profiles)} "
                f"train={result.prepared.train_path} test={result.prepared.test_path}"
            )
        return

    if args.command == "csi-preprocess":
        result = CSIPreprocessingPipeline(render_plots=not args.no_plots).run(download=args.download)
        print(
            "CSI preprocessing complete: "
            f"converted={result.converted_dir} filtered={result.filtered_dir} "
            f"smoothed={result.smoothed_dir} harmonized={result.harmonized_dir}"
        )


if __name__ == "__main__":
    main()
