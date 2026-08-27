from __future__ import annotations

import argparse
import json

from .experiments.config import DATASET_SEED_OFFSETS, SEED_POLICY_V120_3, SEGMENT_SIZES
from .experiments.registry import NOTEBOOK_PROTOCOL_VERSION
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
        help="Comma-separated subset among: ieee_cis,sparkov,ecommerce,caixabank or all.",
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

    manifest_parser = subparsers.add_parser(
        "protocol-manifest",
        help="Print the explicit v120.3 seed-role manifest.",
    )
    manifest_parser.add_argument(
        "--segment-size",
        type=int,
        choices=SEGMENT_SIZES,
        default=25,
        help="CSI segment size used to derive scenario seeds.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "tabular-bootstrap":
        dataset_keys = (
            None
            if args.datasets == "all"
            else [item.strip() for item in args.datasets.split(",") if item.strip()]
        )
        results = TabularBootstrapPipeline(dataset_keys=dataset_keys).run()
        for result in results:
            print(
                f"{result.display_name}: perfis={len(result.profiles)} "
                f"train={result.prepared.train_path} test={result.prepared.test_path}"
            )
        return

    if args.command == "csi-preprocess":
        result = CSIPreprocessingPipeline(render_plots=not args.no_plots).run(
            download=args.download
        )
        print(
            "CSI preprocessing complete: "
            f"converted={result.converted_dir} filtered={result.filtered_dir} "
            f"smoothed={result.smoothed_dir} harmonized={result.harmonized_dir}"
        )
        return

    if args.command == "protocol-manifest":
        scenarios = []
        for dataset_label in DATASET_SEED_OFFSETS:
            seeds = SEED_POLICY_V120_3.for_scenario(dataset_label, args.segment_size)
            scenarios.append(
                {
                    "dataset": dataset_label,
                    "segment_size": args.segment_size,
                    "merge_seed": seeds.merge,
                    "injection_seed": seeds.injection,
                    "evaluation_seed": seeds.evaluation,
                }
            )
        print(
            json.dumps(
                {"protocol_version": NOTEBOOK_PROTOCOL_VERSION, "scenarios": scenarios},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
