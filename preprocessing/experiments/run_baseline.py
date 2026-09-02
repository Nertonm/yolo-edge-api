import argparse
import sys

sys.path.insert(0, '.')

from preprocessing.utils.evaluate import evaluate_pipeline


def parse_classes_arg(raw_value):
    if raw_value is None or raw_value == "":
        return None
    items = raw_value.replace(";", ",").replace(" ", ",")
    values = []
    for part in items.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def build_parser():
    parser = argparse.ArgumentParser(description="Baseline validation for the EPI dataset.")
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Classes a avaliar em CSV ou espaço separado. Ex.: --classes 0,1 ou --classes 0 1",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split do dataset a ser avaliado.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    classes = parse_classes_arg(" ".join(args.classes)) if args.classes else None

    baseline = evaluate_pipeline(
        preprocess_fn=None,
        label=f"baseline (sem preproc, classes={classes})" if classes is not None else "baseline (sem preproc)",
        split=args.split,
        classes=classes,
    )

    print(f"\nBaseline mAP@0.5 = {baseline['map50']:.4f}")
    print("Registre este valor: ele é a referência de todos os experimentos.")
    if classes is not None:
        print(f"Classes avaliadas: {classes}")
