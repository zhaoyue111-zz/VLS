from __future__ import annotations

from vls.v7_1c_class_balanced_loss_sanity import parse_args, run


if __name__ == "__main__":
    arguments = parse_args()
    arguments.include_uniform = True
    if arguments.output_dir == "outputs/v7_1c_class_balanced_loss_sanity":
        arguments.output_dir = "outputs/v7_1d_reliability_contribution_isolation"
    run(arguments)
