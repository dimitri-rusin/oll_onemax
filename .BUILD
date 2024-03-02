#!/usr/bin/env fish

true
and conda activate base
and rm -rf ./.conda_environment/
and rm -rf ./paper_code/onell_algs_rs/target/
and conda env create --prefix ./.conda_environment/ --file .conda.yaml
and conda activate ./.conda_environment/
and pip install --requirement .pip.txt

and cd ./paper_code/onell_algs_rs/
and maturin build --release
and cd -
and pip install --force-reinstall ./paper_code/onell_algs_rs/target/wheels/onell_algs_rs-0.1.0-cp310-cp310-manylinux_2_31_x86_64.whl
