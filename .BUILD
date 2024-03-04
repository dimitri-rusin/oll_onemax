#!/usr/bin/env fish

true

and git submodule init
and git submodule update

and conda activate base
and rm -rf ./.conda_environment/
and rm -rf ./paper_code/onell_algs_rs/target/
and conda env create --prefix ./.conda_environment/ --file .conda.yaml
and conda activate ./.conda_environment/
and pip install --requirement .pip.txt

and cd ./paper_code/onell_algs_rs/
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# export PATH="$HOME/.cargo/bin:$PATH"
and maturin build --release
and cd -
and find ./paper_code/onell_algs_rs/target/wheels -name "*.whl" -print0 | xargs -0 pip install
