line-count:
    cd ./crates/piston-core && scc -irs
install-pyo3:
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --verbose 3.10.6
    pyenv local 3.10.6
    echo $(python --version)
wasm CRATE PROFILE="release":
    #!/usr/bin/env bash
    set -euxo pipefail

    # 1) Build the crate for the browser target
    if [ {{PROFILE}} = "debug" ]; then
        cargo build \
            --manifest-path ./crates/{{CRATE}}/Cargo.toml \
            --target wasm32-unknown-unknown
    else
        cargo build \
            --manifest-path ./crates/{{CRATE}}/Cargo.toml \
            --target wasm32-unknown-unknown \
            --{{PROFILE}}
    fi

    OUT="target/pkg/{{CRATE}}"
    # 2) Run wasm-bindgen (browser glue, reference types)
    mkdir -p $OUT

    wasm-bindgen "target/wasm32-unknown-unknown/{{PROFILE}}/`echo {{CRATE}} | tr '-' '_'`.wasm" \
      --target web \
      --out-dir $OUT \
      --out-name {{CRATE}} \
      --reference-types

    if [ {{PROFILE}} = "release" ]; then
        # 3) Optimize the bindgen output in place
        binaryen/bin/wasm-opt -O4 \
        --strip-debug \
        --enable-simd \
        --flexible-inline-max-function-size '4294967295' \
        -o $OUT/{{CRATE}}_bg.wasm \
            $OUT/{{CRATE}}_bg.wasm

        # Optional: remove extra custom sections (name/producers, etc.)
        wasm-tools strip -o "$OUT/{{CRATE}}_bg.wasm" "$OUT/{{CRATE}}_bg.wasm"
    fi

    cp ./crates/piston-web/package.json $OUT/package.json

export-libtorch: # Install libtorch
    export LIBTORCH=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)') 
    export DYLD_LIBRARY_PATH=${LIBTORCH}/lib
