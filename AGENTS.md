# Project Optimization Rules

## Objective

- Optimize only the LLM inference prefill work executed by `generate()` in `main.cpp`.
- Preserve the program logic, model architecture, algorithm, and required outputs.
- GPU optimizations should use CUDA.

## Files That Must Not Be Modified

- `inpust.bin`
- `answer.bin`
- `decode_answers.bin`
- `model.bin`
- `main.cpp`
- `config.h`
- `makefile`

Treat `inpust.bin` as the exact filename supplied by the user.

## Files That May Be Modified

- `tensor.h`
- `tensor.cu`
- `layer.h`
- `layer.cu`
- `model.h`
- `model.cu`
- `model_loader.h`
- `model_loader.cu`
- `run.sh`
- `AGENTS.md` only for maintaining these project instructions

Do not create or modify other project files unless the user explicitly authorizes it.

## Allowed Optimizations

- Change memory layouts.
- Reorder loops.
- Add computation over padding data where useful.
- Fuse operations or CUDA kernels.
- Make equivalent low-level implementation changes that retain the same model and outputs.

## Prohibited Changes

- Do not change program logic or the model structure.
- Do not replace the model computation with a different algorithm merely producing the same output.
- Do not move measured `generate()` model-inference work outside the timed region.
- Do not perform major inference operations before timing and cache their results for use inside the timed region.
- Do not add warmup execution that circumvents the intended measurement.
- Do not optimize decode or unrelated execution at the expense of changing the measured task; the target is prefill only.

## Prohibited Libraries

Do not use cuBLAS, cuDNN, CLBlast, MAGMA, BLIS, PyTorch, TensorFlow, or similar external compute/ML libraries.

## Validation

- Keep the required output identical according to the project's existing correctness checks.
- Measure performance only under the existing timing and execution rules.
- Never modify immutable files to make validation pass.
