defmodule Ortex do
  @moduledoc """
  Documentation for `Ortex`.

  `Ortex` is an Elixir wrapper around [ONNX Runtime](https://onnxruntime.ai/) using
  [Rustler](https://hexdocs.pm/rustler) and [ORT](https://github.com/pykeio/ort).
  """

  @doc """
  Load an `Ortex.Model` from disk. Optionally pass the execution providers as a list
  of descending priority and graph optimization level 1-3. Any graph optimization level
  beyond the range of 1-3 will disable graph optimization.

  By default, `Ortex` only includes some of the supported execution providers of ONNX Runtime.
  To enable others, first ensure you have downloaded or compiled a version of
  `libonnxruntime` that includes them, then set the environment variable `ORT_LIB_LOCATION`
  to its location. Then add `config :ortex, Ortex.Native, features: [EXECUTION_PROVIDERS]` to your
  `config.exs` where `EXECUTION_PROVIDERS` is a list of strings of which execution providers
  to enable.

  A provider that cannot be registered fails the load and names the provider,
  whether its Cargo feature is off or ONNX Runtime refuses it (a provider
  library that does not load, a CUDA without a device). `ort` would otherwise
  run the session on the CPU and say so only through `tracing`.

  ## Options

    * `:intra_threads` - the number of threads ONNX Runtime spreads a single
      inference across. By default ONNX Runtime uses one thread per physical
      core, so one busy model can occupy most of the machine; a smaller pool
      trades per-inference latency for cores left free for other work, such
      as a second model. Must be a positive integer.
    * `:coreml_cache_dir` - where CoreML keeps the model it compiles, so the
      next session on the same model skips the compile. CoreML recompiles
      only when the graph or the model's metadata changes, not its weights,
      so give each model file's content its own directory.
    * `:tensorrt` - options for the `:tensorrt` provider, ignored when `eps`
      does not ask for it:
        * `:cache_dir` - where TensorRT writes the engine it builds and reads
          it back on the next session. ONNX Runtime reuses an engine by this
          path alone and never notices a new TensorRT, driver or model, so
          give each combination its own directory.
        * `:timing_cache_dir` - where TensorRT keeps its timing cache, which
          speeds up later builds on the same card.
        * `:fp16` - build the engine in half precision. Defaults to `false`.
        * `:profile` - `{min, opt, max}`, the smallest, usual and largest
          input shapes the engine takes, each as ONNX Runtime writes them:
          `"input:1x3x256x192"`, inputs comma separated. Without it TensorRT
          builds for the shape of the first run. Give the same three on every
          load of a cached engine, or ONNX Runtime rebuilds it.

    * `:config_entries` - `{key, value}` strings handed to ONNX Runtime's
      `AddSessionConfigEntry` before the session is built, such as
      `{"mlas.disable_kleidiai", "1"}`. Defaults to `[]`.

  With the `coreml` feature, `:coreml` registers CoreML as an ML Program on
  the CPU and GPU, rather than through `ort`'s legacy NeuralNetwork flags.

  ## Examples

      iex> Ortex.load("./models/tinymodel.onnx")
      iex> Ortex.load("./models/tinymodel.onnx", [:cpu], 0)
      iex> Ortex.load("./models/tinymodel.onnx", [:cpu], 3, intra_threads: 2)

  """
  defdelegate load(path, eps \\ [:cpu], opt \\ 3, options \\ []), to: Ortex.Model

  @doc """
  Run a forward pass through a model.

  This takes a model and tuple of `Nx.Tensors`,
  optionally transfers them to the `Ortex.Backend` if they aren't there already,
  and runs a forward pass through the model. This will return a tuple of `Ortex.Backend`
  tensors, it's up to the user to transfer these back to another backend if additional
  ops are required.

  If there is only one input you can optionally pass a bare tensor rather than a tuple.

  ## Examples

      iex> model = Ortex.load("./models/tinymodel.onnx")
      iex> {%Nx.Tensor{shape: {1, 10}},
      ...>  %Nx.Tensor{shape: {1, 10}},
      ...>  %Nx.Tensor{shape: {1, 10}}} = Ortex.run(
      ...>    model, {
      ...>      Nx.broadcast(0, {1, 100}) |> Nx.as_type(:s32),
      ...>      Nx.broadcast(0, {1, 100}) |> Nx.as_type(:f32)
      ...>    })

  """
  defdelegate run(model, tensors), to: Ortex.Model
end
