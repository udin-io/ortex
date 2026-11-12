defmodule Ortex.Model do
  @moduledoc """
  A model for running Ortex inference with.

  Implements a human-readable representation of a model including the name, dimension, and
  type of each input and output

  ```
  #Ortex.Model<
  inputs: [{"x", "Int32", [nil, 100]}, {"y", "Float32", [nil, 100]}]
  outputs: [
    {"9", "Float32", [nil, 10]},
    {"onnx::Add_7", "Float32", [nil, 10]},
    {"onnx::Add_8", "Float32", [nil, 10]}
  ]>
  ```

  `nil` values represent dynamic dimensions
  """

  @enforce_keys [:reference]
  defstruct [:reference]

  @doc false
  def load(path, eps \\ [:cpu], opt \\ 3, options \\ []) do
    options =
      Keyword.validate!(options,
        intra_threads: nil,
        coreml_cache_dir: nil,
        tensorrt: nil,
        config_entries: []
      )

    intra_threads = intra_threads!(intra_threads: options[:intra_threads])
    cache_dir = cache_dir!(options[:coreml_cache_dir])
    tensorrt = tensorrt!(options[:tensorrt])
    config_entries = config_entries!(options[:config_entries])

    case Ortex.Native.init(path, eps, opt, intra_threads, cache_dir, tensorrt, config_entries) do
      {:error, msg} ->
        raise msg

      model ->
        %Ortex.Model{reference: model}
    end
  end

  defp intra_threads!(intra_threads: nil), do: nil

  defp intra_threads!(intra_threads: threads) when is_integer(threads) and threads > 0,
    do: threads

  defp intra_threads!(intra_threads: threads) do
    raise ArgumentError,
          "expected :intra_threads to be a positive integer, got: #{inspect(threads)}"
  end

  defp cache_dir!(nil), do: nil
  defp cache_dir!(dir) when is_binary(dir) and dir != "", do: dir

  defp cache_dir!(dir) do
    raise ArgumentError,
          "expected :coreml_cache_dir to be a non-empty path, got: #{inspect(dir)}"
  end

  defp config_entries!(entries) when is_list(entries) do
    Enum.map(entries, fn
      {key, value} when is_binary(key) and key != "" and is_binary(value) ->
        if String.contains?(key <> value, <<0>>), do: config_entries_error!(entries)
        {key, value}

      _ ->
        config_entries_error!(entries)
    end)
  end

  defp config_entries!(entries), do: config_entries_error!(entries)

  defp config_entries_error!(entries) do
    raise ArgumentError,
          "expected :config_entries to be a list of {key, value} strings with a non-empty key and no NUL byte, got: #{inspect(entries)}"
  end

  defp tensorrt!(nil), do: nil

  defp tensorrt!(options) when is_list(options) do
    if not Keyword.keyword?(options) do
      raise ArgumentError, "expected :tensorrt to be a keyword list, got: #{inspect(options)}"
    end

    options =
      Keyword.validate!(options, cache_dir: nil, timing_cache_dir: nil, fp16: false, profile: nil)

    {tensorrt_dir!(:cache_dir, options[:cache_dir]),
     tensorrt_dir!(:timing_cache_dir, options[:timing_cache_dir]), fp16!(options[:fp16]),
     profile!(options[:profile])}
  end

  defp tensorrt!(options) do
    raise ArgumentError, "expected :tensorrt to be a keyword list, got: #{inspect(options)}"
  end

  defp tensorrt_dir!(_name, nil), do: nil

  defp tensorrt_dir!(name, dir) when is_binary(dir) do
    if dir == "" or String.contains?(dir, <<0>>) do
      raise ArgumentError, "expected tensorrt #{inspect(name)} to be a non-empty path"
    end

    dir
  end

  defp tensorrt_dir!(name, dir) do
    raise ArgumentError,
          "expected tensorrt #{inspect(name)} to be a non-empty path, got: #{inspect(dir)}"
  end

  defp fp16!(fp16) when is_boolean(fp16), do: fp16

  defp profile!(nil), do: nil

  defp profile!({min, opt, max} = profile) do
    if Enum.all?([min, opt, max], &shapes?/1) do
      profile
    else
      profile_error!(profile)
    end
  end

  defp profile!(profile), do: profile_error!(profile)

  defp shapes?(shapes),
    do: is_binary(shapes) and shapes != "" and not String.contains?(shapes, <<0>>)

  defp profile_error!(profile) do
    raise ArgumentError,
          "expected tensorrt :profile to be {min, opt, max}, three non-empty shape strings " <>
            "with no NUL byte, got: #{inspect(profile)}"
  end

  defp fp16!(fp16) do
    raise ArgumentError, "expected tensorrt :fp16 to be a boolean, got: #{inspect(fp16)}"
  end

  @doc false
  def run(%Ortex.Model{} = model, tensor) when not is_tuple(tensor) do
    run(model, {tensor})
  end

  @doc false
  def run(%Ortex.Model{reference: model}, tensors) do
    # Move tensors into Ortex backend and pass the reference to the Ortex NIF
    output =
      case Ortex.Native.run(
             model,
             tensors
             |> Tuple.to_list()
             |> Enum.map(fn x -> x |> Nx.backend_transfer(Ortex.Backend) end)
             |> Enum.map(fn %Nx.Tensor{data: %Ortex.Backend{ref: x}} -> x end)
           ) do
        {:error, msg} -> raise msg
        output -> output
      end

    # Pack the output into new Ortex.Backend tensor(s)
    output
    |> Enum.map(fn {ref, shape, dtype_atom, dtype_bits} ->
      %Nx.Tensor{
        data: %Ortex.Backend{ref: ref},
        shape: shape |> List.to_tuple(),
        type: {dtype_atom, dtype_bits},
        names: List.duplicate(nil, length(shape))
      }
    end)
    |> List.to_tuple()
  end
end

defimpl Inspect, for: Ortex.Model do
  import Inspect.Algebra

  def inspect(%Ortex.Model{reference: model}, inspect_opts) do
    case Ortex.Native.show_session(model) do
      {:error, msg} ->
        raise msg

      {inputs, outputs} ->
        force_unfit(
          concat([
            color("#Ortex.Model<", :map, inspect_opts),
            line(),
            nest(concat(["  inputs: ", Inspect.Algebra.to_doc(inputs, inspect_opts)]), 2),
            line(),
            nest(concat(["  outputs: ", Inspect.Algebra.to_doc(outputs, inspect_opts)]), 2),
            color(">", :map, inspect_opts)
          ])
        )
    end
  end
end
