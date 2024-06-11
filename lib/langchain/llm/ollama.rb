# frozen_string_literal: true

require "active_support/core_ext/hash"

module Langchain::LLM
  # Interface to Ollama API.
  # Available models: https://ollama.ai/library
  #
  # Usage:
  #    llm = Langchain::LLM::Ollama.new
  #    llm = Langchain::LLM::Ollama.new(url: ENV["OLLAMA_URL"], default_options: {})
  #
  class Ollama < Base
    attr_reader :url, :defaults

    DEFAULTS = {
      temperature: 0.8,
      completion_model_name: "llama3",
      embeddings_model_name: "llama3",
      chat_completion_model_name: "llama3"
    }.freeze

    EMBEDDING_SIZES = {
      codellama: 4_096,
      "dolphin-mixtral": 4_096,
      llama2: 4_096,
      llama3: 4_096,
      llava: 4_096,
      mistral: 4_096,
      "mistral-openorca": 4_096,
      mixtral: 4_096
    }.freeze

    # Initialize the Ollama client
    # @param url [String] The URL of the Ollama instance
    # @param default_options [Hash] The default options to use
    #
    def initialize(url: "http://localhost:11434", default_options: {})
      depends_on "faraday"
      @url = url
      @defaults = DEFAULTS.deep_merge(default_options)
      chat_parameters.update(
        model: {default: @defaults[:chat_completion_model_name]},
        temperature: {default: @defaults[:temperature]},
        template: {},
        stream: {default: false}
      )
      chat_parameters.remap(response_format: :format)
    end

    # Returns the # of vector dimensions for the embeddings
    # @return [Integer] The # of vector dimensions
    def default_dimensions
      # since Ollama can run multiple models, look it up or generate an embedding and return the size
      @default_dimensions ||=
        EMBEDDING_SIZES.fetch(defaults[:embeddings_model_name].to_sym) do
          embed(text: "test").embedding.size
        end
    end

    #
    # Generate the completion for a given prompt
    #
    # @param prompt [String] The prompt to complete
    # @param model [String] The model to use
    #   For a list of valid parameters and values, see:
    #   https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    # @return [Langchain::LLM::OllamaResponse] Response object
    #
    def complete(
      prompt:,
      model: defaults[:completion_model_name],
      images: nil,
      format: nil,
      system: nil,
      template: nil,
      context: nil,
      stream: nil,
      raw: nil,
      mirostat: nil,
      mirostat_eta: nil,
      mirostat_tau: nil,
      num_ctx: nil,
      num_gqa: nil,
      num_gpu: nil,
      num_thread: nil,
      repeat_last_n: nil,
      repeat_penalty: nil,
      temperature: defaults[:temperature],
      seed: nil,
      stop: nil,
      tfs_z: nil,
      num_predict: nil,
      top_k: nil,
      top_p: nil,
      stop_sequences: nil,
      &block
    )
      if stop_sequences
        stop = stop_sequences
      end

      parameters = {
        prompt: prompt,
        model: model,
        images: images,
        format: format,
        system: system,
        template: template,
        context: context,
        stream: stream,
        raw: raw
      }.compact

      llm_parameters = {
        mirostat: mirostat,
        mirostat_eta: mirostat_eta,
        mirostat_tau: mirostat_tau,
        num_ctx: num_ctx,
        num_gqa: num_gqa,
        num_gpu: num_gpu,
        num_thread: num_thread,
        repeat_last_n: repeat_last_n,
        repeat_penalty: repeat_penalty,
        temperature: temperature,
        seed: seed,
        stop: stop,
        tfs_z: tfs_z,
        num_predict: num_predict,
        top_k: top_k,
        top_p: top_p
      }

      parameters[:options] = llm_parameters.compact

      streamed_response = ""

      completed_response = client.post("api/generate") do |req|
        req.body = parameters

        req.options.on_data = proc do |chunk, size|
          chunk.split("\n".dup).each do |line_chunk|
            json_chunk = begin
              JSON.parse(line_chunk)
            # In some instance the chunk exceeds the buffer size and the JSON parser fails
            rescue JSON::ParserError
              nil
            end

            streamed_response += json_chunk.dig("response") unless json_chunk&.dig("response").blank?
          end

          yield json_chunk, size if block
        end
      end

      response = streamed_response.to_s.empty? ? completed_response.body.dig("response") : streamed_response

      Langchain::LLM::OllamaResponse.new(response, model: parameters[:model])
    end

    # Generate a chat completion
    #
    # @param [Hash] params unified chat parmeters from [Langchain::LLM::Parameters::Chat::SCHEMA]
    # @option params [String] :model Model name
    # @option params [Array<Hash>] :messages Array of messages
    # @option params [String] :format Format to return a response in. Currently the only accepted value is `json`
    # @option params [Float] :temperature The temperature to use
    # @option params [String] :template The prompt template to use (overrides what is defined in the `Modelfile`)
    # @option params [Boolean] :stream Streaming the response. If false the response will be returned as a single response object, rather than a stream of objects
    #
    # The message object has the following fields:
    #   role: the role of the message, either system, user or assistant
    #   content: the content of the message
    #   images (optional): a list of images to include in the message (for multimodal models such as llava)
    def chat(params = {})
      parameters = chat_parameters.to_params(params)

      response = client.post("api/chat") do |req|
        req.body = parameters
      end

      Langchain::LLM::OllamaResponse.new(response.body, model: parameters[:model])
    end

    #
    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    # @param model [String] The model to use
    # @param options [Hash] The options to use
    # @return [Langchain::LLM::OllamaResponse] Response object
    #
    def embed(
      text:,
      model: defaults[:embeddings_model_name],
      mirostat: nil,
      mirostat_eta: nil,
      mirostat_tau: nil,
      num_ctx: nil,
      num_gqa: nil,
      num_gpu: nil,
      num_thread: nil,
      repeat_last_n: nil,
      repeat_penalty: nil,
      temperature: defaults[:temperature],
      seed: nil,
      stop: nil,
      tfs_z: nil,
      num_predict: nil,
      top_k: nil,
      top_p: nil
    )
      parameters = {
        prompt: text,
        model: model
      }.compact

      llm_parameters = {
        mirostat: mirostat,
        mirostat_eta: mirostat_eta,
        mirostat_tau: mirostat_tau,
        num_ctx: num_ctx,
        num_gqa: num_gqa,
        num_gpu: num_gpu,
        num_thread: num_thread,
        repeat_last_n: repeat_last_n,
        repeat_penalty: repeat_penalty,
        temperature: temperature,
        seed: seed,
        stop: stop,
        tfs_z: tfs_z,
        num_predict: num_predict,
        top_k: top_k,
        top_p: top_p
      }

      parameters[:options] = llm_parameters.compact

      response = client.post("api/embeddings") do |req|
        req.body = parameters
      end

      Langchain::LLM::OllamaResponse.new(response.body, model: parameters[:model])
    end

    # Generate a summary for a given text
    #
    # @param text [String] The text to generate a summary for
    # @return [String] The summary
    def summarize(text:)
      prompt_template = Langchain::Prompt.load_from_path(
        file_path: Langchain.root.join("langchain/llm/prompts/ollama/summarize_template.yaml")
      )
      prompt = prompt_template.format(text: text)

      complete(prompt: prompt)
    end

    private

    # @return [Faraday::Connection] Faraday client
    def client
      @client ||= Faraday.new(url: url) do |conn|
        conn.request :json
        conn.response :json
        conn.response :raise_error
      end
    end
  end
end
