# frozen_string_literal: true

module Langchain::LLM
  # LLM interface for OpenAI APIs: https://platform.openai.com/overview
  #
  # Gem requirements:
  #    gem "ruby-openai", "~> 6.3.0"
  #
  # Usage:
  #    openai = Langchain::LLM::OpenAI.new(
  #      api_key: ENV["OPENAI_API_KEY"],
  #      llm_options: {}, # Available options: https://github.com/alexrudall/ruby-openai/blob/main/lib/openai/client.rb#L5-L13
  #      default_options: {}
  #    )
  class OpenAI < Base
    DEFAULTS = {
      n: 1,
      temperature: 0.0,
      chat_completion_model_name: "gpt-3.5-turbo",
      embeddings_model_name: "text-embedding-3-small"
    }.freeze

    EMBEDDING_SIZES = {
      "text-embedding-ada-002" => 1536,
      "text-embedding-3-large" => 3072,
      "text-embedding-3-small" => 1536
    }.freeze

    LENGTH_VALIDATOR = Langchain::Utils::TokenLength::OpenAIValidator

    attr_reader :defaults

    # Initialize an OpenAI LLM instance
    #
    # @param api_key [String] The API key to use
    # @param client_options [Hash] Options to pass to the OpenAI::Client constructor
    def initialize(api_key:, llm_options: {}, default_options: {})
      depends_on "ruby-openai", req: "openai"

      @client = ::OpenAI::Client.new(access_token: api_key, **llm_options)

      @defaults = DEFAULTS.merge(default_options)
      chat_parameters.update(
        model: {default: @defaults[:chat_completion_model_name]},
        logprobs: {},
        top_logprobs: {},
        n: {default: @defaults[:n]},
        stream_options: {},
        temperature: {default: @defaults[:temperature]},
        user: {}
      )
      chat_parameters.ignore(:top_k)

      complete_parameters.update(
        model: {default: @defaults[:chat_completion_model_name]},
        logprobs: {},
        top_logprobs: {},
        n: {default: @defaults[:n]},
        stream_options: {},
        temperature: {default: @defaults[:temperature]},
        user: {}
      )
      complete_parameters.alias_field(:stop, as: :stop_sequences)
      complete_parameters.ignore(:top_k)
    end

    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    # @param model [String] ID of the model to use
    # @param encoding_format [String] The format to return the embeddings in. Can be either float or base64.
    # @param user [String] A unique identifier representing your end-user
    # @return [Langchain::LLM::OpenAIResponse] Response object
    def embed(
      text:,
      model: defaults[:embeddings_model_name],
      encoding_format: nil,
      user: nil,
      dimensions: @defaults[:dimensions]
    )
      raise ArgumentError.new("text argument is required") if text.empty?
      raise ArgumentError.new("model argument is required") if model.empty?
      raise ArgumentError.new("encoding_format must be either float or base64") if encoding_format && %w[float base64].include?(encoding_format)

      parameters = {
        input: text,
        model: model
      }
      parameters[:encoding_format] = encoding_format if encoding_format
      parameters[:user] = user if user

      if dimensions
        parameters[:dimensions] = dimensions
      elsif EMBEDDING_SIZES.key?(model)
        parameters[:dimensions] = EMBEDDING_SIZES[model]
      end

      validate_max_tokens(text, parameters[:model])

      response = with_api_error_handling do
        client.embeddings(parameters: parameters)
      end

      Langchain::LLM::OpenAIResponse.new(response)
    end

    # rubocop:disable Style/ArgumentsForwarding
    # Generate a completion for a given prompt
    #
    # @param params [Hash] The parameters to pass to the `chat()` method
    # @option :prompt [String] The prompt to generate a completion for
    # @option :frequency_penalty [Hash] Penalize new tokens based on their existing frequency
    # @option :logit_bias [Hash] Modify the likelihood of specified tokens appearing
    # @option :logprobs [Boolean] Whether to return log probabilities of the output tokens or not
    # @option :max_tokens [Integer] The maximum number of tokens to sample
    # @option :model [String] The model to use
    # @option :n [Integer] Number of results to return
    # @option :presence_penalty [Hash] Penalize new tokens based on whether they appear in the text so far
    # @option :response_format [Hash] The format that the model must output (e.g. `{"type": "json_object"}`)
    # @option :seed [Integer] A unique number to make a best effort to sample deterministically
    # @option :stop [String, Array<String>] The stop sequences to use
    # @option :stream [Boolean] Whether to stream the response
    # @option :stream_options [Hash] `include_usage: true` adds a `usage` field to streamed chunks
    # @option :temperature [Float] The temperature to use
    # @option :tool_choice [String, Hash] Controls which (if any) tool is called by the model.
    # @option :tools [Array<Hash>] A list of tools the model may call (https://platform.openai.com/docs/guides/text-generation/function-calling)
    # @option :top_p [Float] Use nucleus sampling.
    # @option :user [String] A unique identifier representing your end-user
    # @return [Langchain::LLM::OpenAIResponse] Response object
    def complete(params = {})
      warn "DEPRECATED: `Langchain::LLM::OpenAI#complete` is deprecated, and will be removed in the next major version. Use `Langchain::LLM::OpenAI#chat` instead."

      parameters = complete_parameters.to_params(params)
      parameters[:messages] = []
      parameters[:messages] << {role: "system", content: parameters.delete(:system)} if parameters[:system]
      parameters[:messages] << {role: "user", content: parameters.delete(:prompt)}

      chat(parameters)
    end
    # rubocop:enable Style/ArgumentsForwarding

    # Generate a chat completion for given messages.
    #
    # @param [Hash] params unified chat parmeters from [Langchain::LLM::Parameters::Chat::SCHEMA]
    # @option params [Array<Hash>] :messages List of messages comprising the conversation so far
    # @option params [String] :model ID of the model to use
    def chat(params = {}, &block)
      parameters = chat_parameters.to_params(params)

      raise ArgumentError.new("messages argument is required") if Array(parameters[:messages]).empty?
      raise ArgumentError.new("model argument is required") if parameters[:model].to_s.empty?
      if parameters[:tool_choice] && Array(parameters[:tools]).empty?
        raise ArgumentError.new("'tool_choice' is only allowed when 'tools' are specified.")
      end

      # TODO: Clean this part up
      if block
        @response_chunks = []
        parameters[:stream] = proc do |chunk, _bytesize|
          chunk_content = chunk.dig("choices", 0)
          @response_chunks << chunk
          yield chunk_content
        end
      end

      response = with_api_error_handling do
        client.chat(parameters: parameters)
      end

      response = response_from_chunks if block
      reset_response_chunks

      Langchain::LLM::OpenAIResponse.new(response)
    end

    # Generate a summary for a given text
    #
    # @param text [String] The text to generate a summary for
    # @return [String] The summary
    def summarize(text:)
      prompt_template = Langchain::Prompt.load_from_path(
        file_path: Langchain.root.join("langchain/llm/prompts/summarize_template.yaml")
      )
      prompt = prompt_template.format(text: text)

      complete(prompt: prompt)
    end

    def default_dimensions
      @defaults[:dimensions] || EMBEDDING_SIZES.fetch(defaults[:embeddings_model_name])
    end

    private

    attr_reader :response_chunks

    def reset_response_chunks
      @response_chunks = []
    end

    def with_api_error_handling
      response = yield
      return if response.empty?

      raise Langchain::LLM::ApiError.new "OpenAI API error: #{response.dig("error", "message")}" if response&.dig("error")

      response
    end

    def validate_max_tokens(messages, model, max_tokens = nil)
      LENGTH_VALIDATOR.validate_max_tokens!(messages, model, max_tokens: max_tokens, llm: self)
    end

    def response_from_chunks
      grouped_chunks = @response_chunks.group_by { |chunk| chunk.dig("choices", 0, "index") }
      final_choices = grouped_chunks.map do |index, chunks|
        {
          "index" => index,
          "message" => {
            "role" => "assistant",
            "content" => chunks.map { |chunk| chunk.dig("choices", 0, "delta", "content") }.join
          },
          "finish_reason" => chunks.last.dig("choices", 0, "finish_reason")
        }
      end
      @response_chunks.first&.slice("id", "object", "created", "model")&.merge({"choices" => final_choices})
    end
  end
end
