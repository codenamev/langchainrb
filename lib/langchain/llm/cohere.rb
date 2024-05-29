# frozen_string_literal: true

module Langchain::LLM
  #
  # Wrapper around the Cohere API.
  #
  # Gem requirements:
  #     gem "cohere-ruby", "~> 0.9.6"
  #
  # Usage:
  #     llm = Langchain::LLM::Cohere.new(api_key: ENV["COHERE_API_KEY"])
  #
  class Cohere < Base
    DEFAULTS = {
      temperature: 0.0,
      completion_model_name: "command",
      chat_completion_model_name: "command-r-plus",
      embeddings_model_name: "small",
      dimensions: 1024,
      truncate: "START"
    }.freeze

    def initialize(api_key:, default_options: {})
      depends_on "cohere-ruby", req: "cohere"

      @client = ::Cohere::Client.new(api_key: api_key)
      @defaults = DEFAULTS.merge(default_options)
      chat_parameters.update(
        model: {default: @defaults[:chat_completion_model_name]},
        temperature: {default: @defaults[:temperature]}
      )
      chat_parameters.remap(
        system: :preamble,
        messages: :chat_history,
        stop: :stop_sequences,
        top_k: :k,
        top_p: :p
      )

      complete_parameters.update(
        model: {default: @defaults[:completion_model_name]},
        n: {},
        truncate: {},
        temperature: {default: @defaults[:temperature]},
        preset: {},
        end_sequences: {},
        return_likelihoods: {},
        raw_prompting: {}
      )
      complete_parameters.remap(
        n: :num_generations,
        stop: :stop_sequences,
        top_k: :k,
        top_p: :p
      )
      complete_parameters.ignore(
        :system,
        :response_format,
        :tools,
        :tool_choice,
        :logit_bias,
        :repetition_penalty
      )
    end

    #
    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    # @return [Langchain::LLM::CohereResponse] Response object
    #
    def embed(text:)
      response = client.embed(
        texts: [text],
        model: @defaults[:embeddings_model_name]
      )

      Langchain::LLM::CohereResponse.new response, model: @defaults[:embeddings_model_name]
    end

    #
    # Generate a completion for a given prompt
    #
    # @param params [Hash] The parameters to pass to the `generate()` method
    # @option :prompt [String] The prompt to generate a completion for
    # @option :model [String] The identifier of the model to generate with (`command` (default), `command-nightly` (experimental), `command-light`, and `command-light-nightly` (experimental))
    # @option :n [Integer] The maximum number of generations that will be returned. Defaults to 1, min value of 1, max value of 5.
    # @option :stream [boolean] # When true, the response will be a JSON stream of events.
    # @option :max_tokens [Integer] The maximum number of tokens the model will generate as part of the response. Can only be set to 0 if return_likelihoods is set to ALL to get the likelihood of the prompt.
    # @option :truncate [String] One of NONE|START|END to specify how the API will handle inputs longer than the maximum token length. Default: END
    # @option :temperature [Number] A non-negative float that tunes the degree of randomness in generation. Defaults to 0.75, min value of 0.0, max value of 5.0.
    # @option :seed [Number] If specified, the backend will make a best effort to sample tokens deterministically, such that repeated requests with the same seed and parameters should return the same result. However, determinsim cannot be totally guaranteed.
    # @option :preset [String] Identifier of a custom preset. A preset is a combination of parameters, such as prompt, temperature etc. When a preset is specified, the prompt parameter becomes optional, and any included parameters will override the preset's parameters.
    # @option :end_sequences [Array of strings] The generated text will be cut at the beginning of the earliest occurrence of an end sequence. The sequence will be excluded from the text.
    # @option :stop [Array of strings] The generated text will be cut at the end of the earliest occurrence of a stop sequence. The sequence will be included the text.
    # @option :k [Integer] Ensures only the top k most likely tokens are considered for generation at each step. Defaults to 0, min value of 0, max value of 500.
    # @option :p [Number] Ensures that only the most likely tokens, with total probability mass of p, are considered for generation at each step. If both k and p are enabled, p acts after k. Defaults to 0.75. min value of 0.01, max value of 0.99.
    # @option :frequency_penalty [Number] Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens, proportional to how many times they have already appeared in the prompt or prior generation.
    # @option :presence_penalty [Number] Can be used to reduce repetitiveness of generated tokens. Similar to frequency_penalty, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies. Defaults to 0.0, min value of 0.0, max value of 1.0.
    # @option :return_likelihoods [String] One of GENERATION|ALL|NONE to specify how and if the token likelihoods are returned with the response. Defaults to NONE.
    # @option :raw_prompting [Boolean] When enabled, the user's prompt will be sent to the model without any pre-processing.
    #
    # @return [Langchain::LLM::CohereResponse] Response object
    def complete(params = {})
      parameters = complete_parameters.to_params(params)

      parameters[:max_tokens] = Langchain::Utils::TokenLength::CohereValidator.validate_max_tokens!(
        parameters[:prompt],
        parameters[:model],
        llm: client
      )

      response = client.generate(parameters)
      Langchain::LLM::CohereResponse.new response, model: parameters[:model]
    end

    # Generate a chat completion for given messages
    #
    # @param [Hash] params unified chat parameters from [Langchain::LLM::Parameters::Chat::SCHEMA]
    # @option params [Array<String>] :messages Input messages
    # @option params [String] :model The model that will complete your prompt
    # @option params [Integer] :max_tokens Maximum number of tokens to generate before stopping
    # @option params [Array<String>] :stop Custom text sequences that will cause the model to stop generating
    # @option params [Boolean] :stream Whether to incrementally stream the response using server-sent events
    # @option params [String] :system System prompt
    # @option params [Float] :temperature Amount of randomness injected into the response
    # @option params [Array<String>] :tools Definitions of tools that the model may use
    # @option params [Integer] :top_k Only sample from the top K options for each subsequent token
    # @option params [Float] :top_p Use nucleus sampling.
    # @return [Langchain::LLM::CohereResponse] The chat completion
    def chat(params = {})
      raise ArgumentError.new("messages argument is required") if Array(params[:messages]).empty?

      parameters = chat_parameters.to_params(params)

      response = client.chat(**parameters)

      Langchain::LLM::CohereResponse.new(response)
    end

    # Generate a summary in English for a given text
    #
    # More parameters available to extend this method with: https://github.com/andreibondarev/cohere-ruby/blob/0.9.4/lib/cohere/client.rb#L107-L115
    #
    # @param text [String] The text to generate a summary for
    # @return [String] The summary
    def summarize(text:)
      response = client.summarize(text: text)
      response.dig("summary")
    end
  end
end
