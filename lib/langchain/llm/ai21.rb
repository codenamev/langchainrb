# frozen_string_literal: true

module Langchain::LLM
  #
  # Wrapper around AI21 Studio APIs.
  #
  # Gem requirements:
  #   gem "ai21", "~> 0.2.1"
  #
  # Usage:
  #     ai21 = Langchain::LLM::AI21.new(api_key: ENV["AI21_API_KEY"])
  #
  class AI21 < Base
    DEFAULTS = {
      temperature: 0.0,
      model: "j2-ultra"
    }.freeze

    LENGTH_VALIDATOR = Langchain::Utils::TokenLength::AI21Validator

    def initialize(api_key:, default_options: {})
      depends_on "ai21"

      @client = ::AI21::Client.new(api_key)
      @defaults = DEFAULTS.merge(default_options)

      complete_parameters.update(
        model: {default: @defaults[:model]},
        n: {},
        min_tokens: {},
        min_p: {},
        temperature: {default: @defaults[:temperature]},
        epoch: {}
      )
      complete_parameters.ignore(:response_format, :seed, :system, :tool_choice, :tools)
      complete_parameters.remap(
        n: :numResults,
        max_tokens: :maxTokens,
        min_tokens: :minTokens,
        top_p: :topP,
        min_p: :minP,
        stop: :stopSequences,
        top_k: :topKReturn,
        logit_bias: :logitBias,
        frequency_penalty: :frequencyPenalty,
        presence_penalty: :presencePenalty,
        repetition_penalty: :countPenalty
      )
    end

    #
    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a completion for
    # @param params [Hash] The parameters to pass to the API
    # @option params [String] :model The model to use (see: https://api.ai21.com/documentation#tag/completion)
    # @option params [Integer] :n Number of results to return
    # @option params [Integer] :max_tokens The maximum number of tokens to generate
    # @option params [Integer] :min_tokens The minimum number of tokens to generate
    # @option params [Float] :temperature The temperature to use for completion
    # @option params [Float] :top_p Use nucleus sampling.
    # @option params [Float] :min_p Use nucleus sampling.
    # @option params [Integer] :top_k Only sample from the top K options for each subsequent token
    # @option params [Array<String>] :stop The stop sequences to use for completion
    # @option params [Hash] :logit_bias Modify the likelihood of specified tokens appearing
    # @option params [Hash] :frequency_penalty Penalize new tokens based on their existing frequency
    # @option params [Hash] :presence_penalty Penalize new tokens based on whether they appear in the text so far
    # @option params [Hash] :repetition_penalty Penalize new tokens based on whether they appear in the text so far
    # @option params [Hash] :count_penalty Penalize new tokens based on specific counts
    # @option params [Integer] :epoch
    # @return [Langchain::LLM::AI21Response] The completion
    def complete(params = {})
      parameters = complete_parameters.to_params(params)
      prompt = parameters[:prompt]

      parameters[:maxTokens] = LENGTH_VALIDATOR.validate_max_tokens!(prompt, parameters[:model], {llm: client})

      response = client.complete(prompt, parameters.except(:prompt))
      Langchain::LLM::AI21Response.new response, model: parameters[:model]
    end

    #
    # Generate a summary for a given text
    #
    # @param text [String] The text to generate a summary for
    # @param params [Hash] The parameters to pass to the API
    # @return [String] The summary
    #
    def summarize(text:, **params)
      response = client.summarize(text, "TEXT", params)
      response.dig(:summary)
      # Should we update this to also return a Langchain::LLM::AI21Response?
    end
  end
end
