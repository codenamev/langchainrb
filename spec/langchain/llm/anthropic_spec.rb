# frozen_string_literal: true

require "anthropic"

RSpec.describe Langchain::LLM::Anthropic do
  let(:subject) { described_class.new(api_key: "123") }

  describe "#complete" do
    let(:completion) { "How high is the sky?" }
    let(:fixture) { File.read("spec/fixtures/llm/anthropic/complete.json") }
    let(:response) { JSON.parse(fixture) }

    context "with no additional parameters" do
      before do
        allow(subject.client).to receive(:complete)
          .with(parameters: {
            model: described_class::DEFAULTS[:completion_model_name],
            prompt: completion,
            temperature: described_class::DEFAULTS[:temperature],
            max_tokens_to_sample: described_class::DEFAULTS[:max_tokens_to_sample]
          })
          .and_return(response)
      end

      it "returns a completion" do
        expect(subject.complete(prompt: completion).completion).to eq(" The sky has no definitive")
      end

      it "returns model attribute" do
        expect(subject.complete(prompt: completion).model).to eq("claude-2.1")
      end
    end

    context "with additional parameters" do
      before do
        allow(subject.client).to receive(:complete)
          .with(parameters: {
            model: "claude-3-opus-20240229",
            prompt: completion,
            temperature: 1,
            stop_sequences: ["stop"],
            max_tokens_to_sample: 100,
            metadata: {beep: "beep"},
            stream: true
          })
          .and_return(response)
      end

      it "returns a completion" do
        expect(
          subject.complete(
            model: "claude-3-opus-20240229",
            prompt: completion,
            stop_sequences: ["stop"],
            temperature: 1,
            max_tokens: 100,
            metadata: {beep: "beep"},
            stream: true,
            # include unified params that are ignored to ensure they don't
            # pass through to the client
            system: "You are a meniacal robot",
            response_format: "Boop beep, ha ha",
            seed: 1,
            tools: ["?"],
            tool_choice: "?"
          ).completion
        ).to eq(" The sky has no definitive")
      end
    end
  end

  describe "#chat" do
    let(:messages) { [{role: "user", content: "How high is the sky?"}] }
    let(:fixture) { File.read("spec/fixtures/llm/anthropic/chat.json") }
    let(:response) { JSON.parse(fixture) }

    context "with no additional parameters" do
      before do
        allow(subject.client).to receive(:messages)
          .with(parameters: {
            model: described_class::DEFAULTS[:chat_completion_model_name],
            messages: messages,
            temperature: described_class::DEFAULTS[:temperature],
            max_tokens: described_class::DEFAULTS[:max_tokens_to_sample],
            stop_sequences: ["beep"]
          })
          .and_return(response)
      end

      it "returns a completion" do
        expect(
          subject.chat(messages: messages, stop_sequences: ["beep"]).chat_completion
        ).to eq("The sky doesn't have a defined height or upper limit.")
      end

      it "returns model attribute" do
        expect(
          subject.chat(messages: messages, stop_sequences: ["beep"]).model
        ).to eq("claude-3-sonnet-20240229")
      end
    end
  end
end
