# frozen_string_literal: true

require "cohere"

RSpec.describe Langchain::LLM::Cohere do
  let(:subject) { described_class.new(api_key: "123") }

  describe "#embed" do
    before do
      allow(subject.client).to receive(:embed).and_return(
        {
          "id" => "a86a12ca-7ce5-4433-b68a-4d8454b22de7",
          "texts" => ["Hello World"],
          "embeddings" => [[-1.5693359, -0.9458008, 1.9355469]]
        }
      )
    end

    it "returns an embedding" do
      expect(subject.embed(text: "Hello World").embedding).to eq([-1.5693359, -0.9458008, 1.9355469])
    end
  end

  describe "#complete" do
    before do
      allow(subject.client).to receive(:generate).and_return(
        {
          "id" => "812c650e-a0d0-4502-a084-45b0d32fcb9c",
          "generations" => [
            {
              "id" => "8b79fd4f-7c72-4e1d-97a1-3dbe49206db2",
              "text" => "\nWhat is the meaning of life? What is the meaning of life?\nWhat is the meaning"
            }
          ],
          "prompt" => "What is the meaining of life?",
          "meta" => {"api_version" => {"version" => "1"}}
        }
      )

      allow(subject.client).to receive(:tokenize).and_return(
        {
          "tokens" => [
            33555,
            1114,
            34
          ],
          "token_strings" => [
            "hello",
            " world",
            "!"
          ],
          "meta" => {
            "api_version" => {
              "version" => "1"
            }
          }
        }
      )
    end

    it "returns a completion" do
      expect(subject.complete(prompt: "Hello World").completion).to eq("\nWhat is the meaning of life? What is the meaning of life?\nWhat is the meaning")
    end

    context "with custom default_options" do
      let(:subject) {
        described_class.new(
          api_key: "123",
          default_options: {completion_model_name: "base-light"}
        )
      }

      it "passes correct options to the completions method" do
        expect(subject.client).to receive(:generate).with(
          {
            max_tokens: 2045,
            model: "base-light",
            prompt: "Hello World",
            temperature: 0.0,
            truncate: "NONE",
            num_generations: 3,
            stop_sequences: ["."],
            k: 1.0,
            p: 0.5,
            seed: 5,
            preset: "author",
            end_sequences: ["\n"],
            frequency_penalty: 1,
            presence_penalty: 0.1,
            return_likelihoods: "ALL",
            raw_prompting: true
          }
        )
        subject.complete(
          prompt: "Hello World",
          n: 3,
          stop: ["."],
          top_k: 1.0,
          p: 0.5,
          truncate: "NONE",
          seed: 5,
          preset: "author",
          end_sequences: ["\n"],
          frequency_penalty: 1,
          presence_penalty: 0.1,
          return_likelihoods: "ALL",
          raw_prompting: true
        )
      end
    end
  end

  describe "#chat" do
    let(:fixture) { File.read("spec/fixtures/llm/cohere/chat.json") }
    let(:response) { JSON.parse(fixture) }

    before do
      allow(subject.client).to receive(:chat)
        .with(
          model: "command-r-plus",
          temperature: 0.0,
          preamble: "You are a cheerful happy chatbot!",
          chat_history: [{role: "user", message: "How are you?"}]
        )
        .and_return(response)
    end

    it "returns a response" do
      expect(
        subject.chat(
          system: "You are a cheerful happy chatbot!",
          messages: [{role: "user", message: "How are you?"}]
        )
      ).to be_a(Langchain::LLM::CohereResponse)
    end
  end

  describe "#default_dimensions" do
    it "returns the default dimensions" do
      expect(subject.default_dimensions).to eq(1024)
    end
  end

  describe "#summarize" do
    let(:text) { "Text to summarize" }

    before do
      allow(subject.client).to receive(:summarize).and_return(
        {
          "id" => "123",
          "summary" => "Summary",
          "meta" => {"api_version" => {"version" => "1"}}
        }
      )
    end

    it "returns a summary" do
      expect(subject.summarize(text: text)).to eq("Summary")
    end
  end
end
