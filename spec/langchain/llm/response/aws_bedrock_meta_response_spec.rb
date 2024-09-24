# frozen_string_literal: true

RSpec.describe Langchain::LLM::AwsBedrockMetaResponse do
  let(:raw_chat_completions_response) {
    JSON.parse File.read("spec/fixtures/llm/aws_bedrock_meta/complete.json")
  }

  subject { described_class.new(raw_chat_completions_response) }

  describe "#complete" do
    it "returns completion" do
      expect(subject.completion).to eq("The sky has no definitive")
    end
  end
end
