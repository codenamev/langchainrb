# frozen_string_literal: true

class TestLLM < Langchain::LLM::Base
end

class CustomTestLLM < Langchain::LLM::Base
  def initialize
    chat_parameters.update(version: {default: 1})
    complete_parameters.update(version: {default: 2})
  end
end

RSpec.describe Langchain::LLM::Base do
  let(:subject) { described_class.new }

  describe "#chat" do
    it "raises an error" do
      expect { subject.chat }.to raise_error(NotImplementedError)
    end
  end

  describe "#complete" do
    it "raises an error" do
      expect { subject.complete }.to raise_error(NotImplementedError)
    end
  end

  describe "#embed" do
    it "raises an error" do
      expect { subject.embed }.to raise_error(NotImplementedError)
    end
  end

  describe "#summarize" do
    it "raises an error" do
      expect { subject.summarize }.to raise_error(NotImplementedError)
    end
  end

  describe "#chat_parameters(params = {})" do
    subject { TestLLM.new }

    it "returns an instance of ChatParameters" do
      chat_params = subject.chat_parameters
      expect(chat_params).to be_instance_of(Langchain::LLM::Parameters::Chat)
    end

    it "proxies the provided params to the UnifiedParameters" do
      chat_params = subject.chat_parameters({stream: true})
      expect(chat_params).to be_instance_of(Langchain::LLM::Parameters::Chat)
      expect(chat_params[:stream]).to be_truthy
    end

    it "does not cache between child instances" do
      expect(CustomTestLLM.new.chat_parameters.to_params).to include(version: 1)
      expect(TestLLM.new.chat_parameters.to_params).not_to include(version: 1)
    end
  end

  describe "#complete_parameters(params = {})" do
    subject { TestLLM.new }

    it "returns an instance of CompleteParameters" do
      complete_params = subject.complete_parameters
      expect(complete_params).to be_instance_of(Langchain::LLM::Parameters::Complete)
    end

    it "proxies the provided params to the UnifiedParameters" do
      complete_params = subject.complete_parameters({stream: true})
      expect(complete_params).to be_instance_of(Langchain::LLM::Parameters::Complete)
      expect(complete_params[:stream]).to be_truthy
    end

    it "does not cache between child instances" do
      expect(CustomTestLLM.new.complete_parameters.to_params).to include(version: 2)
      expect(TestLLM.new.complete_parameters.to_params).not_to include(version: 2)
    end
  end
end
