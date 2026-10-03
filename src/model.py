from ollama import Client

# Initialize the local Ollama client
client = Client(host="http://localhost:11434")


class OllamaPipelineWrapper:
    """Wrapper class to mimic the interface of LangChain/HuggingFace pipeline.
    
    Allows calls via both `qwen.invoke(prompt)` and `qwen(prompt)` using
    the local Ollama instance with qwen2.5:14b.
    """

    def __init__(self, model_name="qwen2.5:14b"):
        self.model_name = model_name

    def invoke(self, prompt: str) -> str:
        """Process prompt request using Ollama API.

        Args:
            prompt (str): Input text prompt for the model.

        Returns:
            str: Generated text response from the model.
        """
        response = client.generate(
            model=self.model_name,
            prompt=str(prompt),
            options={
                "temperature": 0.0,  # Zero temperature for deterministic output
                "top_p": 0.1,
                "num_predict": 1000,
                "seed": 42,
            },
        )
        return response["response"].strip()

    def __call__(self, prompt: str) -> str:
        """Allow direct callable syntax: qwen(prompt)."""
        return self.invoke(prompt)


# Global qwen pipeline object initialized with Qwen 2.5 14B
qwen = OllamaPipelineWrapper(model_name="qwen2.5:14b")