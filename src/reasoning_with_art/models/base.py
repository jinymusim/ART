from abc import ABC, abstractmethod

MODEL_REGISTRY: dict[str, type["ModelClient"]] = {}


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


class ModelClient(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    def generate(self, prompt: str, image: bytes | None = None) -> str:
        """Generate a response for a single prompt. Delegates to generate_batch."""
        results = self.generate_batch([prompt], images=[image])
        return results[0]

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        images: list[bytes | None] | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate responses for a batch of prompts.

        ``max_new_tokens`` optionally overrides the client's default output
        budget for this call (per-benchmark eval override); None = default.
        """
        ...
