from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    backend: str = "mock"
    instances: int = 2
    repeats: int = 2
    seed: int = 42
    rounds: int = 4
    max_calls: int = 8
    max_output_tokens: int = 256
    reported_token_limit: int = 12000
    max_message_chars: int = 1200
    timeout: float = 45.0
    temperature: float = 0.0
    token_parameter: str = "max_tokens"

    def validate(self):
        for name in ("instances", "repeats", "rounds", "max_calls", "max_output_tokens",
                     "reported_token_limit", "max_message_chars"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.timeout <= 0 or not 0 <= self.temperature <= 2:
            raise ValueError("Invalid timeout or temperature")
        if self.backend not in ("mock", "api"):
            raise ValueError("Unknown backend")
        if self.token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("Invalid token parameter")
