import json
from datetime import datetime, timedelta

class Model:
    def __init__(self, name: str, requests_per_minute: int, requests_per_day: int, ranking: int, top_k: int):
        """Initialize a model with rate limits and parameters."""
        self.name = name
        self.requests_per_minute_limit = requests_per_minute
        self.requests_per_day_limit = requests_per_day
        self.ranking = ranking
        self.top_k = top_k  # Model-specific parameter (e.g., for API calls)

        # Usage counters
        self.current_minute_usage = 0
        self.current_day_usage = 0

        # Reset timestamps
        self.last_minute_reset = datetime.now()
        self.last_day_reset = datetime.now()

        # Track 429 errors
        self.resource_exhausted_count = 0

    def reset_minute_usage(self):
        """Reset minute usage and update timestamp."""
        self.current_minute_usage = 0
        self.last_minute_reset = datetime.now()

    def reset_day_usage(self):
        """Reset day usage and update timestamp."""
        self.current_day_usage = 0
        self.last_day_reset = datetime.now()

    def update_usage_if_needed(self):
        """Reset usage counters if time period has elapsed."""
        now = datetime.now()
        if now - self.last_minute_reset >= timedelta(minutes=1):
            self.reset_minute_usage()
        if now - self.last_day_reset >= timedelta(days=1):
            self.reset_day_usage()

    def can_make_request(self) -> bool:
        """Check if the model can make a request based on current usage."""
        self.update_usage_if_needed()
        return (self.current_minute_usage < self.requests_per_minute_limit and
                self.current_day_usage < self.requests_per_day_limit)

    def increment_usage(self):
        """Increment usage counters after a successful request."""
        self.update_usage_if_needed()
        self.current_minute_usage += 1
        self.current_day_usage += 1

    def available_requests(self) -> int:
        """Return the minimum number of remaining requests."""
        self.update_usage_if_needed()
        remaining_minute = self.requests_per_minute_limit - self.current_minute_usage
        remaining_day = self.requests_per_day_limit - self.current_day_usage
        return min(remaining_minute, remaining_day)

    def __repr__(self):
        return (f"Model(name={self.name}, "
                f"minute_usage={self.current_minute_usage}/{self.requests_per_minute_limit}, "
                f"day_usage={self.current_day_usage}/{self.requests_per_day_limit}, "
                f"resource_exhausted_count={self.resource_exhausted_count}, "
                f"ranking={self.ranking}, "
                f"top_k={self.top_k})")


class ModelManager:
    def __init__(self, json_file: str):
        """Initialize with a JSON file containing model details."""
        self.models = []
        self.load_models(json_file)

    def load_models(self, json_file: str):
        """Load models from JSON file, including top_k."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Expecting a list of models; if JSON has a 'models' key, adjust accordingly
        model_list = data if isinstance(data, list) else data.get("models", [])
        for model_data in model_list:
            model = Model(
                name=model_data["name"],
                requests_per_minute=model_data["requests_per_minute"],
                requests_per_day=model_data["requests_per_day"],
                ranking=model_data["ranking"],
                top_k=model_data["top_k"]
            )
            self.models.append(model)

    def get_sorted_available_models(self, exclude_model_name: str = None):
        """Return sorted list of models that can make requests."""
        available_models = [
            m for m in self.models
            if m.can_make_request() and (exclude_model_name is None or m.name != exclude_model_name)
        ]
        available_models.sort(key=lambda m: (
            -(m.requests_per_day_limit - m.current_day_usage),      # Max daily remaining
            -(m.requests_per_minute_limit - m.current_minute_usage), # Max minute remaining
            m.resource_exhausted_count,                              # Min 429 errors
            m.ranking                                               # Lower ranking
        ))
        return available_models

    def get_available_model(self) -> Model:
        """Get the best available model or None if all are exhausted."""
        for model in self.models:
            model.update_usage_if_needed()
        sorted_models = self.get_sorted_available_models()
        return sorted_models[0] if sorted_models else None

    def increment_request(self, model_name: str):
        """Increment usage for a model after a successful request."""
        model = self.get_model_by_name(model_name)
        if model:
            model.increment_usage()
        else:
            raise ValueError(f"Model with name {model_name} not found.")

    def swap_model(self, current_model_name: str) -> Model:
        """Handle a 429 error by marking the current model exhausted and selecting another."""
        # Update all models' usage first
        for model in self.models:
            model.update_usage_if_needed()

        # Mark the current model as exhausted
        current_model = self.get_model_by_name(current_model_name)
        if current_model:
            current_model.resource_exhausted_count += 1
            # Set usage to limits to prevent reuse until reset
            current_model.current_minute_usage = current_model.requests_per_minute_limit
            current_model.current_day_usage = current_model.requests_per_day_limit

        # Select an alternative model
        alternative_models = self.get_sorted_available_models(exclude_model_name=current_model_name)
        if alternative_models:
            return alternative_models[0]
        # If no alternatives, check current model (unlikely to be available immediately)
        elif current_model and current_model.can_make_request():
            return current_model
        else:
            return None

    def get_model_by_name(self, model_name: str) -> Model:
        """Retrieve a model by name."""
        for model in self.models:
            if model.name == model_name:
                return model
        return None
