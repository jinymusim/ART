from pathlib import Path


def load_image(path: str) -> bytes:
    """Read an image file and return raw bytes."""
    return Path(path).read_bytes()


class ImageProvider:
    """Provides image bytes for each example based on a strategy.

    Strategies:
      - "none": no image
      - "learned": load PNG bytes from `image_path` once and reuse for every call.
    """

    def __init__(
        self,
        strategy: str = "learned",
        image_path: str | None = None,
        width: int = 256,
        height: int = 256,
    ):
        self._strategy = strategy
        self._image_bytes: bytes | None = None
        self._width = width
        self._height = height

        if strategy == "learned":
            if image_path is None:
                raise ValueError("image_path is required for 'learned' strategy")
            self._image_bytes = load_image(image_path)
        elif strategy == "none":
            self._image_bytes = None
        else:
            raise ValueError(f"Unknown image strategy: {strategy}")

    def get_image(self, example_idx: int = 0) -> bytes | None:
        """Return image bytes for the given example index, or None if no image."""
        if self._strategy == "learned":
            return self._image_bytes
        return None
