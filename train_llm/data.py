from pathlib import Path
from typing import Any, Optional
from utils import LOG


def content_to_text(content: Any, image_placeholder: str = "[IMAGE]") -> str:
    """
    Convert either plain string content or typed multimodal content blocks to text.

    Text-only models generally expect message["content"] to be a string. This lets
    the raw dataset use the richer block format while preserving the existing
    text-only trainer path.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        raise ValueError(f"message content must be a string or list of blocks, got {type(content)}.")

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("content blocks must be objects.")

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise ValueError("text content block must have a string 'text' field.")
            parts.append(text)
        elif block_type == "image":
            parts.append(image_placeholder)
        else:
            raise ValueError(f"unsupported content block type: {block_type!r}")

    return "\n".join(part for part in parts if part)


def normalize_messages_for_text(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **message,
            "content": content_to_text(message["content"]),
        }
        for message in messages
    ]


def messages_to_prompt_completion(
    example: dict[str, Any],
    messages_field: str,
    do_validate: bool,
) -> dict[str, Any]:
    """
    Convert a conversational example with a list of messages into a dict with 'prompt' and 'completion' fields.
    The prompt contains all messages except the last one, and the completion contains only the last message, which must be from the assistant.
    """
    messages = example[messages_field]

    if do_validate:
        validate_messages(messages, allow_images=True)

    if messages[-1]["role"] != "assistant":
        raise ValueError("Last message must be assistant.")

    normalized_messages = normalize_messages_for_text(messages)
    prompt = normalized_messages[:-1]
    completion = [normalized_messages[-1]]

    return {
        "prompt": prompt,
        "completion": completion,
    }


def resolve_image_path(path: str, image_root: Optional[str]) -> Path:
    image_path = Path(path)
    if image_path.is_absolute():
        return image_path
    if image_root:
        return Path(image_root) / image_path
    return image_path


def load_image(path: str, image_root: Optional[str]) -> Any:
    from PIL import Image

    resolved = resolve_image_path(path, image_root)
    return Image.open(resolved).convert("RGB")


def content_to_vlm_blocks(content: Any, image_root: Optional[str], images: list[Any]) -> list[dict[str, Any]]:
    """
    Convert raw content to TRL-style VLM blocks.

    Raw image blocks use {"type": "image", "path": "..."} so the dataset records
    where each image belongs. Trainer-facing blocks use {"type": "image"} and the
    loaded image is appended to the parallel images list in the same order.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        raise ValueError(f"message content must be a string or list of blocks, got {type(content)}.")

    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("content blocks must be objects.")

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise ValueError("text content block must have a string 'text' field.")
            blocks.append({"type": "text", "text": text})
        elif block_type == "image":
            image_obj = block.get("image")
            image_path = block.get("path")

            if image_obj is not None:
                images.append(image_obj)
            elif isinstance(image_path, str):
                images.append(load_image(image_path, image_root))
            else:
                raise ValueError("image content block must have either an 'image' object or a string 'path'.")

            blocks.append({"type": "image"})
        else:
            raise ValueError(f"unsupported content block type: {block_type!r}")

    return blocks


def messages_to_vlm_prompt_completion(
    example: dict[str, Any],
    messages_field: str,
    do_validate: bool,
    image_root: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert conversational examples to prompt/completion plus TRL-style images.

    The raw format may mix text and image blocks in message content. Image blocks
    should normally be {"type": "image", "path": "relative/or/absolute.jpg"}.
    """
    messages = example[messages_field]

    if do_validate:
        validate_messages(messages, allow_images=True)

    if messages[-1]["role"] != "assistant":
        raise ValueError("Last message must be assistant.")

    images: list[Any] = []
    normalized_messages = [
        {
            **message,
            "content": content_to_vlm_blocks(message["content"], image_root, images),
        }
        for message in messages
    ]

    return {
        "prompt": normalized_messages[:-1],
        "completion": [normalized_messages[-1]],
        "images": images,
    }


def validate_messages(messages: Any, allow_images: bool = False) -> None:
    """
    Checks that the messages list is a valid conversational format for SFT.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")

    allowed_roles = {"system", "user", "assistant", "tool"}

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {idx} is not an object.")
        role = message.get("role")
        if role not in allowed_roles:
            raise ValueError(f"message {idx} has unsupported role: {role!r}")
        if "content" not in message:
            raise ValueError(f"message {idx} is missing content.")
        validate_content(message["content"], allow_images=allow_images, message_idx=idx)

    if messages[-1].get("role") != "assistant":
        raise ValueError("last message must be an assistant message for SFT.")


def validate_content(content: Any, allow_images: bool, message_idx: int) -> None:
    if isinstance(content, str):
        return

    if not isinstance(content, list):
        raise ValueError(f"message {message_idx} content must be a string or a list of blocks.")

    for block_idx, block in enumerate(content):
        if not isinstance(block, dict):
            raise ValueError(f"message {message_idx} content block {block_idx} is not an object.")

        block_type = block.get("type")
        if block_type == "text":
            if not isinstance(block.get("text"), str):
                raise ValueError(
                    f"message {message_idx} content block {block_idx} must have a string 'text' field."
                )
        elif block_type == "image":
            if not allow_images:
                raise ValueError(
                    f"message {message_idx} content block {block_idx} is an image, "
                    "but data.format is text-only."
                )
            if "path" not in block and "image" not in block:
                raise ValueError(
                    f"message {message_idx} content block {block_idx} must have 'path' or 'image'."
                )
        else:
            raise ValueError(f"message {message_idx} content block {block_idx} has unsupported type: {block_type!r}")


class TokenCountLoggingCollator:
    """
    Wraps an existing data collator and logs token-count statistics.

    Works after SFTTrainer has constructed its own collator.
    """

    def __init__(
        self,
        base_collator: Any,
        clearml_task: Optional[Any] = None,
        log_every: int = 100,
        log_prefix: str = "tokens",
    ):
        self.base_collator = base_collator
        self.clearml_task = clearml_task
        self.log_every = int(log_every)
        self.log_prefix = log_prefix
        self.call_idx = 0
        self.logger = clearml_task.get_logger() if clearml_task is not None else None

    def __call__(self, features):
        batch = self.base_collator(features)
        self.call_idx += 1

        if self.log_every <= 0 or self.call_idx % self.log_every != 0:
            return batch

        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")

        if attention_mask is None or labels is None:
            return batch

        input_tokens = attention_mask.sum().item()
        total_positions = attention_mask.numel()

        supervised_tokens = labels.ne(-100).sum().item()
        ignored_tokens = max(input_tokens - supervised_tokens, 0)
        padding_tokens = max(total_positions - input_tokens, 0)

        batch_size = labels.shape[0]
        seq_len = labels.shape[1] if labels.ndim >= 2 else labels.numel()

        supervised_ratio = supervised_tokens / input_tokens if input_tokens else 0.0
        avg_input_tokens = input_tokens / batch_size if batch_size else 0.0
        avg_supervised_tokens = supervised_tokens / batch_size if batch_size else 0.0

        LOG.info(
            "Token counts | batch=%d | input=%d | supervised=%d | ignored=%d | padding=%d | "
            "supervised_ratio=%.4f | avg_input=%.1f | avg_supervised=%.1f | shape=(%d,%d)",
            self.call_idx,
            input_tokens,
            supervised_tokens,
            ignored_tokens,
            padding_tokens,
            supervised_ratio,
            avg_input_tokens,
            avg_supervised_tokens,
            batch_size,
            seq_len,
        )

        if self.logger is not None:
            step = self.call_idx
            self.logger.report_scalar(self.log_prefix, "input_tokens", input_tokens, iteration=step)
            self.logger.report_scalar(self.log_prefix, "supervised_tokens", supervised_tokens, iteration=step)
            self.logger.report_scalar(self.log_prefix, "ignored_tokens", ignored_tokens, iteration=step)
            self.logger.report_scalar(self.log_prefix, "padding_tokens", padding_tokens, iteration=step)
            self.logger.report_scalar(self.log_prefix, "supervised_ratio", supervised_ratio, iteration=step)
            self.logger.report_scalar(self.log_prefix, "avg_input_tokens", avg_input_tokens, iteration=step)
            self.logger.report_scalar(self.log_prefix, "avg_supervised_tokens", avg_supervised_tokens, iteration=step)

        return batch
    
    
def inspect_first_batch(trainer: Any, tokenizer: Any) -> None:
    import torch

    LOG.info("Inspecting first train dataloader batch.")
    batch = next(iter(trainer.get_train_dataloader()))
    LOG.info("First batch keys: %s", sorted(batch.keys()))

    labels = batch.get("labels")
    input_ids = batch.get("input_ids")

    if labels is None:
        raise RuntimeError("Batch has no labels. SFTTrainer did not prepare labels.")

    labels_tensor = labels if torch.is_tensor(labels) else torch.tensor(labels)
    supervised = labels_tensor.ne(-100).sum().item()
    total = labels_tensor.numel()
    pct = 100.0 * supervised / total if total else 0.0

    LOG.info("Supervised labels: %d / %d (%.4f%%)", supervised, total, pct)

    if supervised == 0:
        raise RuntimeError(
            "All labels are -100. assistant_only_loss masking failed. "
            "Check that the model chat template has {% generation %} markers or set training.chat_template_path."
        )

    if input_ids is not None:
        decode_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        input_tensor = input_ids if torch.is_tensor(input_ids) else torch.tensor(input_ids)
        first_ids = input_tensor[0].detach().cpu().tolist()
        label_ids = labels_tensor[0].detach().cpu().tolist()

        decoded_input = decode_tokenizer.decode(first_ids[:512], skip_special_tokens=False)

        supervised_token_ids = [
            token_id
            for token_id, label_id in zip(first_ids, label_ids)
            if label_id != -100
        ]
        decoded_supervised = decode_tokenizer.decode(supervised_token_ids[:256], skip_special_tokens=False)

        LOG.info("First example rendered prefix:\n%s", decoded_input)
        LOG.info("First example supervised-token prefix:\n%s", decoded_supervised)
