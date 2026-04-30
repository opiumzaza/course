"""Module A: collect Telegram channel comments into a raw CSV dataset."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import timezone
from pathlib import Path
from typing import TypeVar

import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError, MsgIdInvalidError, PeerIdInvalidError, RPCError
from telethon.tl.custom.message import Message

import config


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def parse_channels(channels: str | None) -> list[str]:
    """Parse channel CLI input or return configured defaults."""
    source: Iterable[str] = channels.split(",") if channels else config.TELEGRAM_CHANNELS
    parsed = [channel.strip().lstrip("@") for channel in source if channel.strip()]
    if not parsed:
        raise ValueError("At least one Telegram channel username is required.")
    return parsed


def bounded_limit(limit: int) -> int:
    """Clamp the requested post limit to the supported collection range."""
    if limit < config.MIN_POST_LIMIT:
        LOGGER.warning(
            "Requested limit %s is below %s; using %s.",
            limit,
            config.MIN_POST_LIMIT,
            config.MIN_POST_LIMIT,
        )
        return config.MIN_POST_LIMIT
    if limit > config.MAX_POST_LIMIT:
        LOGGER.warning(
            "Requested limit %s is above %s; using %s.",
            limit,
            config.MAX_POST_LIMIT,
            config.MAX_POST_LIMIT,
        )
        return config.MAX_POST_LIMIT
    return limit


async def retry_telegram_call(
    operation: Callable[[], Awaitable[T]],
    label: str,
    skip_exceptions: tuple[type[BaseException], ...] = (),
) -> T | None:
    """Run a Telegram API operation with retry and rate-limit handling.

    `skip_exceptions` lets a specific call site declare exception types that
    are expected, permanent, and must not be retried (e.g. comment fetching
    against posts with no discussion thread). They are logged at DEBUG and
    the call returns None. All other exceptions retain the existing retry
    and ERROR-level logging behaviour.
    """
    delay = config.RATE_LIMIT_BASE_DELAY_SECONDS
    for attempt in range(1, config.TELEGRAM_REQUEST_RETRIES + 1):
        try:
            return await operation()
        except FloodWaitError as exc:
            LOGGER.warning("FloodWaitError during %s: waiting %s seconds.", label, exc.seconds)
            await asyncio.sleep(exc.seconds)
            if not ask_continue_after_flood_wait(label, exc.seconds):
                LOGGER.warning("User chose to stop after Telegram flood wait during %s.", label)
                return None
        except skip_exceptions as exc:
            LOGGER.debug(
                "Skipping %s due to expected %s.",
                label,
                exc.__class__.__name__,
            )
            return None
        except (OSError, ConnectionError, TimeoutError, RPCError) as exc:
            if attempt == config.TELEGRAM_REQUEST_RETRIES:
                LOGGER.error("Telegram operation failed after %s attempts: %s", attempt, label)
                return None
            LOGGER.warning(
                "Telegram operation failed (%s/%s) for %s: %s. Retrying in %s seconds.",
                attempt,
                config.TELEGRAM_REQUEST_RETRIES,
                label,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2
    return None


def ask_continue_after_flood_wait(label: str, waited_seconds: int) -> bool:
    """Ask whether collection should continue after a Telegram flood wait."""
    prompt = (
        f"Telegram requested a {waited_seconds}s wait during {label}. "
        "Continue collection? [y/N]: "
    )
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        LOGGER.warning("No interactive input available after flood wait; continuing safely.")
        return True


async def collect_comments_for_channel(
    client: TelegramClient,
    channel: str,
    limit: int,
) -> list[dict[str, object]]:
    """Collect recent post comments for one public Telegram channel."""
    records: list[dict[str, object]] = []

    async def get_entity() -> object:
        """Resolve the configured channel username to a Telethon entity."""
        return await client.get_entity(channel)

    channel_entity = await retry_telegram_call(get_entity, f"resolve @{channel}")
    if channel_entity is None:
        return records

    post_count = 0
    posts_with_comments = 0
    try:
        async for post in client.iter_messages(channel_entity, limit=limit):
            if not isinstance(post, Message) or post.id is None:
                continue
            post_count += 1
            if not post_has_comment_thread(post):
                continue
            posts_with_comments += 1
            comments = await collect_comments_for_post(client, channel_entity, channel, post)
            records.extend(comments)
    except FloodWaitError as exc:
        LOGGER.warning("FloodWaitError while listing posts for %s: waiting %s seconds.", channel, exc.seconds)
        await asyncio.sleep(exc.seconds)
        ask_continue_after_flood_wait(f"listing posts for @{channel}", exc.seconds)
    except (OSError, ConnectionError, TimeoutError, RPCError) as exc:
        LOGGER.error("Could not list posts for %s after %s posts: %s", channel, post_count, exc)

    if post_count > 0 and posts_with_comments == 0:
        LOGGER.warning(
            "@%s exposed %s posts but none had a reachable comment thread. "
            "The channel likely has comments disabled or no linked discussion group.",
            channel,
            post_count,
        )
    print(f"[A] Collected {len(records)} comments from {channel}")
    return records


def post_has_comment_thread(post: Message) -> bool:
    """Return whether a channel post advertises a reachable comment thread."""
    replies = getattr(post, "replies", None)
    if replies is None:
        return False
    if not getattr(replies, "comments", False):
        return False
    return int(getattr(replies, "replies", 0) or 0) > 0


async def collect_comments_for_post(
    client: TelegramClient,
    channel_entity: object,
    channel: str,
    post: Message,
) -> list[dict[str, object]]:
    """Collect comments attached to a single Telegram channel post."""
    records: list[dict[str, object]] = []
    post_timestamp = to_utc_iso(post.date)

    if not post_has_comment_thread(post):
        return records

    async def list_comments() -> list[Message]:
        """Fetch all available comments for one post."""
        return [
            comment
            async for comment in client.iter_messages(channel_entity, reply_to=post.id)
            if isinstance(comment, Message)
        ]

    comments = await retry_telegram_call(
        list_comments,
        f"comments for @{channel}/{post.id}",
        skip_exceptions=(PeerIdInvalidError, MsgIdInvalidError),
    )
    if comments is None:
        return records

    for comment in comments:
        if comment.sender_id is None:
            continue
        records.append(
            {
                "user_id": str(comment.sender_id),
                "message_text": comment.message or "",
                "timestamp": to_utc_iso(comment.date),
                "post_id": str(post.id),
                "reply_to_msg_id": str(comment.reply_to_msg_id or post.id),
                "channel": channel,
                "post_timestamp": post_timestamp,
            }
        )
    return records


def to_utc_iso(value: object) -> str:
    """Convert Telethon datetime values to UTC ISO-8601 strings."""
    if hasattr(value, "astimezone"):
        return value.astimezone(timezone.utc).isoformat()
    raise ValueError(f"Expected datetime-like value, received {type(value)!r}.")


async def collect_all(limit: int = config.DEFAULT_POST_LIMIT, channels: str | None = None) -> Path:
    """Collect comments for all requested channels and save raw CSV output."""
    config.configure_logging()
    config.ensure_directories()
    config.validate_telegram_credentials()
    selected_channels = parse_channels(channels)
    post_limit = bounded_limit(limit)

    client = TelegramClient(
        str(config.BASE_DIR / "telegram_session"),
        config.get_api_id(),
        str(config.TELEGRAM_API_HASH),
    )

    all_records: list[dict[str, object]] = []
    async with client:
        if not await client.is_user_authorized():
            await client.start(phone=str(config.TELEGRAM_PHONE))
        for channel in selected_channels:
            all_records.extend(await collect_comments_for_channel(client, channel, post_limit))

    frame = pd.DataFrame(
        all_records,
        columns=[
            "user_id",
            "message_text",
            "timestamp",
            "post_id",
            "reply_to_msg_id",
            "channel",
            "post_timestamp",
        ],
    )
    frame.to_csv(config.RAW_COMMENTS_PATH, index=False)
    return config.RAW_COMMENTS_PATH


def build_parser() -> argparse.ArgumentParser:
    """Build the Module A command-line parser."""
    parser = argparse.ArgumentParser(description="Collect Telegram comments for bot-farm analysis.")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_POST_LIMIT, help="Max posts per channel.")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channel usernames.")
    return parser


def main() -> None:
    """Run Telegram collection from the command line."""
    args = build_parser().parse_args()
    config.configure_logging()
    config.confirm_overwrite_runtime_outputs()
    output_path = asyncio.run(collect_all(limit=args.limit, channels=args.channels))
    print(f"Module A output saved → {output_path}")


if __name__ == "__main__":
    main()
