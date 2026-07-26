"""Streaming chat client for your local llama-server.

This is the "how does an application actually consume a model" piece. There is
nothing model-specific in it: llama-server speaks the OpenAI API, so the only
thing separating this from code that talks to a frontier model is the base URL.
Point ``--base-url`` elsewhere and this script keeps working.

Usage:
    python clients/chat_cli.py
    python clients/chat_cli.py --temperature 1.0
    python clients/chat_cli.py --no-stream

Commands inside the chat: /reset, /system <text>, /temp <float>, /stats, /exit
"""

from __future__ import annotations

import argparse
import sys
import time

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

try:
    from tinyllm.config import gen_cfg, serve_cfg
    DEFAULT_BASE_URL = serve_cfg.base_url
    DEFAULT_MODEL = serve_cfg.served_model_name
    DEFAULT_TEMP = gen_cfg.temperature
    DEFAULT_MAX_TOKENS = gen_cfg.max_new_tokens
except ImportError:  # running outside the venv
    DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
    DEFAULT_MODEL = "tinyllm"
    DEFAULT_TEMP = 0.8
    DEFAULT_MAX_TOKENS = 256

console = Console()

BANNER = """[bold]tinyllm[/bold] — chatting with a model you trained from scratch.

It writes children's stories. That is genuinely all it does; anything else
produces confident nonsense. Try:
  [dim]Write a story about a lost puppy who finds its way home.[/dim]
  [dim]Write a short story using the words: ball, tree, happy.[/dim]

[dim]/reset  /system <text>  /temp <float>  /stats  /exit[/dim]"""


def check_server(client: OpenAI, model: str) -> bool:
    """Confirm the server is up before dropping the user into a prompt."""
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        if model not in ids and ids:
            console.print(f"[yellow]Note:[/yellow] server advertises {ids}, not {model!r}. "
                          f"Using {ids[0]!r} instead.")
            return ids[0]
        return model
    except Exception as e:
        console.print(Panel(
            f"[red]Cannot reach {client.base_url}[/red]\n\n"
            f"{type(e).__name__}: {e}\n\n"
            "Start the server first:\n"
            "  [bold].\\scripts\\serve.ps1[/bold]",
            title="server unreachable", border_style="red",
        ))
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--no-stream", action="store_true", help="wait for the full reply")
    args = ap.parse_args()

    # llama-server ignores the key, but the SDK requires one to be set.
    client = OpenAI(base_url=args.base_url, api_key="none")

    model = check_server(client, args.model)
    if model is None:
        return 1

    console.print(Panel(BANNER, border_style="cyan"))
    console.print(f"[dim]{args.base_url}  ·  model={model}  ·  temp={args.temperature}[/dim]\n")

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    last_stats = None

    while True:
        try:
            user = console.input("[bold cyan]you[/bold cyan] › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return 0

        if not user:
            continue

        # ---- commands ------------------------------------------------------
        if user in ("/exit", "/quit"):
            console.print("[dim]bye[/dim]")
            return 0
        if user == "/reset":
            messages = [m for m in messages if m["role"] == "system"]
            console.print("[dim]history cleared[/dim]\n")
            continue
        if user.startswith("/system "):
            messages = [{"role": "system", "content": user[8:].strip()}] + \
                       [m for m in messages if m["role"] != "system"]
            console.print("[dim]system prompt set[/dim]\n")
            continue
        if user.startswith("/temp "):
            try:
                args.temperature = float(user.split()[1])
                console.print(f"[dim]temperature = {args.temperature}[/dim]\n")
            except (IndexError, ValueError):
                console.print("[red]usage: /temp 0.8[/red]\n")
            continue
        if user == "/stats":
            console.print(f"[dim]{last_stats or 'no requests yet'}[/dim]\n")
            continue

        messages.append({"role": "user", "content": user})

        # ---- request -------------------------------------------------------
        t0 = time.time()
        n_tokens = 0
        try:
            if args.no_stream:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
                reply = resp.choices[0].message.content or ""
                n_tokens = resp.usage.completion_tokens if resp.usage else 0
                console.print("[bold green]tinyllm[/bold green] › ", end="")
                console.print(Markdown(reply))
            else:
                stream = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    stream=True,
                )
                console.print("[bold green]tinyllm[/bold green] › ", end="")
                chunks: list[str] = []
                for event in stream:
                    if not event.choices:
                        continue
                    piece = event.choices[0].delta.content
                    if piece:
                        chunks.append(piece)
                        n_tokens += 1
                        # Print raw during streaming; markdown rendering needs
                        # the whole document and would fight the incremental output.
                        console.print(piece, end="", markup=False, highlight=False)
                reply = "".join(chunks)
                console.print()
        except KeyboardInterrupt:
            console.print("\n[dim]interrupted[/dim]\n")
            messages.pop()
            continue
        except Exception as e:
            console.print(f"\n[red]{type(e).__name__}: {e}[/red]\n")
            messages.pop()
            continue

        dt = time.time() - t0
        last_stats = f"{n_tokens} tokens in {dt:.2f}s ({n_tokens / max(dt, 1e-6):.1f} tok/s)"
        console.print(f"[dim]{last_stats}[/dim]\n")

        messages.append({"role": "assistant", "content": reply})

        # A 512-token context fills up fast. Drop the oldest exchanges rather
        # than letting the server silently truncate the front of the prompt --
        # which would cut the system message first, the one thing worth keeping.
        if len(messages) > 9:
            system = [m for m in messages if m["role"] == "system"]
            messages = system + messages[-8:]


if __name__ == "__main__":
    sys.exit(main())
