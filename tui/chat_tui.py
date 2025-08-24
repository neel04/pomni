import multiprocessing as mp
import os

import keras
import keras_hub
from rich.align import Align
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static


class PrintConsole(RichLog):
    """A RichLog subclass that captures stdout/stderr via Textual events.Print."""

    def __init__(self, **kwargs):
        super().__init__(highlight=True, markup=True, **kwargs)

    def on_print(self, event: events.Print) -> None:  # type: ignore[override]
        style = "red" if event.stderr else "dim"
        self.write(Text(event.text.rstrip("\n"), style=style))


class ChatMessage(Static):
    """A single chat message widget."""

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def render(self):
        if self.role == "user":
            text = Text(self.content, style="bold cyan")
            return Panel(
                text,
                title="[bold orange3]You[/bold orange3]",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
            )
        else:
            # Render assistant content as Markdown with syntax highlighting when possible
            content = self.content
            try:
                has_md = (
                    ("```" in content)
                    or ("#" in content)
                    or ("*" in content and "**" in content)
                    or ("- " in content)
                    or ("1." in content)
                )
                if has_md:
                    renderable = Markdown(content, code_theme="monokai")
                else:
                    renderable = Text(content, style="green")
            except Exception:
                renderable = Text(content, style="green")
            return Panel(
                renderable,
                title="[bold orange3]Pomni[/bold orange3]",
                title_align="left",
                border_style="green",
                padding=(0, 1),
            )


class StatusBar(Static):
    """Status bar for displaying model loading status."""

    status_text = reactive("Initializing...")

    def render(self):
        return Panel(
            Align.center(self.status_text, vertical="middle"),
            border_style="dim",
            height=3,
        )


class ChatContainer(ScrollableContainer):
    """Container for chat messages."""

    def compose(self) -> ComposeResult:
        yield Static(
            Panel(
                Align.center(
                    "[bold orange3]✨ Welcome to Pomni Chat ✨[/bold orange3]\n"
                    "[dim]Chat with a fine-tuned Gemma model[/dim]",
                    vertical="middle",
                ),
                border_style="orange3",
                padding=1,
            ),
            id="welcome",
        )


class PomniChatTUI(App):
    """A TUI chatbot application using Gemma model."""

    CSS = """
    /* App-wide polish */
    Screen { background: $background; }
    Header { background: $panel; color: $foreground; border: none; }
    Footer { background: $panel; color: $foreground; }

    .body { layout: vertical; height: 100%; }

    ChatContainer {
        height: 1fr;
        border: round $primary 20%;
        margin: 1;
        padding: 1;
        background: $surface;
        scrollbar-color: $scrollbar;
        scrollbar-background: $scrollbar-background;
    }
    
    /* Console area */
    PrintConsole {
        height: 6;
        border: round $panel 20%; 
        color: $text-muted;
        overflow: auto;
        margin: 0 1 1 1;
        background: $panel;
        scrollbar-color: $scrollbar;
        scrollbar-background: $scrollbar-background;
    }
    
    #user_input {
        margin: 1;
        height: 3;
    }
    Input:focus { border: solid $primary; background: $boost 5%; }
    
    StatusBar {
        height: 3;
        margin: 0 1;
    }
    
    ChatMessage { margin: 0 0 1 0; }
    
    LoadingIndicator { height: 1; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("ctrl+`", "toggle_console", "Toggle Console"),
    ]

    def __init__(self):
        super().__init__()
        self.model = None
        self.chat_history = []
        self.is_loading = True
        self.title = "Pomni Chat"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(classes="body"):
            yield ChatContainer(id="chat_container")
            yield PrintConsole(id="console")
            yield StatusBar(id="status_bar")
            yield Input(
                placeholder="Type your message here... (Press Enter to send)",
                id="user_input",
                disabled=True,
            )
        yield Footer()

    async def on_mount(self) -> None:
        """Called when app starts."""
        # Set a pleasant default theme (allow override via env)
        try:
            default_theme = os.environ.get("POMNI_THEME", "textual-dark")
            self.theme = default_theme
        except Exception:
            pass

        # Begin capturing stdout/stderr to the embedded console
        try:
            console = self.query_one("#console", PrintConsole)
            self.begin_capture_print(target=console, stdout=True, stderr=True)
        except Exception:
            # If console is not available for any reason, continue without capture
            pass
        self.load_model_async()

    def action_toggle_console(self) -> None:
        """Show / hide the diagnostics console."""
        try:
            console = self.query_one("#console", PrintConsole)
            console.display = not console.display
        except Exception:
            pass

    @work(thread=True)
    def load_model_async(self) -> None:
        """Load the model in a background thread."""
        self.update_status(
            "Loading Gemma model from HuggingFace... This may take a while."
        )

        try:
            # Load from HuggingFace only
            model = keras.saving.load_model("hf://Neel-Gupta/pomni_4B")
            self.update_status("Successfully loaded model from HuggingFace!")

            # Compile the model
            sampler = keras_hub.samplers.TopKSampler(k=50, seed=420)
            model.compile(sampler=sampler)
            self.model = model

            self.is_loading = False
            self.update_status("✅ Model loaded successfully! You can start chatting.")

            # Enable input
            input_widget = self.query_one("#user_input", Input)
            input_widget.disabled = False
            input_widget.focus()

        except Exception as e:
            # Log the error and leave input disabled
            self.update_status(f"❌ Error loading model from HuggingFace: {e}")
            self.is_loading = False

    def update_status(self, message: str) -> None:
        """Update the status bar.

        Calls synchronously if already on the app thread; otherwise schedules
        via call_from_thread() to safely cross thread boundaries.
        """
        import threading

        # Textual apps maintain an internal _thread_id for the app's thread
        if getattr(self, "_thread_id", None) == threading.get_ident():
            # We're on the app thread; update directly
            self._update_status_sync(message)
        else:
            # We're on a worker / different thread; marshal to app thread
            self.call_from_thread(self._update_status_sync, message)

    def _update_status_sync(self, message: str) -> None:
        """Synchronous status update."""
        status_bar = self.query_one("#status_bar", StatusBar)
        status_bar.status_text = message

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        if not event.value.strip() or self.is_loading or self.model is None:
            return

        user_message = event.value.strip()

        # Clear input
        input_widget = self.query_one("#user_input", Input)
        input_widget.value = ""

        # Add user message to chat
        chat_container = self.query_one("#chat_container", ChatContainer)

        # Remove welcome message if it exists
        try:
            welcome = chat_container.query_one("#welcome")
            welcome.remove()
        except Exception:
            pass

        await chat_container.mount(ChatMessage("user", user_message))
        chat_container.scroll_end(animate=True)

        # Add to history
        self.chat_history.append({"role": "user", "content": user_message})

        # Disable input while generating
        input_widget.disabled = True
        self.update_status("🤔 Thinking...")

        # Generate response in background
        self.generate_response_async(user_message)

    def _clean_control_tokens(self, text: str) -> str:
        """Remove common control/termination tokens and trim whitespace."""
        if not isinstance(text, str):
            return text
        replacements = [
            "<end_of_turn>",
            "<endofturn>",
            "<eos>",
            "<eot>",
            "</s>",
            "<|eot_id|>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|end|>",
            "[END]",
            "[EOT]",
        ]
        for tok in replacements:
            text = text.replace(tok, "")
        # Also strip any trailing XML-like tags that sometimes leak
        # e.g., <eos>, <|end|>, etc.
        import re

        text = re.sub(r"\s*(<\/?\|?\w+\|?>)+\s*$", "", text)
        return text.strip()

    @work(thread=True)
    def generate_response_async(self, prompt: str) -> None:
        """Generate model response in background thread."""
        try:
            # System prompt for nice behavior
            system_prompt = "You are an assistant. Always be concise: give short, direct answers with only essential details."
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

            # Generate response with prompt stripping and automatic stop tokens
            response = self.model.generate(
                full_prompt,
                max_length=128,
                stop_token_ids="auto",
                strip_prompt=True,
            )

            # Clean response to remove any residual control tokens
            clean_response = self._clean_control_tokens(response)

            # Add to UI
            self.call_from_thread(self.add_assistant_message, clean_response)

        except Exception as e:
            self.call_from_thread(
                self.add_assistant_message, f"Sorry, I encountered an error: {str(e)}"
            )

    async def add_assistant_message(self, message: str) -> None:
        """Add assistant message to chat."""
        chat_container = self.query_one("#chat_container", ChatContainer)
        await chat_container.mount(ChatMessage("assistant", message))
        chat_container.scroll_end(animate=True)

        # Add to history
        self.chat_history.append({"role": "assistant", "content": message})

        # Re-enable input
        input_widget = self.query_one("#user_input", Input)
        input_widget.disabled = False
        input_widget.focus()

        self.update_status("✅ Ready for your next message!")

    def action_clear_chat(self) -> None:
        """Clear the chat history."""
        self.chat_history.clear()
        chat_container = self.query_one("#chat_container", ChatContainer)

        # Remove all existing ChatMessage widgets
        for message in list(chat_container.query(ChatMessage)):
            message.remove()

        # Remove any existing welcome widget to avoid DuplicateIds
        try:
            existing_welcome = chat_container.query_one("#welcome")
            existing_welcome.remove()
        except Exception:
            pass

        # Mount a single welcome widget
        chat_container.mount(
            Static(
                Panel(
                    Align.center(
                        "[bold orange3]✨ Chat Cleared ✨[/bold orange3]\n"
                        "[dim]Start a new conversation[/dim]",
                        vertical="middle",
                    ),
                    border_style="orange3",
                    padding=1,
                ),
                id="welcome",
            )
        )
        self.update_status("Chat history cleared!")

    async def on_unmount(self) -> None:
        """Cleanup print capture when app is closing."""
        try:
            console = self.query_one("#console", PrintConsole)
            self.end_capture_print(target=console)
        except Exception:
            pass


def main() -> None:
    """Console script entry point to launch the Pomni TUI."""
    # On macOS and Linux, prefer 'fork' for Keras/JAX unless overridden
    try:
        mp.set_start_method("fork", force=True)
    except Exception:
        pass
    app = PomniChatTUI()
    app.run()


if __name__ == "__main__":
    main()
