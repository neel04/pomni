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
from textual.containers import ScrollableContainer, Vertical, Grid
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static, Label


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




class DownloadConfirmScreen(ModalScreen[tuple[bool, str]]):
    """Modal dialog asking user to confirm model download on startup and select model.

    Returns (proceed, model_repo) where model_repo is the HF repo id.
    """

    CSS = """
    DownloadConfirmScreen { 
        align: center middle;
        layout: vertical;
    }
    #dialog { 
        grid-size: 2; 
        grid-gutter: 1 2; 
        grid-rows: 1fr auto auto auto; 
        padding: 2; 
        width: 76; 
        height: 20;
        border: thick $background 80%;
        background: $surface;
        align: center middle;
    }
    #question { 
        column-span: 2; 
        height: 1fr; 
        width: 1fr; 
        content-align: center middle;
    }
    #model_label { 
        column-span: 2; 
        padding: 0 0;
    }
    #model_buttons { 
        column-span: 2; 
        layout: horizontal; 
        content-align: center middle;
        height: 3;
        width: 100%;
    }
    #model_buttons Button {
        width: 1fr;
        margin: 0 1;
    }
    Button { 
        width: 100%; 
    }
    .selected { 
        border: solid $primary; 
    }
    """
    def __init__(self) -> None:
        super().__init__()
        # Default selection: 4B recommended
        self._selected: str = "Neel-Gupta/pomni_4B"

    def compose(self) -> ComposeResult:
        yield Grid(
            Label(
                "This app will download and load a chat model from HuggingFace.\n"
                "The download may be several GB and could take time depending on your connection.\n\n"
                "Select a model and proceed to start the chat:",
                id="question",
            ),
            Label("Choose a model (4B recommended):", id="model_label"),
            Grid(
                Button("4B — Recommended", id="opt_4b", variant="warning", classes="recommended selected"),
                Button("12B", id="opt_12b", variant="primary"),
                id="model_buttons",
            ),
            Button("Yes, proceed", id="yes", variant="success"),
            Button("No, exit", id="no", variant="error"),
            id="dialog",
        )

    BINDINGS = [
        ("left", "focus_prev", ""),
        ("right", "focus_next", ""),
        ("escape", "dismiss_no", ""),
        ("1", "select_4b", ""),
        ("2", "select_12b", ""),
    ]

    def action_focus_next(self) -> None:
        try:
            self.focus_next()
        except Exception:
            pass

    def action_focus_prev(self) -> None:
        try:
            self.focus_previous()
        except Exception:
            pass

    def action_dismiss_no(self) -> None:
        self.dismiss((False, self._selected))

    def action_select_4b(self) -> None:
        self._set_selection("Neel-Gupta/pomni_4B")

    def action_select_12b(self) -> None:
        self._set_selection("Neel-Gupta/pomni")

    def _set_selection(self, repo: str) -> None:
        self._selected = repo
        try:
            b4 = self.query_one("#opt_4b", Button)
            b12 = self.query_one("#opt_12b", Button)
            # Reset classes
            b4.set_class(False, "selected")
            b12.set_class(False, "selected")
            # Apply to selected
            if repo.endswith("_4B"):
                b4.set_class(True, "selected")
            else:
                b12.set_class(True, "selected")
        except Exception:
            pass

    def on_mount(self) -> None:
        # Set initial focus to the "Yes" button so arrows work immediately
        try:
            self.query_one("#yes", Button).focus()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "opt_4b":
            self._set_selection("Neel-Gupta/pomni_4B")
            return
        if event.button.id == "opt_12b":
            self._set_selection("Neel-Gupta/pomni")
            return
        if event.button.id == "yes":
            self.dismiss((True, self._selected))
        else:
            self.dismiss((False, self._selected))


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

    /* Modal centering & layout for confirm */
    DownloadConfirmScreen { align: center middle; }
    #download_dialog { width: 72; }
    #download_buttons { layout: horizontal; grid-gutter: 2; align-horizontal: center; }
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
        self._selected_repo: str = "Neel-Gupta/pomni_4B"

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

        # Show confirmation modal before any downloads / loading
        self.push_screen(DownloadConfirmScreen(), self._on_download_decision)

    def _on_download_decision(self, result) -> None:
        """Handle user's choice from the download confirmation dialog.

        Result is a tuple (proceed: bool, repo: str) from DownloadConfirmScreen.
        """
        try:
            proceed, repo = result if isinstance(result, tuple) else (bool(result), "Neel-Gupta/pomni_4B")
        except Exception:
            proceed, repo = False, "Neel-Gupta/pomni_4B"
        self._selected_repo = repo
        if proceed:
            # User accepted; start loading
            self.update_status(f"Preparing to download / load model: {repo} …")
            self.load_model_async()
        else:
            # User declined; quit the app immediately
            self.update_status("Exiting without downloading the model.")
            self.exit()

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
            repo = getattr(self, "_selected_repo", "Neel-Gupta/pomni_4B")
            model = keras.saving.load_model(f"hf://{repo}")
            self.update_status(f"Successfully loaded model from HuggingFace: {repo}!")

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
