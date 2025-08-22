import os
import asyncio
from typing import Optional

import keras
import keras_hub
import requests
from keras_hub.models import Gemma3CausalLM
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, Static, LoadingIndicator, RichLog
from textual.binding import Binding
from textual.reactive import reactive
from textual import work
from rich.text import Text
from rich.panel import Panel
from rich.align import Align


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
                title="[bold cyan]You[/bold cyan]",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
            )
        else:
            text = Text(self.content, style="green")
            return Panel(
                text,
                title="[bold green]Pomni[/bold green]",
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
                    "[bold magenta]✨ Welcome to Pomni Chat ✨[/bold magenta]\n"
                    "[dim]Chat with a fine-tuned Gemma model[/dim]",
                    vertical="middle"
                ),
                border_style="magenta",
                padding=1,
            ),
            id="welcome"
        )


class PomniChatTUI(App):
    """A TUI chatbot application using Gemma model."""
    
    CSS = """
    ChatContainer {
        height: 1fr;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    # Console area
    PrintConsole {
        height: 8;
        border: solid $surface; 
        color: $text-muted;
        overflow: auto;
        margin: 0 1 1 1;
    }
    
    Input {
        dock: bottom;
        margin: 1;
    }
    
    StatusBar {
        dock: bottom;
        height: 3;
    }
    
    ChatMessage {
        margin: 0 0 1 0;
    }
    
    LoadingIndicator {
        dock: bottom;
        height: 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
    ]
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.chat_history = []
        self.is_loading = True
        
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
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
        # Begin capturing stdout/stderr to the embedded console
        try:
            console = self.query_one("#console", PrintConsole)
            self.begin_capture_print(target=console, stdout=True, stderr=True)
        except Exception:
            # If console is not available for any reason, continue without capture
            pass
        self.load_model_async()
    
    @work(thread=True)
    def load_model_async(self) -> None:
        """Load the model in a background thread."""
        self.update_status("Loading Gemma model... This may take a while.")
        
        try:
            # Try loading from HuggingFace first
            self.update_status("Attempting to load model from HuggingFace...")
            try:
                model = keras.saving.load_model("hf://Neel-Gupta/pomni_4B")
                self.update_status("Successfully loaded model from HuggingFace!")
                
                # Compile the model
                sampler = keras_hub.samplers.TopKSampler(k=5, seed=42)
                model.compile(sampler=sampler)
                self.model = model
                
            except Exception as e:
                self.update_status("HuggingFace loading failed. Trying local weights...")
                
                # Fallback to local weights
                preset = "gemma3_instruct_1b"
                weights_path = "/Users/neel/Downloads/finetuned_gemma3_1b.weights.h5"
                weights_url = "https://filebin.net/gmmu2zultifcjlgi/finetuned_gemma3_1b.weights.h5"
                
                if not os.path.exists(weights_path):
                    self.update_status("Downloading model weights...")
                    if not self.download_weights(weights_url, weights_path):
                        self.update_status("Failed to download weights. Using base model.")
                
                model = Gemma3CausalLM.from_preset(preset, dtype="bfloat16")
                
                if os.path.exists(weights_path):
                    try:
                        model.load_weights(weights_path)
                        self.update_status("Successfully loaded fine-tuned weights!")
                    except Exception as e:
                        self.update_status(f"Error loading weights: {e}. Using base model.")
                else:
                    self.update_status("Using base model.")
                
                sampler = keras_hub.samplers.TopKSampler(k=5, seed=42)
                model.compile(sampler=sampler)
                self.model = model
            
            self.is_loading = False
            self.update_status("✅ Model loaded successfully! You can start chatting.")
            
            # Enable input
            input_widget = self.query_one("#user_input", Input)
            input_widget.disabled = False
            input_widget.focus()
            
        except Exception as e:
            self.update_status(f"❌ Error loading model: {str(e)}")
            self.is_loading = False
    
    def download_weights(self, url: str, dest: str) -> bool:
        """Download weights with progress indication."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            
            with open(dest, "wb") as f:
                bytes_downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size > 0:
                        progress = min(int((bytes_downloaded / total_size) * 100), 100)
                        self.update_status(
                            f"Downloading... {bytes_downloaded // 1024**2}MB / {total_size // 1024**2}MB ({progress}%)"
                        )
            
            return True
        except requests.exceptions.RequestException as _:
            return False
    
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
        except:
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
    
    @work(thread=True)
    def generate_response_async(self, prompt: str) -> None:
        """Generate model response in background thread."""
        try:
            # System prompt for nice behavior
            system_prompt = "You are a helpful, polite, and friendly AI assistant. Please provide clear, concise, and accurate responses while maintaining a warm and respectful tone."
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            # Generate response
            response = self.model.generate(full_prompt, max_length=128)
            
            # Clean response
            if full_prompt in response:
                clean_response = response[len(full_prompt):].strip()
            else:
                clean_response = response.strip()
            
            # Add to UI
            self.call_from_thread(self.add_assistant_message, clean_response)
            
        except Exception as e:
            self.call_from_thread(
                self.add_assistant_message,
                f"Sorry, I encountered an error: {str(e)}"
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
        
        # Remove all messages
        for message in chat_container.query(ChatMessage):
            message.remove()
        
        # Add welcome back
        chat_container.mount(
            Static(
                Panel(
                    Align.center(
                        "[bold magenta]✨ Chat Cleared ✨[/bold magenta]\n"
                        "[dim]Start a new conversation[/dim]",
                        vertical="middle"
                    ),
                    border_style="magenta",
                    padding=1,
                ),
                id="welcome"
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
 
 
if __name__ == "__main__":
     app = PomniChatTUI()
     app.run()
