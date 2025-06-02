from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress
from rich.live import Live
import time
from typing import Dict, Any, Optional

class TerminalUI:
    def __init__(self):
        """Initialize terminal UI."""
        self.console = Console()
        
    def display_welcome(self):
        """Display welcome message."""
        welcome_text = """
# Memory AI Agent

An AI assistant that remembers your conversations and can search the web.

- Type your message to start chatting
- Type 'exit' to quit
- Type 'clear' to clear the screen
- Type 'search: your query' to search the web
- After each response, you'll be asked to save it to memory (Y/N)
        """
        self.console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))
        
    def display_user_input(self, text: str):
        """Display user input."""
        self.console.print(Panel(text, title="You", border_style="green"))
        
    def display_ai_response(self, text: str, model_name: str, gradual=True):
        """Display AI response with markdown formatting, gradually if enabled.
        
        Args:
            text: The response text
            model_name: Name of the model used
            gradual: Whether to display text gradually
        """
        if not gradual:
            # Display all at once if gradual is disabled
            self.console.print(Panel(
                Markdown(text), 
                title=f"AI ({model_name})", 
                border_style="purple"
            ))
            return
            
        # For gradual display, use Rich's Live display
        words = text.split()
        
        # Create the initial panel with empty content
        panel = Panel("", title=f"AI ({model_name})", border_style="purple")
        
        with Live(panel, console=self.console, refresh_per_second=20) as live:
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                
                # Update panel content
                panel.renderable = Markdown(current_text)
                
                # Control the typing speed
                time.sleep(0.05)
                
                # Refresh less frequently for better performance
                if i % 3 == 0:
                    live.refresh()
                    
            # Ensure final text is shown completely
            panel.renderable = Markdown(text)
            live.refresh()
        
    def display_thinking(self, action: str):
        """Display thinking animation."""
        with Progress() as progress:
            task = progress.add_task(f"[cyan]{action}...", total=100)
            
            while not progress.finished:
                progress.update(task, advance=0.9)
                time.sleep(0.01)
                
    def display_tool_usage(self, tool_name: str, time_taken: float):
        """Display tool usage information."""
        table = Table(title=f"Tool Usage: {tool_name}")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Time taken", f"{time_taken:.4f} seconds")
        
        self.console.print(table)
        
    def clear_screen(self):
        """Clear the terminal screen."""
        self.console.clear()
        
    def ask_memory_confirmation(self):
        """Ask user for confirmation to add the response to memory.
        
        Returns:
            bool: Whether to add to memory
        """
        while True:
            response = input("\nSave this exchange to memory? (Y/N): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                self.console.print("[yellow]Please enter Y or N.[/yellow]")