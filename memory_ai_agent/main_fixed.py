import sys
import time
import re
from rich.panel import Panel
from rich.markdown import Markdown
from utils.memory_vectordb import Memory  # Use the VectorDB-based Memory class
from utils.search import SearchTool
from utils.terminal import TerminalUI
from models.llm import LLMInterface
from prompts.system_prompt import get_system_prompt, get_knowledge_check_prompt
from config import (
    OPENROUTER_API_KEY, 
    SITE_URL, 
    SITE_NAME, 
    MODELS, 
    DEFAULT_MODEL,
    MEMORY_LIMIT
)

def main():
    # Initialize components - without requiring API key
    memory = Memory(
        memory_file="memory.json", 
        vector_db_path="vector_db.pkl",
        memory_limit=MEMORY_LIMIT
    )
    search_tool = SearchTool()
    terminal = TerminalUI()
    
    try:
        # Try to use the API key if available, but don't require it
        api_key = OPENROUTER_API_KEY or "demo"  # Use "demo" as fallback
        llm = LLMInterface(
            api_key=api_key,
            site_url=SITE_URL or "http://localhost",
            site_name=SITE_NAME or "Memory AI Agent",
            default_model=DEFAULT_MODEL,
            available_models=MODELS
        )
    except Exception as e:
        # Print error but continue with minimal functionality
        print(f"Warning: Could not initialize LLM interface: {str(e)}")
        print("Running in local-only mode with limited functionality.")
        llm = None
    
    # Display welcome message
    terminal.display_welcome()
    
    # Main loop
    current_model = DEFAULT_MODEL
    
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Handle special commands
        if user_input.lower() == "exit":
            terminal.console.print("[bold yellow]Goodbye![/bold yellow]")
            break
            
        if user_input.lower() == "clear":
            terminal.clear_screen()
            continue
            
        # Check for model change command
        model_change_match = re.match(r"model:\s*(\w+)", user_input.lower())
        if model_change_match:
            requested_model = model_change_match.group(1)
            if requested_model in MODELS:
                current_model = requested_model
                terminal.console.print(f"[bold green]Switched to {current_model} model[/bold green]")
                continue
            else:
                terminal.console.print(f"[bold red]Model {requested_model} not available. Available models: {', '.join(MODELS.keys())}[/bold red]")
                continue
        
        # Display user input
        terminal.display_user_input(user_input)
        
        # Get conversation history
        conversation_history = memory.get_formatted_history()
        
        # Check for relevant memories for context - with semantic search
        try:
            relevant_context = memory.get_relevant_context(user_input)
            if relevant_context:
                terminal.console.print(Panel(
                    Markdown(relevant_context),
                    title="Relevant Memories Found",
                    border_style="yellow"
                ))
        except Exception as e:
            terminal.console.print(f"[bold red]Error retrieving relevant context: {str(e)}[/bold red]")
            relevant_context = ""
        
        # Check if it's an explicit search query
        search_match = re.match(r"search:\s*(.*)", user_input)
        search_results = None
        
        if search_match:
            query = search_match.group(1)
            terminal.display_thinking(f"Searching for '{query}'")
            
            # Perform search
            results, search_time = search_tool.search(query)
            search_results = search_tool.format_results(results)
            
            # Display tool usage
            terminal.display_tool_usage("DuckDuckGo Search", search_time)
        elif llm is not None:
            # Check if we need to search for information
            terminal.display_thinking("Checking if I need to search for information")
            try:
                needs_search, search_query, check_time = llm.check_knowledge(
                    user_input=user_input,
                    system_prompt=get_knowledge_check_prompt(),
                    conversation_history=conversation_history,
                    model=current_model
                )
                
                terminal.display_tool_usage(f"Knowledge Check ({current_model})", check_time)
                
                # If we need to search, do it
                if needs_search:
                    terminal.display_thinking(f"Searching for '{search_query}'")
                    
                    # Perform search
                    results, search_time = search_tool.search(search_query)
                    search_results = search_tool.format_results(results)
                    
                    # Display tool usage
                    terminal.display_tool_usage("DuckDuckGo Search (Auto)", search_time)
            except Exception as e:
                terminal.console.print(f"[bold red]Error during knowledge check: {str(e)}[/bold red]")
                search_results = None
        
        # Generate response
        terminal.display_thinking("Thinking")
        
        # Include relevant memory context in the prompt
        system_prompt = get_system_prompt()
        if relevant_context:
            system_prompt += f"\n\n{relevant_context}"
        
        if llm is not None:
            try:
                response, model_used, llm_time = llm.generate_response(
                    user_input=user_input,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    search_results=search_results,
                    model=current_model
                )
                
                # Display tool usage
                terminal.display_tool_usage(f"LLM ({model_used})", llm_time)
                
                # Display response gradually
                terminal.display_ai_response(response, model_used, gradual=True)
                
                # Automatically save to memory without asking
                try:
                    memory.add_conversation(user_input, response)
                    # Don't show save confirmation to avoid cluttering the UI
                except Exception as e:
                    terminal.console.print(f"[red]Error saving to memory: {str(e)}[/red]")
            except Exception as e:
                terminal.console.print(f"[bold red]Error generating response: {str(e)}[/bold red]")
                response = "I'm sorry, I couldn't generate a response at this time. (Running in local-only mode)"
                terminal.display_ai_response(response, "local", gradual=False)
        else:
            # No LLM available, provide a generic response
            response = "I'm running in local-only mode with limited functionality. To enable full functionality, please set the OPENROUTER_API_KEY in your config."
            terminal.display_ai_response(response, "local", gradual=False)
            
            # Still save to memory even in local mode
            try:
                memory.add_conversation(user_input, response)
            except Exception:
                pass
        
        # Extract user information (simple approach - could be enhanced)
        if "my name is" in user_input.lower():
            match = re.search(r"my name is\s+([A-Za-z]+)", user_input, re.IGNORECASE)
            if match:
                name = match.group(1)
                memory.add_user_info("name", name)
                
if __name__ == "__main__":
    main()
