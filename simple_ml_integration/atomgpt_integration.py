"""
atomgpt_integration.py - AtomGPT integration with custom ML tools
Using native Agapi client with simple keyword-based routing
"""

import os
from dotenv import load_dotenv
import re
from agapi.client import Agapi
from ml_backend import train_random_forest, evaluate_model, create_visualizations
import json

load_dotenv()

# Initialize AtomGPT client
api_key = os.getenv("ATOMGPT_API_KEY")
agapi_client = Agapi(api_key=api_key)

# No actual tooling for function calling os

# Tool definitions (for the LLM to understand)
AVAILABLE_TOOLS = """
You have access to these ML tools:

1. train_random_forest
   - Purpose: Train a Random Forest model
   - Required: data_path (CSV file), target_column (column name), task_type ('classification' or 'regression')
   - Optional: n_estimators (default: 100)
   - Example: {"tool": "train_random_forest", "args": {"data_path": "sales.csv", "target_column": "sales", "task_type": "regression"}}

2. evaluate_model
   - Purpose: Evaluate the trained model's performance
   - Required: None (uses the currently trained model)
   - Returns metrics
   - Example: {"tool": "evaluate_model", "args": {}}

3. create_visualizations
   - Purpose: Generate plots and charts
   - Required: None
   - Optional: output_dir (default: "./outputs")
   - Example: {"tool": "create_visualizations", "args": {}}
"""


def execute_tool(tool_name: str, tool_args: dict):
    """Execute a tool and return results"""
    try:
        if tool_name == "train_random_forest":
            return train_random_forest(**tool_args)
        elif tool_name == "evaluate_model":
            return evaluate_model()
        elif tool_name == "create_visualizations":
            return create_visualizations(**tool_args)
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def atomgpt_tooling(user_message: str, verbose: bool = True):
    """
    AtomGPT-Driven Tool Calling: The LLM decides which tools to call and when
    
    This is closer to how MCP servers and function calling work:
    - AtomGPT sees available tools
    - AtomGPT decides which tool(s) to call
    - We execute tools as requested
    - AtomGPT decides if more tools are needed or if it's done
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"User Request: {user_message}")
        print(f"{'='*70}\n")
    
    conversation_history = []
    max_iterations = 10  # Prevent infinite loops
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"{'─'*70}")
            print(f"Iteration {iteration + 1}")
            print(f"{'─'*70}\n")
        
        # Build the initial prompt for AtomGPT
        if iteration == 0:
            # First iteration - present the task & tools
            prompt = f"""{AVAILABLE_TOOLS}
            User request: "{user_message}"

            You are an ML assistant. Based on the user's request, decide which tool(s) to call.

            Important rules:
            1. For training requests, you typically want to call tools in this order: train_random_forest → evaluate_model → create_visualizations
            2. Respond with ONLY a JSON object in this format:
            {{"tool": "tool_name", "args": {{"param": "value"}}, "reasoning": "why you're calling this tool"}}
            3. If no more tools are needed, respond with: {{"done": true, "summary": "what was accomplished"}}
            4. One tool at a time - don't try to call multiple tools in one response

            What tool should be called first?"""
        else:
            # Subsequent iterations - show what's been done
            prompt = f"""{AVAILABLE_TOOLS}
            Original user request: "{user_message}"

            Actions completed so far:
            {chr(10).join(conversation_history)}

            Based on what's been done, what should happen next?

            Rules:
            1. Respond with: {{"tool": "tool_name", "args": {{"param": "value"}}, "reasoning": "why"}}
            2. OR if done: {{"done": true, "summary": "what was accomplished"}}
            3. One tool at a time

            What's next?"""
        
        # Ask the AtomGPT what to do
        if verbose:
            print("Asking AtomGPT what to do...")
        
        atomgpt_response = agapi_client.ask(prompt)
        
        if verbose:
            print(f"AtomGPT response:\n{atomgpt_response}\n")
        
        # Parse the AtomGPTs decision
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', atomgpt_response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                decision = json.loads(atomgpt_response)
            
            # Check if AtomGPT says we're done
            if decision.get("done"):
                if verbose:
                    print(f"✅ AtomGPT says we're done!\n")
                    print(f"{'='*70}\n")
                
                # Get final summary from AtomGPT
                summary_prompt = f"""The user asked: "{user_message}"
                    Here's what was accomplished:
                    {chr(10).join(conversation_history)}

                    Provide a natural, conversational summary highlighting:
                    - What model was trained (on which data, predicting what)
                    - Key performance metrics
                    - Where visualizations were saved
                    - Overall assessment of how well the model performed

                    Be concise and helpful."""
                
                summary = agapi_client.ask(summary_prompt)
                return summary
            
            # AtomGPT wants to call a tool
            tool_name = decision.get("tool")
            tool_args = decision.get("args", {})
            reasoning = decision.get("reasoning", "No reasoning provided")
            
            if not tool_name:
                return "I couldn't determine what action to take. Please try rephrasing your request."
            
            if verbose:
                print(f"AtomGPT decided to call: {tool_name}")
                print(f"Reasoning: {reasoning}")
                print(f"Arguments: {json.dumps(tool_args, indent=2)}")
            
            # Execute the tool
            result = execute_tool(tool_name, tool_args)
            
            if verbose:
                print(f"Result: {result.get('status', 'unknown')}")
                if result.get('status') == 'success':
                    if 'metrics' in result:
                        print(f"   Metrics: {result['metrics']}")
                    if 'message' in result:
                        print(f"   {result['message']}")
                print()
            
            # Add to conversation history
            conversation_history.append(
                f"- Called {tool_name} with {json.dumps(tool_args)}\n"
                f"  Result: {json.dumps(result, default=str)}"
            )
            
            # If tool failed, let AtomGPT know and ask what to do
            if result.get('status') == 'error':
                error_msg = result.get('message', 'Unknown error')
                if verbose:
                    print(f"❌ Tool failed: {error_msg}\n")
                
                # Ask AtomGPT how to handle the error
                error_prompt = f"""The tool {tool_name} failed with error: {error_msg}
                Should we:
                1. Try a different approach?
                2. Stop and report the error to the user?

                Respond with: {{"done": true, "summary": "error explanation"}} if we should stop."""
                                
                error_response = agapi_client.ask(error_prompt)
                
                try:
                    error_decision = json.loads(re.search(r'\{.*\}', error_response, re.DOTALL).group())
                    if error_decision.get("done"):
                        return error_decision.get("summary", f"Error: {error_msg}")
                except:
                    return f"I encountered an error: {error_msg}"
        
        except (json.JSONDecodeError, AttributeError) as e:
            if verbose:
                print(f"⚠️  Could not parse AtomGPT response as JSON: {e}\n")
            
            # AtomGPT didn't return valid JSON - ask it to try again
            if iteration < max_iterations - 1:
                continue
            else:
                return "I had trouble understanding what actions to take. Please try rephrasing your request."
    
    # Max iterations reached
    return "I completed some actions but may not have finished everything. Please check the results."

# Convenience wrapper
def simple_chat(user_message: str, verbose: bool = True):
    """Wrapper that uses atomgpt's tooling approach"""
    return atomgpt_tooling(user_message, verbose)


# Main execution for testing
if __name__ == "__main__":
    print("\n" + "🟦"*35)
    print("ATOMGPT ML ASSISTANT")
    print("🟦"*35 + "\n")
    
    # Test with simple chat
    response = simple_chat(
        "Train a random forest on sales_data.csv predicting sales, then evaluate it and create visualizations"
    )
    
    print(f"\n{'='*70}")
    print("🤖 ATOMGPT FINAL SUMMARY:")
    print(f"{'='*70}\n")
    print(response)
    print(f"\n{'='*70}\n")