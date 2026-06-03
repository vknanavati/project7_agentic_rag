# agent.py
# The core reasoning loop of the Agentic RAG system
# Implements the ReAct pattern: Thought → Action → Observation → repeat
# until the LLM writes a Final Answer or we hit the maximum iterations
# Analogy: a detective who keeps gathering clues, re-evaluating what they know,
#          and only closes the case when they are confident they have the full picture

import os                                       # for reading environment variables
import time                                     # for sleeping between API calls if needed
from groq import Groq                           # the Groq API client

import config                                   # our central settings file
from tools import (                             # import our four retrieval tools
    call_tool,                                  # the dispatcher that routes tool calls
    format_tool_schemas_for_prompt,             # formats tool descriptions for the system prompt
)


# --- Groq Client ---
# Instantiate the Groq client once at module load time
# It reads the GROQ_API_KEY environment variable automatically
# Analogy: picking up the phone once and keeping the line open,
#          rather than dialing a new number on every question
client = Groq()


# --- System Prompt ---

def build_system_prompt():
    # Plain explanation: constructs the instruction text that tells the Groq model
    #                    exactly how to behave, what tools it has, and what format to follow
    # Analogy: the briefing a detective gets before starting a case —
    #          here are your resources, here are the rules, here is how to report back

    # Get the formatted tool descriptions from tools.py
    tool_descriptions = format_tool_schemas_for_prompt()

    # Build the full system prompt as a multi-line string
    # Every detail here directly shapes how the LLM behaves in the reasoning loop
    prompt = f"""You are Birdie, an expert birding assistant with access to a knowledge base \
about North American backyard birds found in Connecticut and Pennsylvania.

You answer questions by searching the knowledge base using the available tools \
and synthesizing what you find into a complete answer.

You have access to the following tools:

{tool_descriptions}

CRITICAL INSTRUCTION — ONE ACTION PER RESPONSE:
You must write EXACTLY ONE Thought, ONE Action, and ONE Action Input per response.
Then STOP. Do not write another Thought or Action. Do not write Final Answer yet.
Wait for the Observation (tool result) before continuing.

You must follow this EXACT format on every single response:

Thought: [Reason about what you know so far and what you still need.]
Action: [ONE tool name: search_knowledge_base, keyword_search, filter_by_species, or compare_species]
Action Input: [The input to pass to the tool. No quotes around it.]

WHEN TO WRITE YOUR FINAL ANSWER:
After 3-4 searches, stop and write your Final Answer using whatever you have found.
Do not keep searching for perfect information — good retrieved evidence is enough.
If the knowledge base has partial information, share what you found and note any gaps.

To write your final answer use this exact format:

Final Answer: [Your complete answer based on the retrieved chunks.]

CRITICAL: "Final Answer:" is NOT a tool. Never write "Action: Final Answer" or "Action: None".
Write "Final Answer:" as a standalone line followed immediately by your answer.
If you have enough information after 3-4 searches, stop searching and write your Final Answer.

RULES:
- ONE Thought, ONE Action, ONE Action Input per response — then stop
- No quotes around Action Input values
- Species names use lowercase with underscores (e.g. dark_eyed_junco, american_goldfinch)
- After 3-4 searches write your Final Answer with whatever evidence you have collected
- Never write "Action: None" — always either call a tool or write Final Answer
"""

    return prompt


# --- Output Parser ---

def parse_llm_output(text):
    # Plain explanation: reads the raw text the LLM wrote and extracts
    #                    the structured fields — Thought, Action, Action Input,
    #                    or Final Answer — so our loop knows what to do next
    # Analogy: reading a detective's case notes and pulling out the key decisions —
    #          what they concluded, what they did next, and what they found

    # Check if the LLM has decided it is done
    # If "Final Answer:" appears anywhere in the text, the loop should stop
    if config.FINAL_ANSWER_TOKEN in text:
        # Split on the Final Answer token and take everything after it
        answer = text.split(config.FINAL_ANSWER_TOKEN)[-1].strip()
        return {"type": "final_answer", "content": answer}

    # Otherwise the LLM should have written a Thought + Action + Action Input block
    # We extract each field by finding the token and taking the text after it

    # Extract the Thought
    thought = ""
    if config.THOUGHT_TOKEN in text:
        # Take everything after "Thought:" up to the next token or end of string
        thought_part = text.split(config.THOUGHT_TOKEN)[-1]
        # Stop at "Action:" if it appears — everything before that is the Thought
        if config.ACTION_TOKEN in thought_part:
            thought = thought_part.split(config.ACTION_TOKEN)[0].strip()
        else:
            thought = thought_part.strip()

    # Extract the Action (tool name)
    action = ""
    if config.ACTION_TOKEN in text:
        action_part = text.split(config.ACTION_TOKEN)[-1]
        # Stop at "Action Input:" — everything before that is the tool name
        if config.ACTION_INPUT_TOKEN in action_part:
            action = action_part.split(config.ACTION_INPUT_TOKEN)[0].strip()
        else:
            action = action_part.strip()

    # Extract the Action Input (what to pass to the tool)
    action_input = ""
    if config.ACTION_INPUT_TOKEN in text:
        # Everything after "Action Input:" is the tool input
        action_input = text.split(config.ACTION_INPUT_TOKEN)[-1].strip()

    # Return a dictionary describing what the LLM decided to do
    return {
        "type": "action",                       # signals the loop should call a tool
        "thought": thought,                     # the LLM's reasoning (for display)
        "action": action,                       # the tool name to call
        "action_input": action_input,           # the input to pass to the tool
    }


# --- Tool Result Formatter ---

def format_tool_result(tool_name, tool_input, results):
    # Plain explanation: converts the list of retrieved chunks into a readable
    #                    string that gets appended to the conversation history
    #                    so the LLM can read what the tool returned
    # Analogy: the detective's assistant summarising the evidence they collected
    #          and writing it up in the case file for the detective to read

    # Start with a header showing which tool ran and what input it received
    lines = [
        f"Tool: {tool_name}",
        f"Input: {tool_input}",
        f"Results ({len(results)} chunks retrieved):",
        "",
    ]

    # If no chunks were returned, say so explicitly
    if not results:
        lines.append("No relevant chunks found.")
        return "\n".join(lines)

    # Format each retrieved chunk with its species tag and relevance score
    for i, result in enumerate(results, 1):             # enumerate starting at 1
        species = result["metadata"].get("species", "Unknown")  # get species name from metadata
        score = result["score"]                                  # relevance score
        text = result["text"]                                    # the chunk text itself

        lines.append(f"[Chunk {i}] Species: {species} | Score: {score}")
        lines.append(text[:200])                              # the actual retrieved text
        lines.append("")                                # blank line between chunks

    return "\n".join(lines)                             # join everything into one string


# --- Main Agent Loop ---

def run_agent(question, collection, embedding_model, verbose=True):
    # Plain explanation: runs the full ReAct reasoning loop for one question —
    #                    sends the question to Groq, parses the response,
    #                    calls tools, appends results, and repeats until
    #                    the LLM writes a Final Answer or we hit MAX_ITERATIONS
    # Analogy: the detective opening a new case file, working the clues one by one,
    #          and closing the file only when the case is solved

    # Build the conversation history that we will keep extending each iteration
    # The system prompt goes first — it sets the rules for the entire session
    messages = [
        {"role": "system", "content": build_system_prompt()},   # rules and tool descriptions
        {"role": "user", "content": question},                  # the user's question
    ]

    time.sleep(3)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

    # Track how many iterations have run
    iterations = 0

    # Keep looping until the LLM produces a Final Answer or we hit the limit
    while iterations < config.MAX_ITERATIONS:

        iterations += 1                         # increment the iteration counter

        if verbose:
            print(f"\n--- Iteration {iterations} ---")

        # --- Step 1: Call the Groq LLM ---
        # Send the full conversation history to Groq and get a response
        # The LLM reads everything — system prompt, user question, and all
        # prior tool results — before writing its next Thought + Action
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,                    # which LLM to use
            messages=messages,                          # the full conversation so far
            max_tokens=config.GROQ_MAX_TOKENS,          # cap the response length
            temperature=config.GROQ_TEMPERATURE,        # 0.0 = deterministic reasoning
        )

        # Extract the raw text from the Groq response object
        llm_output = response.choices[0].message.content.strip()

        if verbose:
            print(f"\nLLM Output:\n{llm_output}")

        # --- Step 2: Parse the LLM Output ---
        # Figure out whether the LLM wrote a Final Answer or a tool call
        parsed = parse_llm_output(llm_output)

        # --- Step 3: Check for Final Answer ---
        # If the LLM is done, return the answer and stop the loop
        if parsed["type"] == "final_answer":
            if verbose:
                print(f"\nFinal Answer reached after {iterations} iteration(s).")
            return {
                "answer": parsed["content"],            # the LLM's final answer text
                "iterations": iterations,               # how many loops it took
                "question": question,                   # echo the original question
            }

        # --- Step 4: Call the Tool ---
        # The LLM wrote a tool name and input — run the actual Python function
        tool_name = parsed["action"]                    # which tool to call
        tool_input = parsed["action_input"].strip("'\"")  # strip quotes the LLM added

        # If the agent wrote "None" as the action it means it is done but
        # forgot to write Final Answer — prompt it explicitly to write one
        if tool_name.lower() in ["none", "", "n/a"]:
            if verbose:
                print("\nAgent signaled done without Final Answer — prompting for it.")
            messages.append({
                "role": "assistant",
                "content": llm_output,
            })
            messages.append({
                "role": "user",
                "content": "You have enough information. Now write your Final Answer:",
            })
            continue                                     # go back to top of loop to get the answer

        if verbose:
            print(f"\nThought: {parsed['thought']}")
            print(f"Calling tool: {tool_name}")
            print(f"Input: {tool_input}")

        # Call the tool via our dispatcher in tools.py
        tool_results = call_tool(tool_name, tool_input, collection, embedding_model)

        # Format the results into a readable string
        formatted_result = format_tool_result(tool_name, tool_input, tool_results)

        if verbose:
            print(f"\nTool returned {len(tool_results)} chunk(s).")

        # --- Step 5: Append to Conversation History ---
        # Add the LLM's output and the tool result to the message history
        # On the next iteration, the LLM will read all of this before responding
        messages.append({
            "role": "assistant",                        # the LLM's Thought + Action
            "content": llm_output,
        })
        messages.append({
            "role": "user",                             # tool results go in as a user message
            "content": f"Observation:\n{formatted_result}",  # labeled as an Observation
        })

        # Keep conversation history from growing too large
        # Always preserve: system prompt (index 0) and original question (index 1)
        # Keep only the last 6 messages after those two
        if len(messages) > 8:                           # system + question + 6 recent messages
            messages = messages[:2] + messages[-6:]     # slice to keep first 2 and last 6

    # --- Max Iterations Reached ---
    # The loop ran out of attempts without a Final Answer
    # Return whatever the last LLM output was rather than crashing
    if verbose:
        print(f"\nMax iterations ({config.MAX_ITERATIONS}) reached without a Final Answer.")

    return {
        "answer": "I was unable to fully answer the question within the allowed number of searches. Please try rephrasing.",
        "iterations": iterations,
        "question": question,
    }