"""
Basic usage example for Dinnovos Agent
"""

from dinnovos import Agent, OpenAILLM

# Create an LLM interface
llm = OpenAILLM(
    api_key="your-api-key-here",  # Replace with your API key
    model="gpt-4"
)

# Create a Dinnovos agent
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful Python programming assistant.",
    max_history=10
)

# Have a conversation
print("Dinnovos Agent - Basic Usage Example")
print("=" * 50)

# First message
response1 = agent.chat("What is a list comprehension in Python?")
print(f"\nUser: What is a list comprehension in Python?")
print(f"Dinnovos: {response1}")

# Follow-up message (with context)
response2 = agent.chat("Can you give me a practical example?")
print(f"\nUser: Can you give me a practical example?")
print(f"Dinnovos: {response2}")

# Get conversation history
print("\n" + "=" * 50)
print("Conversation History:")
history = agent.get_history()
for i, msg in enumerate(history):
    print(f"{i+1}. {msg['role']}: {msg['content'][:50]}...")

# Reset conversation
agent.reset()
print("\n" + "=" * 50)
print("Conversation reset!")

# New conversation with different system prompt
agent.set_system_prompt("You are a creative writing assistant.")
response3 = agent.chat("Write a short poem about coding")
print(f"\nUser: Write a short poem about coding")
print(f"Dinnovos: {response3}")