from flask import Flask, request, jsonify
from langchain_openai import OpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
import chainlit as cl  # Optional, if you want to include Chainlit

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize the LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# Tools initialization
# Wikipedia Tool
wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia.run,
    description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.",
)

# Math Tool
problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(
    name="Calculator",
    func=problem_chain.run,
    description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.",
)

# Reasoning Tool
word_problem_template = """You are a reasoning agent tasked with solving 
the user's logic-based questions. Logically arrive at the solution, and be 
factual. In your answers, clearly detail the steps involved and give the 
final answer. Provide the response in bullet points. 
Question: {question} 
Answer:"""

math_assistant_prompt = PromptTemplate(
    input_variables=["question"], template=word_problem_template
)
word_problem_chain = LLMChain(llm=llm, prompt=math_assistant_prompt)
word_problem_tool = Tool.from_function(
    name="Reasoning Tool",
    func=word_problem_chain.run,
    description="Useful to solve and answer reasoning-based or logic-based questions.",
)

# Initialize the agent
agent = initialize_agent(
    tools=[wikipedia_tool, math_tool, word_problem_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)

# Flask endpoint for user queries
@app.route("/query", methods=["POST"])
def process_query():
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Use the agent to process the query
        response = agent.run(user_input)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: Chainlit integration
@cl.on_chat_start
def chainlit_chatbot():
    cl.user_session.set("agent", agent)

@cl.on_message
async def chainlit_process_query(message: cl.Message):
    agent = cl.user_session.get("agent")
    response = await agent.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(response["output"]).send()

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
