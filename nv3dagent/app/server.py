#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval agent.

Relevant LangChain documentation:

* Creating a custom agent: https://python.langchain.com/docs/modules/agents/how_to/custom_agent
* Streaming with agents: https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events
* General streaming documentation: https://python.langchain.com/docs/expression_language/streaming

**ATTENTION**
1. To support streaming individual tokens you will need to use the astream events
   endpoint rather than the streaming endpoint.
2. This example does not truncate message history, so it will crash if you
   send too many messages (exceed token length).
3. The playground at the moment does not render agent output well! If you want to
   use the playground you need to customize it's output server side using astream
   events by wrapping it within another runnable.
4. See the client notebook it has an example of how to use stream_events client side!
"""
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from fastapi import FastAPI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.pydantic_v1 import BaseModel

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langserve import add_routes
from langchain.tools import Tool
import os
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, Agent, ConversationalAgent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.runnables import (
    ConfigurableField,
    ConfigurableFieldSpec,
    Runnable,
    RunnableConfig,
)
# add main "src" folder to the sys path
from pathlib import Path
import sys
from packages.xplane_agent import XPlaneAgent
from packages.xplane_config import get_model_name
from packages.utils.llm import get_llm
from langgraph.graph import END, StateGraph, MessageGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.utils import Input, Output


nvapi_key="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"
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", nvidia_api_key=nvapi_key, max_tokens=1024)

class CustomAgentExecutor(Runnable):
    """A custom runnable that will be used by the agent executor."""

    def __init__(self, **kwargs):
        """Initialize the runnable."""
        super().__init__(**kwargs)
        xplane_agent=XPlaneAgent(llm)

        workflow = MessageGraph()

        workflow.add_node("process_query", xplane_agent._process_query)
        workflow.set_entry_point("process_query")
        workflow.add_node("execute_command", xplane_agent._execute_command)
        workflow.add_node("call_tts", xplane_agent._call_tts)

        workflow.add_conditional_edges(
            "process_query",
            xplane_agent._route_response,
            {"command": "execute_command", "tts" : "call_tts", "end": END},
        )

        workflow.add_edge("execute_command", END)
        workflow.add_edge("execute_command", "call_tts")
        workflow.add_edge("call_tts", END)

        agent = workflow.compile()

        self.agent = agent

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Used and wrap the str user query into HumanMessage """
        print(" ------------------- input = \n ", type(input), input)
        query_str = HumanMessage(input['input'])
        res=self.agent.invoke(query_str)
        output=res[-1].content
        print(">>>>>>>>>>>>>>>>>>>>>>>>>\n", output)
        d={'output':output}
        return d


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)

# is lacking in schemas.
class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any

runnable = CustomAgentExecutor()
add_routes(
    app,
    runnable.with_types(input_type=Input, output_type=Output),
    disabled_endpoints=["batch"],  # not implemented
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)