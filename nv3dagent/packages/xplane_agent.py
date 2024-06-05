from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import LLM
from langgraph.graph import END, StateGraph, MessageGraph

from typing import TypedDict

#from packages.xplane_config import get_command_list, trigger_command
from packages.utils.llm import get_llm
from packages.utils.kbevents_client import send_kb_event  

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
import sys
import os
import importlib.machinery
import os
from packages.xplane_config import get_command_list, trigger_command




SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class AgentState(TypedDict):
    """"
        Will hold the agent state in between messages
    """
    None


class XPlaneAgent:
    """
    Implementation of the XPlane agent
    """  
    ACTION_TAG = "@action"
    TTS_TAG = "@tts"

    system_prompt = """You are an action agent for the flight simulator environment. 
You goal is to select an action with parameters from a list of provided actions.
If user asks a question, just answer the question in a short precise way.
Otherwise write call to a function with parameters in Python syntax.
Start this sentence with "{action_tag}" token.

If some required parameters are missing and there are no default values in functions signature, ask user to provide missing parameters.
If docstring of the function specify which parameters are supported by this function try to match provided parameter value against these options.
If there is no match inform user about that and suggest supported options.
If there are multiple possible matches of command parameter against supported values, ask user to select between them.
If user request looks like a command but there is no appropriate action from the list of actions then respond with "I don't know how to do that".

Command list:
{command_list}
"""    

    def __init__(self, llm: LLM):
        """
            Initialize the XPlane agent and create appropriate LangGraph workflow
        """
        self._llm = llm

        # get list of commands of XPlane app (currently manually defined in the config)
        command_list = get_command_list()
        print(command_list)

        # update system prompt with relevant functions
        self._system_prompt = self.system_prompt.format(action_tag=self.ACTION_TAG, command_list=command_list,
                                            tts_tag=self.TTS_TAG)
        self.history={'_execute_command':[], '_route_response':[]}
        # workflow = StateGraph(AgentState)
        

    def _process_query(self, state: AgentState  ):
        """
        Main step of processing user query
        """
        # add system message to the state
        #input = HumanMessage(content=)
        
        print(" >>>>>>>>>>>>>> inside _process_query:\n", state)
        state.insert(0,SystemMessage(self._system_prompt))
        print(state)
        result = self._llm.invoke(state).content
        print(result)

        # add the result to the state
        state.append(AIMessage(result))

    def _route_response(self, state: AgentState):
        """
        Route the response to the appropriate handler
        """
        self.history['_route_response'].append(state)
        last_message = state[-1].content
        if self.ACTION_TAG in last_message:
            return "command"
        elif self.TTS_TAG in last_message:
            #return "tts"
            return "END"
        return "END"  

    def _execute_command(self, state: AgentState): 
        """
        Parse the response and execute appropriate VRED commands (variants).
        It supports execution of multiple atomic commands as well, each one in a new line
        """
        command_str = state[-1].content
        if self.ACTION_TAG not in command_str:
            print(f"No {self.ACTION_TAG} in the response")
            return

        print("Instructions: " + command_str)
        instructions = command_str.split("\n")
        print(f"instruction num: {len(instructions)}")

        response = ""
        for instruction in instructions:
            print(instruction)
            if self.ACTION_TAG not in instruction:
                continue

            command = instruction.split(self.ACTION_TAG)[1].strip().replace("\\","").replace(' ','')
            print("Command: " + command)

            cmd_response = trigger_command(command)

            if response:
                response += " and " + cmd_response
            else:
                response += cmd_response
        self.history['_execute_command'].append(self.TTS_TAG + " " + response)
        # update the state with the response
        if response:
            state.append(AIMessage(self.TTS_TAG + " " + response))

    def _call_tts(self, state: AgentState): 
        """
        Call text to speech function on the response
        """
        pass

    
        # print(res)

if __name__ == "__main__":
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # llm = get_llm("gpt-4-turbo")
    llm = get_llm("ai-llama3-70b")

    xplane_agent = XPlaneAgent(llm)
    xplane_agent.ask("Show me front of the plane")
