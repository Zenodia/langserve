import importlib.machinery
import os
from packages.utils.kbevents_client import send_kb_event  



# define useful config data
config_data = {
    "remote_ip": "10.63.181.233",
    # "remote_ip": "localhost",
    # "remote_ip": "10.110.94.104",
    "port": 5000,
    "llm_model": "gpt-4-turbo", 
}

# manually define list of commands in the application + their keyboard bindings
commands_bindings = {"show_top_back_of_the_plane()" : "shift+1",
                    "show_top_front_of_the_plane()" : "shift+2",
                    "show_runway()" : "shift+3",
                    "show_side_of_the_plane()" : "shift+4",
                    "show_tower()" : "shift+5",
                    "show_back_of_the_plane()" : "shift+6",
                    "reset_panel_view()" : "w",
                    "move_view_left(angle)": "q",
                    "move_view_right(angle)": "e",
                    "move_view_up(angle)": "r",
                    "move_view_down(angle)": "f",
                    "show_map_panel()": "m",
                    }

command_file = "xplane_commands"
file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + command_file + ".py"

def get_command_list():
    """
        Getting supported command list.
    """
    return list(commands_bindings.keys())


def get_model_name():
    """
        Getting LLM model name from config
    """
    return config_data["llm_model"]


def get_command_kbevent(command: str):
    """
        Return command keyboard event
    """
    if command in commands_bindings:
        return commands_bindings[command]
    else:
        return None


def get_readable_command(command: str): 
    """"
        Coinvert command in a form of function to a readable by the agent response
    """
    command = command.split("(")[0]
    command = command.replace("_", " ")
    print("Readable command: " + command)
    return command


def call_function_from_file(module_name, path, function_signature):
    """
        Call function from a file in a dynamic way
    """
    try:
        print(f"call_function_from_file() module_name={module_name}, path={path}, function_signature={function_signature}")
        # module = importlib.import_module(module_name, path)
        loader = importlib.machinery.SourceFileLoader(module_name, path)
        module = loader.load_module()

        full_function_signature = f"module.{function_signature}"

        print("Calling function: " + full_function_signature)
        result = eval(full_function_signature)
        print(result)
        return result
    except Exception as e:
        print(f"Error execution the function call {e}")
        return None


def trigger_command(command: str):
    """
        Trigger keyword event for the select command in the remote machine where flight simulator is running.
    """
    # try to run the custom defined function
    res = call_function_from_file(command_file, file_path, command)

    if res:
        # if succeded
        return res
    else:
        # otherwise for any other generic function just trigger relevant kb_event
        kb_event = get_command_kbevent(command)
        if kb_event:
            print(f"Executing command: {command}. Triggering keyboard event: {kb_event}")
            send_kb_event(kb_event, config_data["remote_ip"], config_data["port"])
            return "Triggering " + get_readable_command(command)
        else:
            return "Can't execute " + get_readable_command(command)
        

if __name__ == "__main__":
    command_file = "xplane_commands"
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + command_file + ".py"
    print(file_path)
    call_function_from_file(command_file, file_path, "move_view_left(15)")