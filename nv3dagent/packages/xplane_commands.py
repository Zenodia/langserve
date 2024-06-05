from packages.xplane_config import get_command_kbevent, send_kb_event, config_data

ANGLE_UNIT = 5 # degree


def move_view(command, angle):
    """
        Generic implementation for moving view by given angle in one of the directions
    """
    print(f"{command} with angle {angle}")
    
    kbevent = get_command_kbevent(f"{command}(angle)")

    if (angle < 5 or angle > 90):
        return "Angle range is limited between 5 to 90"
    
    kbevent_multiplier = int(angle/ANGLE_UNIT)
    combined_kbevent = (kbevent + "+") * kbevent_multiplier
    if combined_kbevent.endswith("+"):
        combined_kbevent = combined_kbevent[:-1]
    print("Combined kbevent: " + combined_kbevent)

    send_kb_event(combined_kbevent, config_data["remote_ip"], config_data["port"])

    return f"{command} {angle} degrees" 


def move_view_left(angle=ANGLE_UNIT):
    return move_view("move_view_left", angle)


def move_view_right(angle=ANGLE_UNIT):
    return move_view("move_view_right", angle)


def move_view_up(angle=ANGLE_UNIT):
    return move_view("move_view_up", angle)


def move_view_down(angle=ANGLE_UNIT):
    return move_view("move_view_down", angle)
