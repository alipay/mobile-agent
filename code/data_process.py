import jax
import json
import pickle
import argparse
import numpy as np
import jax.numpy as jnp
from operator import itemgetter
import action_type as action_type_lib

_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4 # Augment of the width
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4 # Augment of the height
_SWIPE_DISTANCE_THRESHOLD = 0.04 # Interval determining if an action is a tap or a swipe.

parser = argparse.ArgumentParser(description='Process input and output file paths.')
parser.add_argument("--input_dir", type=str, required=True, help="The path of the input file.")
parser.add_argument("--output_dir", type=str, required=True, help="The path of the output file.")
args = parser.parse_args()

def is_tap_action(normalized_start_yx, normalized_end_yx, swipe_distance_threshold):
    """
    Determines if the action between two points is a tap based on the distance.

    Args:
        normalized_start_yx (array-like): The starting (y, x) coordinates, normalized between 0 and 1.
        normalized_end_yx (array-like): The ending (y, x) coordinates, normalized between 0 and 1.
        swipe_distance_threshold (float): The maximum distance that still qualifies as a tap.

    Returns:
        bool: True if the action is a tap, False otherwise.
    """
    distance = jnp.linalg.norm(
        jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx)
    )
    return distance <= _SWIPE_DISTANCE_THRESHOLD

def _yx_in_bounding_boxes(yx, bounding_boxes):
  """Check if the (y,x) point is contained in each bounding box.

  Args:
    yx: The (y, x) coordinate in pixels of the point.
    bounding_boxes: A 2D int array of shape (num_bboxes, 4), where each row
      represents a bounding box: (y_top_left, x_top_left, box_height,
      box_width). Note: containment is inclusive of the bounding box edges.

  Returns:
    is_inside: A 1D bool array where each element specifies if the point is
      contained within the respective box.
  """
  y, x = yx

  # `bounding_boxes` has shape (n_elements, 4); we extract each array along the
  # last axis into shape (n_elements, 1), then squeeze unneeded dimension.
  top, left, height, width = [
      jnp.squeeze(v, axis=-1) for v in jnp.split(bounding_boxes, 4, axis=-1)
  ]

  # The y-axis is inverted for AndroidEnv, so bottom = top + height.
  bottom, right = top + height, left + width

  return jnp.logical_and(y >= top, y <= bottom) & jnp.logical_and(
      x >= left, x <= right)

def _check_drag_actions_match(drag_touch_yx, drag_lift_yx):
    """
    Determines the main direction of a drag action based on touch and lift coordinates.

    Args:
        drag_touch_yx (array-like): The starting (y, x) coordinates of the drag.
        drag_lift_yx (array-like): The ending (y, x) coordinates of the drag.

    Returns:
        str: The main direction of the drag, which can be "UP", "DOWN", "LEFT", or "RIGHT".
    """
    # Calculate the change in the y and x coordinates from touch to lift.
    drag_deltas = drag_lift_yx - drag_touch_yx
    drag_magnitudes = jnp.abs(drag_deltas)
    main_axis = jnp.argmax(drag_magnitudes)
    
    # Determine the main direction based on the axis with the greatest change.
    if main_axis == 0:  # y-axis
        return "UP" if drag_deltas[0] < 0 else "DOWN"
    else:  # x-axis
        return "LEFT" if drag_deltas[1] < 0 else "RIGHT"




def _resize_annotation_bounding_boxes(annotation_positions, annotation_width_augment_fraction, annotation_height_augment_fraction):
  """Resize the bounding boxes by the given fractions.

  Args:
    annotation_positions: Array of shape (N, 4), where each row represents the
      (y, x, height, width) of the bounding boxes.
    annotation_width_augment_fraction: The fraction to augment the box widths,
      E.g., 1.4 == 240% total increase.
    annotation_height_augment_fraction: Same as described for width, but for box
      height.

  Returns:
    Resized bounding box.

  """
  # print("annotation_positions: ",type(annotation_positions), annotation_positions)
  annotation_positions=annotation_positions.reshape([-1,4])
  height_change = (
      annotation_height_augment_fraction * annotation_positions[:, 2])
  width_change = (
      annotation_width_augment_fraction * annotation_positions[:, 3])

  # Limit bounding box positions to the screen.
  resized_annotations = jnp.stack([
      jnp.maximum(0, annotation_positions[:, 0] - (height_change / 2)),
      jnp.maximum(0, annotation_positions[:, 1] - (width_change / 2)),
      jnp.minimum(1, annotation_positions[:, 2] + height_change),
      jnp.minimum(1, annotation_positions[:, 3] + width_change),
  ],
                                  axis=1)
  return resized_annotations

def custom_translate(input_string, replace_dict):
    replace_dict = {
    '\\': '',
    '"': "'"
    }
    """Translate characters in input_string using the replace_dict mappings."""
    return ''.join(replace_dict.get(char, char) for char in input_string)

def find_nearest_bounding_box(result_touch_yx, bounding_boxes):
    """
    Find the nearest bounding box to a given point.
    
    Parameters:
    - result_touch_yx: A tuple (y, x) representing the point.
    - bounding_boxes: A list of tuples, where each tuple represents
                      a bounding box in the format (x, y, weight, height).
    
    Returns:
    - The bounding box that is closest to the point.
    """
    for box_no,bounding_box in enumerate(bounding_boxes):
        if(_yx_in_bounding_boxes(result_touch_yx,bounding_box)):
            return box_no
    ratio=1
    _NUMBER_ENLARGE=10
    for enlarge_i in _NUMBER_ENLARGE:
        ratio+=0.1*enlarge_i*enlarge_i
        for box_no,ui_position in enumerate(ui_positions):
            resize_ui_positions=_resize_annotation_bounding_boxes(ui_position,annotation_width_augment_fraction=ratio,annotation_height_augment_fraction=ratio)
            if(_yx_in_bounding_boxes(result_touch_yx,resize_ui_positions)):
                return box_no


if __name__ == '__main__':
    with open(input_dir, "rb") as rp, open(output_dir, 'w') as f:
        raw_data = pickle.load(rp)
        for file_i,data in enumerate(raw_data):
            history_behivor=""
            for sub_data in data["data"]:
                output=""
                keys = ['goal', 'step_id', 'ui_text', 'ui_type', 'ui_positions', 'result_touch_yx', 'result_lift_yx', 'result_action']
                goal, step_id, ui_text, ui_type, ui_positions, result_touch_yx, result_lift_yx, result_action = list(map(lambda key: custom_translate(sub_data.get(key)), keys))
                result_action_type, result_action_text= result_action


                ui=["id:{} ui_text:{} ui_type:{}".format(ui_i, ui_text[ui_i], ui_type[ui_i]) for ui_i in range(len(ui_text))]
                ui_screen="\n".join(ui)
                prompt = 'Given a mobile screen and a question, provide the action based on the screeninformation.\nPrevious Actions:{}\nScreen:\n{}\nInstruction:{}\nAnswer:'.format(history_behivor, ui_screen, goal)
                
                if(result_action_type=="TYPE"):
                    output = "action_type:{} typed_text:{}".format(result_action_type, result_action_text)
                elif(result_action_type=="DUAL_POINT"):
                    if(is_tap_action(result_touch_yx,result_lift_yx)):
                        box_no= find_nearest_bounding_box(result_touch_yx,ui_positions)
                        output = "action_type:{} ui_text:{} ui_type:{} id:{}".format(result_action_type, ui_text[box_no], ui_type[box_no], box_no) else "" if box_no>-1
                    else:
                        drags_match = _check_drag_actions_match(jnp.asarray(result_touch_yx),jnp.asarray(result_lift_yx))
                        output = "action_type:SCROLL direction:".format(drags_match)
                else:
                    output = "action_type:"+result_action_type
         
                output = custom_translate(output)
                
                write_prompt = '{"id":"{}##{}","instruction":"{}","input":"","output":"{}"}\n'.format(file_i, goal, prompt, output)
                write_prompt=custom_translate(write_prompt,{"\n":"\\n"})
                if(output!=""):
                    f.write(write_prompt)
                if(output.find(" id:")>-1):
                    output=output[:output.find(" id:")]
                history_behivor+= "step_id:{} {}\n".format(step_id, output)