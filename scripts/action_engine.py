from enum import Enum
import re

class ActionType(Enum):
    FINE_GRAINED = 1  # Basic movements
    HIGH_LEVEL = 2    # Goal-oriented actions

class ActionEngine:
    def __init__(self, robot_controller):
        """
        Initialize the Action Engine with a robot controller.
        
        Args:
            robot_controller: Interface to control the Triton robot
        """
        self.robot_controller = robot_controller
        
        # Define the action set
        self.action_set = {
            # Fine-grained actions
            "move_forward": "Move the robot forward by 0.5 meters",
            "turn_left": "Rotate the robot 90 degrees counter-clockwise",
            "turn_right": "Rotate the robot 90 degrees clockwise",
            "stop": "Stop the robot's movement",
            
            # High-level actions
            "move_to_the_target": "Navigate to the identified target object or location",
            "explore": "Look around to gather more visual information",
            "backtrack": "Return to the previous position"
        }
        
        # Define action type mapping
        self.action_types = {
            "move_forward": ActionType.FINE_GRAINED,
            "turn_left": ActionType.FINE_GRAINED,
            "turn_right": ActionType.FINE_GRAINED,
            "stop": ActionType.FINE_GRAINED,
            "move_to_the_target": ActionType.HIGH_LEVEL,
            "explore": ActionType.HIGH_LEVEL,
            "backtrack": ActionType.HIGH_LEVEL
        }
    
    def get_action_set(self):
        """Return the current action set."""
        return self.action_set
    
    def parse_llm_response(self, llm_response):
        """
        Parse the LLM's response to extract the chosen action.
        
        Args:
            llm_response: String response from the LLM
            
        Returns:
            action_name: String name of the selected action
        """
        # Look for the SELECTED ACTION pattern
        pattern = r"SELECTED ACTION:\s*\[?(\d+)\]?"
        match = re.search(pattern, llm_response)
        
        if match:
            action_index = int(match.group(1)) - 1  # Convert to 0-based index
            if 0 <= action_index < len(self.action_set):
                action_name = list(self.action_set.keys())[action_index]
                return action_name
        
        # Fallback: search for action names directly in the response
        for action_name in self.action_set.keys():
            if action_name in llm_response.lower():
                return action_name
        
        # Default to stop if no action could be determined
        return "stop"
    
    def execute_action(self, action_name):
        """
        Execute the selected action on the Triton robot.
        
        Args:
            action_name: String name of the action to execute
            
        Returns:
            observation: String description of the result
        """
        if action_name not in self.action_set:
            return "Invalid action. Robot did not move."
        
        # Execute the appropriate action based on type
        if self.action_types[action_name] == ActionType.FINE_GRAINED:
            return self._execute_fine_grained_action(action_name)
        else:
            return self._execute_high_level_action(action_name)
    
    def _execute_fine_grained_action(self, action_name):
        """Execute a fine-grained movement action."""
        if action_name == "move_forward":
            self.robot_controller.move_forward(0.5)  # Move forward 0.5 meters
            return "Robot moved forward."
        
        elif action_name == "turn_left":
            self.robot_controller.rotate(-90)  # Rotate 90 degrees counter-clockwise
            return "Robot turned left."
        
        elif action_name == "turn_right":
            self.robot_controller.rotate(90)  # Rotate 90 degrees clockwise
            return "Robot turned right."
        
        elif action_name == "stop":
            self.robot_controller.stop()
            return "Robot stopped."
    
    def _execute_high_level_action(self, action_name):
        """Execute a high-level goal-oriented action."""
        if action_name == "move_to_the_target":
            # This would involve object detection and path planning
            # Simplified implementation for demonstration
            self.robot_controller.navigate_to_target()
            return "Robot is navigating to the identified target."
        
        elif action_name == "explore":
            # Look around to gather more visual information
            self.robot_controller.rotate(360, speed=20)  # Slow 360-degree scan
            return "Robot performed a full scan of the surroundings."
        
        elif action_name == "backtrack":
            # Return to the previous position
            self.robot_controller.backtrack()
            return "Robot returned to the previous position."