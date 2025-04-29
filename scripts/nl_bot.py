import rospy
from threading import Thread
import time
import cv2
import base64
from io import BytesIO
from PIL import Image

class NL_BOT:
    def __init__(self, mm_llm_api_key=None):
        """
        Initialize the NL-BOT system.
        
        Args:
            mm_llm_api_key: API key for the multimodal LLM (if needed)
        """
        # Store API key for multimodal LLM
        self.mm_llm_api_key = mm_llm_api_key
        
        # Initialize robot controller
        self.robot_controller = RobotController()
        
        # Initialize action engine
        self.action_engine = ActionEngine(self.robot_controller)
        
        # Initialize prompt engine
        self.prompt_engine = PromptEngine()
        
        # Processing flag to ensure we don't process multiple instructions at once
        self.is_processing = False
        
        # Main processing loop
        self.running = True
        self.processing_thread = Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        rospy.loginfo("NL-BOT system initialized and ready")
    
    def processing_loop(self):
        """Main processing loop to handle instructions."""
        rate = rospy.Rate(1)  # Check for new instructions at 1 Hz
        while self.running and not rospy.is_shutdown():
            # TODO: Check for new instructions from a source (e.g., user input, topic subscription)
            rate.sleep()
    
    def encode_image(self, image):
        """Encode an image as base64 for API calls."""
        # Convert OpenCV image to PIL Image and then to base64
        if image is None:
            return None
        
        # Convert from OpenCV BGR to RGB color format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Save to buffer and encode
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def call_multimodal_llm(self, prompt, image):
        """
        Call a multimodal LLM API with the prompt and image.
        
        Args:
            prompt: Text prompt
            image: Camera image (OpenCV format)
        
        Returns:
            LLM response text
        """
        # Encode image for API if it exists
        encoded_image = self.encode_image(image) if image is not None else None
        
        # TODO: Implement actual API call to the multimodal LLM
        
        # Fallback for testing when no image or API key is available
        fallback_responses = [
            "I can see a a chair at the end. SELECTED ACTION: [1]",
            "I can see a table next to the window. SELECTED ACTION: [5]",
            "There appears to be an obstacle ahead. SELECTED ACTION: [3]"
        ]
        import random
        return random.choice(fallback_responses)
    
    def process_instruction(self, instruction):
        """
        Process a natural language instruction.
        
        Args:
            instruction: Natural language instruction from user
            
        Returns:
            dict with action, observation, and LLM response
        """
        if self.is_processing:
            return {"error": "Already processing an instruction, please wait"}
        
        self.is_processing = True
        
        try:
            # Get action set from action engine
            action_set = self.action_engine.get_action_set()
            
            # Generate prompt for LLM
            prompt = self.prompt_engine.generate_prompt(instruction, action_set)
            
            # Get current camera image
            image = self.robot_controller.get_current_image()
            if image is None:
                rospy.logwarn("No camera image available")
            
            # Call multi-modal LLM with prompt and image
            llm_response = self.call_multimodal_llm(prompt, image)
            
            # Parse LLM response to get the chosen action
            action_name = self.action_engine.parse_llm_response(llm_response)
            
            # Execute the action
            observation = self.action_engine.execute_action(action_name)
            
            # Update navigation history
            self.prompt_engine.add_to_history(action_name, observation)
            
            return {
                "action": action_name,
                "observation": observation,
                "llm_response": llm_response
            }
        
        except Exception as e:
            rospy.logerr(f"Error processing instruction: {e}")
            return {"error": f"Failed to process instruction: {str(e)}"}
        
        finally:
            self.is_processing = False
    
    def run_continuous_navigation(self, instruction):
        """
        Continuously navigate based on an instruction until goal is reached
        or maximum steps are taken.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            Summary of navigation attempt
        """
        max_steps = 20
        step_count = 0
        goal_reached = False
        
        rospy.loginfo(f"Starting continuous navigation: '{instruction}'")
        
        while step_count < max_steps and not goal_reached and not rospy.is_shutdown():
            # Process the instruction
            result = self.process_instruction(instruction)
            
            if "error" in result:
                rospy.logerr(f"Navigation failed: {result['error']}")
                return f"Navigation failed: {result['error']}"
            
            step_count += 1
            
            # Check if the LLM indicates goal reached
            if "goal reached" in result["llm_response"].lower() or "destination reached" in result["llm_response"].lower():
                goal_reached = True
                rospy.loginfo("Goal reached!")
            
            # Give the robot time to execute the action and update sensors
            time.sleep(1)
        
        # Navigation summary
        if goal_reached:
            return f"Successfully reached the goal in {step_count} steps"
        else:
            return f"Failed to reach the goal in {max_steps} steps"
    
    def shutdown(self):
        """Shutdown the NL-BOT system."""
        self.running = False
        self.robot_controller.stop()
        rospy.loginfo("NL-BOT system shutdown")