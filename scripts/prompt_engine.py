class PromptEngine:
    def __init__(self):
        self.system_prompt = """
        You are NL-BOT, a natural language navigation system for the Triton robot.
        Your task is to help the robot navigate through an indoor environment based on natural language instructions.
        You will be provided with:
        1. A natural language instruction
        2. The robot's current visual observation
        3. A history of previous navigation actions
        4. A set of available actions

        You must select the best action for the robot to take to follow the instruction.
        Only select actions from the provided set.
        Explain your reasoning before selecting an action.
        """
        self.navigation_history = []
        self.max_history_length = 10  # Keep last 10 actions
        
    def add_to_history(self, action, observation_description):
        """Add an action and its result to the navigation history."""
        entry = f"Action: {action}, Result: {observation_description}"
        self.navigation_history.append(entry)
        if len(self.navigation_history) > self.max_history_length:
            self.navigation_history.pop(0)  # Remove oldest entry
    
    def format_action_set(self, action_set):
        """Format the action set for the prompt."""
        action_descriptions = []
        for idx, (action_name, action_desc) in enumerate(action_set.items(), 1):
            action_descriptions.append(f"{idx}. {action_name}: {action_desc}")
        return "\n".join(action_descriptions)
    
    def generate_prompt(self, instruction, action_set):
        """Generate the full prompt for the LLM."""
        action_descriptions = self.format_action_set(action_set)
        history_text = "\n".join(self.navigation_history) if self.navigation_history else "No previous actions."
        
        prompt = f"""
        {self.system_prompt}

        ## INSTRUCTION
        {instruction}

        ## NAVIGATION HISTORY
        {history_text}

        ## AVAILABLE ACTIONS
        {action_descriptions}

        ## VISUAL OBSERVATION
        [The current camera feed from the robot will be processed by the multi-modal LLM]

        Based on the above information, analyze the situation and select the most appropriate action to follow the instruction.
        First, explain your reasoning.
        Then, state your chosen action in the format: SELECTED ACTION: [action number]
        """
        return prompt