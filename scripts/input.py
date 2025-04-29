#!/usr/bin/env python

import rospy
import sys

def main():
    # Initialize the NL-BOT system
    mm_llm_api_key = "your_api_key_here"  
    nl_bot = NL_BOT(mm_llm_api_key)
    
    try:
        # Process a single instruction
        instruction = "Turn left at the second door"
        
        # Option 1: Process a single step
        result = nl_bot.process_instruction(instruction)
        print(f"Executed action: {result['action']}")
        print(f"Observation: {result['observation']}")
        
        # Option 2: Run continuous navigation until goal is reached
        # summary = nl_bot.run_continuous_navigation(instruction)
        # print(f"Navigation summary: {summary}")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Ensure proper shutdown
        nl_bot.shutdown()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass