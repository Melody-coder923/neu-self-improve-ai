#!/usr/bin/env python3                                                                 
"""Debug reward: print input format to verify TRL reward_funcs signature."""                        
                                                                                                    
def debug_reward(prompts, completions, **kwargs):                                                   
    print("prompts[0]:", prompts[0][:80])                                                           
    print("completions[0]:", completions[0][:80])                                                   
    print("kwargs keys:", list(kwargs.keys()))                                                      
    return [0.0] * len(completions)                                                                 
