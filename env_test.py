from splendor_env import SplendorEnv
import numpy as np

def test_raw_environment():
    """Test the raw SplendorEnv without any wrappers"""
    print("Testing raw SplendorEnv...")
    
    env = SplendorEnv()
    obs, info = env.reset(seed=42)
    
    step_count = 0
    empty_mask_count = 0
    
    print(f"Initial agent: {env.agent_selection}")
    print(f"Initial mask sum: {obs['action_mask'].sum()}")
    
    for step in range(500):  # Test many steps
        current_agent = env.agent_selection
        obs = env.observe(current_agent)
        mask = obs['action_mask']
        
        if mask.sum() == 0:
            empty_mask_count += 1
            print(f"\n[EMPTY MASK #{empty_mask_count}] Step {step}")
            print(f"  Current agent: {current_agent}")
            print(f"  Terminations: {env.terminations}")
            print(f"  Truncations: {env.truncations}")
            
            if env.game:
                agent_idx = env.agent_name_mapping[current_agent]
                try:
                    legal_actions = env.game.legal_actions(agent_idx)
                    print(f"  Direct legal actions: {len(legal_actions)}")
                    print(f"  Game VPs: {[p.VPs for p in env.game.players]}")
                    print(f"  Game gems: {[p.gems.sum() for p in env.game.players]}")
                except Exception as e:
                    print(f"  Error getting legal actions: {e}")
            
            # This should not happen in a working game
            print("  CRITICAL: Empty mask detected!")
            break
        
        # Take random valid action
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            action_idx = np.random.choice(valid_indices)
            env.step(action_idx)
            step_count += 1
            
            # Check if game ended
            if any(env.terminations.values()) or any(env.truncations.values()):
                print(f"\nGame ended at step {step}")
                print(f"Terminations: {env.terminations}")
                print(f"Rewards: {env.rewards}")
                break
        else:
            print(f"No valid actions at step {step} - this shouldn't happen!")
            break
    
    print(f"\nTest completed. Steps taken: {step_count}")
    print(f"Empty masks encountered: {empty_mask_count}")
    

def test_action_consistency():
    """Test if all_actions and legal_actions are consistent"""
    print("\nTesting action consistency...")
    
    env = SplendorEnv()
    obs, info = env.reset(seed=123)
    
    # Check all_actions vs legal_actions consistency
    print(f"Total actions in all_actions: {len(env.all_actions)}")
    
    for step in range(50):
        current_agent = env.agent_selection
        agent_idx = env.agent_name_mapping[current_agent]
        
        # Get legal actions directly from game
        legal_actions_raw = env.game.legal_actions(agent_idx)
        
        # Get legal actions through observe
        obs = env.observe(current_agent)
        mask = obs['action_mask']
        legal_indices = np.where(mask)[0]
        legal_actions_from_mask = [env.all_actions[i] for i in legal_indices]
        
        # Compare
        if set(legal_actions_raw) != set(legal_actions_from_mask):
            print(f"MISMATCH at step {step}:")
            print(f"  Raw legal: {sorted(legal_actions_raw)}")
            print(f"  From mask: {sorted(legal_actions_from_mask)}")
            print(f"  Missing from mask: {set(legal_actions_raw) - set(legal_actions_from_mask)}")
            print(f"  Extra in mask: {set(legal_actions_from_mask) - set(legal_actions_raw)}")
            break
        
        if len(legal_actions_raw) == 0:
            print(f"Empty legal actions at step {step} - game should be terminated")
            break
            
        # Take action
        action_to_take = legal_actions_raw[0]  # Take first legal action
        action_idx = env.all_actions.index(action_to_take)
        env.step(action_idx)
        
        if any(env.terminations.values()):
            break
    
    print("Action consistency test completed.")


if __name__ == "__main__":
    test_raw_environment()
    test_action_consistency()