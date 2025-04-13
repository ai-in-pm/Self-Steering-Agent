"""
Visualization components for Self-Steering Language Models Demo

This module contains visualization components to illustrate how
Self-Steering Language Models work.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import random
from typing import List, Dict, Any

def visualize_smc_process(num_particles: int = 8, num_steps: int = 5):
    """
    Visualize the Sequential Monte Carlo (SMC) process.
    
    Args:
        num_particles: Number of particles to visualize
        num_steps: Number of steps in the process
    """
    st.subheader("Sequential Monte Carlo (SMC) Visualization")
    st.markdown("""
    This visualization shows how the Sequential Monte Carlo (SMC) algorithm works in Self-Steering Language Models.
    Each row represents a particle (a candidate generation), and each column represents a step in the generation process.
    The color intensity represents the particle's weight, with darker colors indicating higher weights.
    """)
    
    # Initialize particles
    particles = [f"Particle {i+1}" for i in range(num_particles)]
    
    # Initialize weights
    weights = np.ones((num_particles, num_steps))
    
    # Simulate the SMC process
    for step in range(1, num_steps):
        # Update weights based on constraints
        for i in range(num_particles):
            # Randomly update weights to simulate constraint satisfaction
            weights[i, step] = weights[i, step-1] * random.uniform(0.1, 1.0)
        
        # Normalize weights
        weights[:, step] = weights[:, step] / np.sum(weights[:, step])
        
        # Simulate resampling by copying high-weight particles
        if step < num_steps - 1:
            # Find low-weight particles
            low_weight_indices = np.argsort(weights[:, step])[:num_particles//4]
            
            # Find high-weight particles
            high_weight_indices = np.argsort(weights[:, step])[-num_particles//4:]
            
            # Replace low-weight particles with copies of high-weight particles
            for i, low_idx in enumerate(low_weight_indices):
                high_idx = high_weight_indices[i % len(high_weight_indices)]
                weights[low_idx, step] = weights[high_idx, step]
    
    # Create a DataFrame for visualization
    data = []
    for i in range(num_particles):
        for j in range(num_steps):
            data.append({
                "Particle": particles[i],
                "Step": f"Step {j+1}",
                "Weight": weights[i, j]
            })
    
    df = pd.DataFrame(data)
    
    # Create a heatmap
    heatmap = pd.pivot_table(
        df, 
        values="Weight", 
        index="Particle", 
        columns="Step"
    )
    
    # Display the heatmap
    st.dataframe(
        heatmap,
        use_container_width=True,
        hide_index=False,
    )
    
    # Add a color legend
    st.markdown("""
    **Color Legend:**
    - Darker color = Higher weight (more likely to be selected)
    - Lighter color = Lower weight (less likely to be selected)
    """)
    
    return heatmap

def visualize_constraint_satisfaction(text: str, constraints: Dict[str, Any]):
    """
    Visualize how the generated text satisfies the constraints.
    
    Args:
        text: The generated text
        constraints: Dictionary of constraints
    """
    st.subheader("Constraint Satisfaction Visualization")
    
    if "word_positions" in constraints:
        # Visualize word position constraints
        words = text.split()
        word_positions = constraints["word_positions"]
        
        # Create a DataFrame for visualization
        data = []
        for i, word in enumerate(words):
            position = i + 1
            is_constrained = position in word_positions
            target_word = word_positions.get(position, "")
            satisfied = is_constrained and word.lower() == target_word.lower()
            
            data.append({
                "Position": position,
                "Word": word,
                "Constrained": is_constrained,
                "Target Word": target_word,
                "Satisfied": satisfied
            })
        
        df = pd.DataFrame(data)
        
        # Display the DataFrame
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Satisfied": st.column_config.CheckboxColumn(
                    "Constraint Satisfied",
                    help="Whether the word satisfies the constraint",
                    default=False,
                ),
                "Constrained": st.column_config.CheckboxColumn(
                    "Is Constrained",
                    help="Whether the position has a constraint",
                    default=False,
                ),
            },
        )
    
    elif "sentence_count" in constraints and "min_words_per_sentence" in constraints:
        # Visualize sentence constraints
        sentences = text.split(". ")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create a DataFrame for visualization
        data = []
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            word_count = len(words)
            satisfied = word_count >= constraints["min_words_per_sentence"]
            
            data.append({
                "Sentence": i + 1,
                "Word Count": word_count,
                "Min Words Required": constraints["min_words_per_sentence"],
                "Satisfied": satisfied
            })
        
        df = pd.DataFrame(data)
        
        # Display the DataFrame
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Satisfied": st.column_config.CheckboxColumn(
                    "Constraint Satisfied",
                    help="Whether the sentence satisfies the constraint",
                    default=False,
                ),
            },
        )
    
    elif "size" in constraints:
        # Visualize square poem constraints
        lines = text.strip().split("\n")
        
        # Create a DataFrame for visualization
        data = []
        for i, line in enumerate(lines):
            words = line.split()
            word_count = len(words)
            satisfied = word_count == constraints["size"]
            
            data.append({
                "Line": i + 1,
                "Word Count": word_count,
                "Target Words": constraints["size"],
                "Satisfied": satisfied
            })
        
        df = pd.DataFrame(data)
        
        # Display the DataFrame
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Satisfied": st.column_config.CheckboxColumn(
                    "Constraint Satisfied",
                    help="Whether the line satisfies the constraint",
                    default=False,
                ),
            },
        )
    
    elif "budget" in constraints and "max_items" in constraints:
        # Visualize budget constraints
        lines = text.strip().split("\n")
        
        # Remove the header
        if lines[0].startswith("Ingredients:"):
            lines = lines[1:]
        
        # Create a DataFrame for visualization
        data = []
        total_cost = 0.0
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Extract the cost
            import re
            match = re.search(r'\$(\d+(?:\.\d+)?)', line)
            if match:
                cost = float(match.group(1))
                total_cost += cost
                
                data.append({
                    "Item": i + 1,
                    "Ingredient": line.strip(),
                    "Cost": f"${cost:.2f}",
                    "Running Total": f"${total_cost:.2f}",
                    "Under Budget": total_cost <= constraints["budget"]
                })
        
        df = pd.DataFrame(data)
        
        # Display the DataFrame
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Under Budget": st.column_config.CheckboxColumn(
                    "Under Budget",
                    help="Whether the running total is under budget",
                    default=False,
                ),
            },
        )
        
        # Display the budget summary
        st.metric("Total Cost", f"${total_cost:.2f}", f"${constraints['budget'] - total_cost:.2f} under budget")

def animate_smc_execution(num_particles: int = 8, num_steps: int = 5):
    """
    Animate the SMC execution process.
    
    Args:
        num_particles: Number of particles to visualize
        num_steps: Number of steps in the process
    """
    st.subheader("SMC Execution Animation")
    
    # Create placeholders for the animation
    progress_bar = st.progress(0)
    status_text = st.empty()
    particles_container = st.empty()
    
    # Initialize particles
    particles = [{"text": "", "weight": 1.0} for _ in range(num_particles)]
    
    # Simulate the SMC process
    for step in range(num_steps):
        # Update progress
        progress = int((step + 1) / num_steps * 100)
        progress_bar.progress(progress)
        
        # Update status
        status_text.text(f"Step {step + 1}/{num_steps}: {'Initializing' if step == 0 else 'Extending' if step < num_steps - 1 else 'Finalizing'}")
        
        # Update particles
        for i in range(num_particles):
            # Extend the particle's text
            if step == 0:
                particles[i]["text"] = f"The"
            else:
                # Add a word based on the step
                words = ["beautiful", "mountain", "adventure", "inspires", "creative", "thoughts", "and", "feelings", "within", "us"]
                particles[i]["text"] += f" {words[step % len(words)]}"
            
            # Update the particle's weight
            particles[i]["weight"] *= random.uniform(0.5, 1.0)
        
        # Normalize weights
        total_weight = sum(p["weight"] for p in particles)
        for i in range(num_particles):
            particles[i]["weight"] /= total_weight
        
        # Display the particles
        particles_df = pd.DataFrame([
            {
                "Particle": f"Particle {i+1}",
                "Text": p["text"],
                "Weight": f"{p['weight']:.4f}"
            }
            for i, p in enumerate(particles)
        ])
        
        particles_container.dataframe(
            particles_df,
            use_container_width=True,
            hide_index=True,
        )
        
        # Simulate resampling
        if step < num_steps - 1:
            # Sort particles by weight
            particles.sort(key=lambda p: p["weight"])
            
            # Replace low-weight particles with copies of high-weight particles
            for i in range(num_particles // 4):
                particles[i] = particles[num_particles - i - 1].copy()
            
            # Reset weights after resampling
            for i in range(num_particles):
                particles[i]["weight"] = 1.0
        
        # Add a small delay for the animation
        time.sleep(0.5)
    
    # Final status
    status_text.text(f"Execution complete! Selected particle: {particles[-1]['text']}")
    
    return particles[-1]["text"]
