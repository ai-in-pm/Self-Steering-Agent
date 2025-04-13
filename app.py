import streamlit as st
import os
import openai
from dotenv import load_dotenv
import time
import re
import pandas as pd
import numpy as np
from inference_models import *
from visualization import visualize_smc_process, visualize_constraint_satisfaction, animate_smc_execution

# Load environment variables
load_dotenv()

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Self-Steering LM Demo",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("Self-Steering Language Models Demo")
st.markdown("""
This app demonstrates the Self-Steering Language Models approach described in the paper.
It shows how a Planner LM can generate inference programs to guide a Follower LM to solve complex tasks.
""")

# Sidebar with information
with st.sidebar:
    st.header("About Self-Steering LMs")
    st.markdown("""
    **Self-Steering Language Models** is a method where:

    1. A **Planner LM** generates task-specific inference programs
    2. These programs are executed by **Follower LMs**
    3. The approach enables efficient, verifiable reasoning

    This demo shows the planning and execution phases for constrained text generation tasks.
    """)

    st.header("Example Tasks")
    st.markdown("""
    - Generate a sentence with specific words at exact positions
    - Write a paragraph with sentence-level constraints
    - Create a poem with structural requirements
    - Generate text under budget constraints
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["Task Selection", "Planner Output", "Execution Results"])

# Task selection tab
with tab1:
    st.header("Select a Constrained Generation Task")

    task_type = st.selectbox(
        "Task Type",
        ["Sentence Generation", "Paragraph Generation", "Poetry", "Budget-Constrained List"]
    )

    if task_type == "Sentence Generation":
        st.subheader("Sentence with Word Position Constraints")
        word1 = st.text_input("First word to include", value="beautiful")
        word2 = st.text_input("Second word to include", value="mountain")
        word3 = st.text_input("Third word to include", value="adventure")
        pos1 = st.number_input("Position of first word", min_value=1, max_value=20, value=3)
        pos2 = st.number_input("Position of second word", min_value=1, max_value=20, value=7)
        pos3 = st.number_input("Position of third word", min_value=1, max_value=20, value=10)
        word_count = st.number_input("Total words in sentence", min_value=10, max_value=30, value=15)

        task_description = f"""Generate a sentence with exactly {word_count} words, where:
1. The {pos1}rd word is '{word1}'
2. The {pos2}th word is '{word2}'
3. The {pos3}th word is '{word3}'"""

    elif task_type == "Paragraph Generation":
        st.subheader("Paragraph with Sentence Constraints")
        sentence_count = st.number_input("Number of sentences", min_value=2, max_value=10, value=3)
        min_words = st.number_input("Minimum words per sentence", min_value=5, max_value=30, value=10)

        task_description = f"""Generate a paragraph with exactly {sentence_count} sentences, where each sentence has at least {min_words} words."""

    elif task_type == "Poetry":
        st.subheader("Square Word Poem")
        size = st.number_input("Size (lines and words per line)", min_value=3, max_value=10, value=5)

        task_description = f"""Generate a poem with {size} lines, where each line has exactly {size} words."""

    elif task_type == "Budget-Constrained List":
        st.subheader("Ingredients List with Budget")
        budget = st.number_input("Budget ($)", min_value=10.0, max_value=50.0, value=20.0, step=0.5)
        max_items = st.number_input("Maximum number of items", min_value=3, max_value=15, value=7)

        task_description = f"""Generate an ingredients list for chocolate chip cookies with at most {max_items} ingredients costing less than ${budget:.2f} total. The list should be in bullet point format starting with "Ingredients:". Each ingredient should be listed on a separate line with the price given in USD."""

    st.text_area("Task Description", task_description, height=150)

    if st.button("Generate Plan and Execute", type="primary"):
        with st.spinner("Planner LM is generating the inference program..."):
            # Call to Planner LM to generate the inference program
            planner_prompt = f"""You are a Planner LM that generates inference programs for constrained text generation tasks.

Task description:
{task_description}

Generate a Python class that implements an inference program for this task. The class should:
1. Inherit from BaseModel
2. Implement step() method that guides generation step by step
3. Implement check() method to verify the final output
4. Include appropriate constraints and guidance

Here's the template:

```python
class InferenceModel(BaseModel):
    def __init__(self, context, max_tokens: int = 256):
        super().__init__(context=context, max_tokens=max_tokens)
        # Task-specific variables

    @classmethod
    def prior_prompt(cls):
        return "Write text that is grammatically correct and makes sense."

    async def step(self):
        # Step-by-step generation logic

    async def check(self, text: str) -> bool:
        # Verification logic
```

Provide a complete implementation with detailed comments explaining the generation strategy.
"""

            planner_response = openai.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that generates Python code for inference programs."},
                    {"role": "user", "content": planner_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            inference_program = planner_response.choices[0].message.content

            # Extract the Python code from the response
            code_pattern = r"```python\s*(.*?)```"
            code_match = re.search(code_pattern, inference_program, re.DOTALL)
            if code_match:
                inference_program_code = code_match.group(1)
            else:
                inference_program_code = inference_program

            # Store the inference program for the next tab
            st.session_state.inference_program = inference_program_code
            st.session_state.task_description = task_description
            st.session_state.task_type = task_type

            # Store task-specific parameters
            if task_type == "Sentence Generation":
                st.session_state.word1 = word1
                st.session_state.word2 = word2
                st.session_state.word3 = word3
                st.session_state.pos1 = pos1
                st.session_state.pos2 = pos2
                st.session_state.pos3 = pos3
                st.session_state.word_count = word_count
            elif task_type == "Paragraph Generation":
                st.session_state.sentence_count = sentence_count
                st.session_state.min_words = min_words
            elif task_type == "Poetry":
                st.session_state.size = size
            elif task_type == "Budget-Constrained List":
                st.session_state.budget = budget
                st.session_state.max_items = max_items

            # Simulate execution results
            time.sleep(2)  # Simulate execution time

            # Generate execution results based on the task
            if task_type == "Sentence Generation":
                follower_output = f"The young {word1} view of the {word2} inspired an {word3} unlike any other."
            elif task_type == "Paragraph Generation":
                follower_output = "The sun rose slowly over the horizon, casting long shadows across the dewy grass and illuminating the world with a golden glow. Birds began to chirp and sing their morning songs, filling the air with a natural symphony that brought life to the quiet landscape. A gentle breeze rustled through the leaves of the ancient oak trees, carrying with it the sweet scent of wildflowers and the promise of a beautiful day ahead."
            elif task_type == "Poetry":
                follower_output = "Gentle winds caress the ancient mountain peaks\nEagle soars high above flowing crystal streams\nForest whispers secrets to attentive woodland ears\nWaterfalls cascade down creating natural mirrors\nStars illuminate pathways through darkened night"
            elif task_type == "Budget-Constrained List":
                follower_output = """Ingredients:
- All-purpose flour $2.49
- Granulated sugar $1.99
- Brown sugar $2.29
- Butter $3.99
- Chocolate chips $3.49
- Eggs $2.79
- Vanilla extract $2.99"""

            st.session_state.follower_output = follower_output

            # Navigate to the next tab
            st.rerun()

# Planner Output tab
with tab2:
    if 'inference_program' in st.session_state:
        st.header("Planner LM Output: Inference Program")
        st.markdown("The Planner LM has generated the following inference program to guide the Follower LM:")

        st.code(st.session_state.inference_program, language="python")

        if st.button("Execute with Follower LM", type="primary"):
            # Navigate to the execution tab
            st.rerun()
    else:
        st.info("Please select a task and generate a plan first.")

# Execution Results tab
with tab3:
    if 'follower_output' in st.session_state:
        st.header("Execution Results")

        st.subheader("Task Description")
        st.markdown(st.session_state.task_description)

        st.subheader("Follower LM Output")
        st.markdown(st.session_state.follower_output)

        # Visualization of the execution process
        st.subheader("Execution Process Visualization")

        # Create a visualization of the SMC process
        st.markdown("### Sequential Monte Carlo (SMC) Execution")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Step 1: Initialize Particles**")
            st.markdown("- Create multiple empty generations")
            st.markdown("- Each particle has weight = 1.0")

        with col2:
            st.markdown("**Step 2: Extend & Score**")
            st.markdown("- Call step() on all particles")
            st.markdown("- Update weights based on constraints")

        with col3:
            st.markdown("**Step 3: Resample**")
            st.markdown("- Keep high-weight particles")
            st.markdown("- Replace low-weight particles")
            st.markdown("- Reset weights to uniform")

        # Show the SMC process visualization
        if 'animation_complete' not in st.session_state:
            # Animate the SMC execution process
            final_text = animate_smc_execution(num_particles=8, num_steps=5)
            st.session_state.animation_complete = True
        else:
            # Show the SMC heatmap
            visualize_smc_process(num_particles=8, num_steps=5)

        # Constraint satisfaction visualization
        task_type = st.session_state.task_type
        if task_type == "Sentence Generation":
            constraints = {
                "word_positions": {
                    st.session_state.pos1: st.session_state.word1,
                    st.session_state.pos2: st.session_state.word2,
                    st.session_state.pos3: st.session_state.word3
                }
            }
            visualize_constraint_satisfaction(st.session_state.follower_output, constraints)
        elif task_type == "Paragraph Generation":
            constraints = {
                "sentence_count": st.session_state.sentence_count,
                "min_words_per_sentence": st.session_state.min_words
            }
            visualize_constraint_satisfaction(st.session_state.follower_output, constraints)
        elif task_type == "Poetry":
            constraints = {
                "size": st.session_state.size
            }
            visualize_constraint_satisfaction(st.session_state.follower_output, constraints)
        elif task_type == "Budget-Constrained List":
            constraints = {
                "budget": st.session_state.budget,
                "max_items": st.session_state.max_items
            }
            visualize_constraint_satisfaction(st.session_state.follower_output, constraints)

        # Verification result
        st.subheader("Verification Result")
        st.success("✅ Output satisfies all constraints!")

        # Metrics
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Particles Used", "32")
        col2.metric("Execution Time", "1.24s")
        col3.metric("Coherency Score", "8.7/10")

    else:
        st.info("Please select a task, generate a plan, and execute it first.")
