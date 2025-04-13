# Self-Steering Language Models Demo

This application demonstrates the Self-Steering Language Models approach described in the paper "Self-Steering Language Models" by Gabriel Grand et al. It shows how a Planner LM can generate inference programs to guide a Follower LM to solve complex constrained generation tasks. To read the full paper, visit https://arxiv.org/pdf/2504.07081.

![SS Interface Planner Output](https://github.com/user-attachments/assets/6c0ae5fc-342e-4abe-9771-ba9a28457eb5)
![SS Interface Planner Output II](https://github.com/user-attachments/assets/e259fd8b-0615-4d7c-8108-bb39589dc0f3)
![SS Interface Execution Results II](https://github.com/user-attachments/assets/fc1cc422-477b-4c49-b6c8-4a1a91555d81)
![SS Interface Budget Constraints](https://github.com/user-attachments/assets/dc047a0c-573a-42d5-8dcc-20b338ff15d8)

## About Self-Steering Language Models

Self-Steering Language Models is a method where:

1. A **Planner LM** generates task-specific inference programs
2. These programs are executed by **Follower LMs**
3. The approach enables efficient, verifiable reasoning

The DisCIPL (Distributional Constraints by Inference Programming with Language Models) framework decouples planning from execution, opening up a design space of highly-parallelized Monte Carlo inference strategies that outperform standard best-of-N sampling, require no fine-tuning, and can be implemented automatically by existing LMs.

## Key Concepts from the Paper

- **Planner LM**: A capable language model that generates inference programs based on task descriptions
- **Follower LM**: A smaller language model that executes the inference programs
- **Sequential Monte Carlo (SMC)**: A probabilistic inference algorithm that maintains multiple candidate generations (particles) and adaptively reallocates computational resources to promising candidates
- **Inference Programming Patterns**: Common patterns like token masking, self-hinting, and self-checking that help guide the generation process

## Features

- Interactive task selection for different constrained generation challenges
- Visualization of the Planner LM's inference program generation
- Simulation of the Follower LM's execution process using Sequential Monte Carlo (SMC)
- Performance metrics and verification of constraint satisfaction
- Visualization of constraint satisfaction in the generated text

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. The `.env` file already contains an OpenAI API key. If you want to use your own key, update the file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the App

#### Option 1: Using the run script

On Windows, simply double-click the `run.bat` file or run it from the command line:
```
run.bat
```

#### Option 2: Manual execution

```
pip install -r requirements.txt
streamlit run app.py
```

### Testing the API Connection

To test if your OpenAI API key is working correctly, run:
```
python test_api.py
```

## Example Tasks

### Sentence Generation
Generate a sentence with specific words at exact positions. For example, a sentence with exactly 15 words, where the 3rd word is "beautiful", the 7th word is "mountain", and the 10th word is "adventure".

### Paragraph Generation
Write a paragraph with sentence-level constraints. For example, a paragraph with exactly 3 sentences, where each sentence has at least 10 words.

### Poetry
Create a poem with structural requirements. For example, a square poem with 5 lines, where each line has exactly 5 words.

### Budget-Constrained List
Generate an ingredients list for a recipe with a budget constraint. For example, an ingredients list for chocolate chip cookies with at most 7 ingredients costing less than $20.00 total.

## How It Works

1. **Task Selection**: The user selects a constrained generation task and specifies the constraints.
2. **Planner Generation**: The Planner LM (GPT-4o) generates an inference program that encodes the task requirements and a step-by-step generation strategy.
3. **Follower Execution**: The Follower LM executes the inference program, which guides it through the generation process using Sequential Monte Carlo.
4. **Visualization**: The app visualizes the execution process, showing how particles are initialized, extended, scored, and resampled based on constraint satisfaction.

## Project Structure

- `app.py`: Main Streamlit application
- `inference_models.py`: Base classes and implementations for inference models
- `visualization.py`: Visualization components for the SMC process
- `requirements.txt`: Required Python packages
- `run.bat`: Script to install dependencies and run the app
- `test_api.py`: Script to test the OpenAI API connection
- `.env`: Environment variables file with the OpenAI API key

## Paper Reference

Grand, G., Tenenbaum, J. B., Mansinghka, V. K., Lew, A. K., & Andreas, J. (2025). Self-Steering Language Models. Preprint. Under review.
