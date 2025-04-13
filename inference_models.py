"""
Inference Models for Self-Steering Language Models Demo

This module contains base classes and implementations for the inference models
used in the Self-Steering Language Models demonstration.
"""

import re
import string
import asyncio
from typing import List, Tuple, Optional, Dict, Any, Set
import datetime

class BaseModel:
    """Base class for all inference models."""
    
    def __init__(self, context, max_tokens: int = 256):
        """
        Initialize the base model.
        
        Args:
            context: The context for the model (in a real implementation, this would be the LM context)
            max_tokens: Maximum number of tokens to generate
        """
        self.context = context
        self.max_tokens = max_tokens
    
    @classmethod
    def prior_prompt(cls):
        """Return the prior prompt for the model."""
        return "Write text that is grammatically correct and makes sense."
    
    async def hint(self, hint_text: str):
        """
        Provide a hint to the Follower LM.
        
        Args:
            hint_text: The hint text to provide
        """
        # In a real implementation, this would inject the hint into the Follower LM's context
        print(f"Hint: {hint_text}")
    
    async def next_word(self):
        """Sample the next word from the Follower LM."""
        # In a real implementation, this would sample from the Follower LM
        return "word"
    
    async def next_token(self):
        """Sample the next token from the Follower LM."""
        # In a real implementation, this would sample from the Follower LM
        return "token"
    
    async def extend(self, start: str = "", stop: List[str] = None, allow_eos: bool = False):
        """
        Extend the current generation with a sequence of tokens.
        
        Args:
            start: The starting text for the extension
            stop: List of stop sequences
            allow_eos: Whether to allow EOS token
            
        Returns:
            Tuple of (generated_text, eos_sampled)
        """
        # In a real implementation, this would extend the generation using the Follower LM
        return f"{start} extended text", False
    
    async def extend_with(self, text: str):
        """
        Extend the current generation with the given text.
        
        Args:
            text: The text to extend with
        """
        # In a real implementation, this would extend the generation with the given text
        print(f"Extending with: {text}")
    
    async def sample(self, mask):
        """
        Sample from the Follower LM with a mask.
        
        Args:
            mask: The mask to apply
            
        Returns:
            Boolean indicating whether the sample was successful
        """
        # In a real implementation, this would sample from the Follower LM with a mask
        return True
    
    async def observe(self, text: str):
        """
        Force the Follower LM to generate the given text.
        
        Args:
            text: The text to force
        """
        # In a real implementation, this would force the Follower LM to generate the given text
        print(f"Observing: {text}")
    
    def condition(self, condition: bool):
        """
        Apply a condition to the current generation.
        
        Args:
            condition: The condition to apply
        """
        # In a real implementation, this would apply the condition to the current generation
        if not condition:
            print("Condition failed, rejecting generation")
    
    async def end(self):
        """End the current generation."""
        # In a real implementation, this would end the current generation
        print("Ending generation")


class TokenMask:
    """Base class for token masks."""
    
    def __init__(self, model):
        """
        Initialize the token mask.
        
        Args:
            model: The model to apply the mask to
        """
        self.model = model
    
    async def __aenter__(self):
        """Enter the context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        pass


class PunctuationMask(TokenMask):
    """Mask that only allows punctuation tokens."""
    pass


class EOSMask(TokenMask):
    """Mask that only allows EOS token."""
    pass


class NewLineMask(TokenMask):
    """Mask that only allows newline tokens."""
    
    def __init__(self, model, n=1):
        """
        Initialize the newline mask.
        
        Args:
            model: The model to apply the mask to
            n: Number of newlines to generate
        """
        super().__init__(model)
        self.n = n


class TokenLengthMask(TokenMask):
    """Mask that restricts tokens based on character length."""
    
    def __init__(self, model, max_chars=None):
        """
        Initialize the token length mask.
        
        Args:
            model: The model to apply the mask to
            max_chars: Maximum number of characters allowed
        """
        super().__init__(model)
        self.max_chars = max_chars


# Example inference models for different tasks

class SentenceWithWordPositionsModel(BaseModel):
    """
    Generates a sentence with specific words at exact positions.
    """
    
    def __init__(
        self,
        context,
        max_tokens: int = 256,
        word_count: int = 15,
        word_positions: Dict[int, str] = None
    ):
        """
        Initialize the model.
        
        Args:
            context: The context for the model
            max_tokens: Maximum number of tokens to generate
            word_count: Total number of words in the sentence
            word_positions: Dictionary mapping positions (1-indexed) to words
        """
        super().__init__(context=context, max_tokens=max_tokens)
        self.word_count = word_count
        self.word_positions = word_positions or {}
        self.current_position = 0
    
    @classmethod
    def prior_prompt(cls):
        return "Write a sentence that is grammatically correct and makes sense."
    
    async def step(self):
        """
        Generation strategy:
        - For each position, check if it's a constrained position
        - If it is, force the word at that position
        - If not, sample a word freely
        - Continue until word_count is reached
        """
        self.current_position += 1
        
        # Provide a hint about the current position
        await self.hint(f"Generating word {self.current_position} of {self.word_count}")
        
        # Check if the current position has a constrained word
        if self.current_position in self.word_positions:
            # Force the constrained word
            await self.observe(self.word_positions[self.current_position])
        else:
            # Sample a word freely
            await self.next_word()
        
        # If we've reached the word count, end generation
        if self.current_position >= self.word_count:
            await self.end()
    
    async def check(self, text: str) -> bool:
        """
        Check that the generated text satisfies the word constraints.
        
        Args:
            text: The generated text
            
        Returns:
            Boolean indicating whether the text satisfies the constraints
        """
        # Split the text into words
        words = text.split()
        
        # Check the word count
        if len(words) != self.word_count:
            return False
        
        # Check each constrained position
        for position, word in self.word_positions.items():
            if position > len(words) or words[position - 1].lower() != word.lower():
                return False
        
        return True


class ParagraphWithSentenceConstraintsModel(BaseModel):
    """
    Generates a paragraph with constraints on the number of sentences and words per sentence.
    """
    
    def __init__(
        self,
        context,
        max_tokens: int = 512,
        sentence_count: int = 3,
        min_words_per_sentence: int = 10
    ):
        """
        Initialize the model.
        
        Args:
            context: The context for the model
            max_tokens: Maximum number of tokens to generate
            sentence_count: Number of sentences in the paragraph
            min_words_per_sentence: Minimum number of words per sentence
        """
        super().__init__(context=context, max_tokens=max_tokens)
        self.sentence_count = sentence_count
        self.min_words_per_sentence = min_words_per_sentence
        self.current_sentence = 0
        self.current_word_in_sentence = 0
    
    @classmethod
    def prior_prompt(cls):
        return "Write a paragraph that is grammatically correct and makes sense."
    
    async def step(self):
        """
        Generation strategy:
        - Generate sentences word by word
        - Ensure each sentence has at least min_words_per_sentence
        - After reaching min_words_per_sentence, allow (but don't force) end punctuation
        - Continue until sentence_count is reached
        """
        # Increment the word counter
        self.current_word_in_sentence += 1
        
        # Provide a hint about the current sentence and word
        await self.hint(
            f"Sentence {self.current_sentence + 1}/{self.sentence_count}, "
            f"Word {self.current_word_in_sentence}/{self.min_words_per_sentence}+"
        )
        
        # Generate the next word
        await self.next_word()
        
        # If we've reached the minimum words per sentence, allow end punctuation
        if self.current_word_in_sentence >= self.min_words_per_sentence:
            # Sample whether to end the sentence
            if await self.sample(PunctuationMask(self)):
                # End the sentence
                await self.next_token()
                
                # Reset the word counter and increment the sentence counter
                self.current_word_in_sentence = 0
                self.current_sentence += 1
                
                # If we've reached the sentence count, end generation
                if self.current_sentence >= self.sentence_count:
                    await self.end()
    
    async def check(self, text: str) -> bool:
        """
        Check that the generated text satisfies the sentence constraints.
        
        Args:
            text: The generated text
            
        Returns:
            Boolean indicating whether the text satisfies the constraints
        """
        # Split the text into sentences (simple approximation)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Check the sentence count
        if len(sentences) != self.sentence_count:
            return False
        
        # Check each sentence's word count
        for sentence in sentences:
            words = sentence.split()
            if len(words) < self.min_words_per_sentence:
                return False
        
        return True


class SquareWordPoemModel(BaseModel):
    """
    Generates a poem with N lines, where each line has exactly N words.
    """
    
    def __init__(
        self,
        context,
        max_tokens: int = 512,
        size: int = 5
    ):
        """
        Initialize the model.
        
        Args:
            context: The context for the model
            max_tokens: Maximum number of tokens to generate
            size: Number of lines and words per line
        """
        super().__init__(context=context, max_tokens=max_tokens)
        self.size = size
        self.current_line = 0
        self.current_word_in_line = 0
    
    @classmethod
    def prior_prompt(cls):
        return "Write a poem that is creative and evocative."
    
    async def step(self):
        """
        Generation strategy:
        - Generate lines word by word
        - Ensure each line has exactly size words
        - After each line, generate a newline
        - Continue until size lines are reached
        """
        # Increment the word counter
        self.current_word_in_line += 1
        
        # Provide a hint about the current line and word
        await self.hint(
            f"Line {self.current_line + 1}/{self.size}, "
            f"Word {self.current_word_in_line}/{self.size}"
        )
        
        # Generate the next word
        await self.next_word()
        
        # If we've reached the end of the line, generate a newline
        if self.current_word_in_line >= self.size:
            async with NewLineMask(self, n=1):
                await self.next_token()
            
            # Reset the word counter and increment the line counter
            self.current_word_in_line = 0
            self.current_line += 1
            
            # If we've reached the line count, end generation
            if self.current_line >= self.size:
                await self.end()
    
    async def check(self, text: str) -> bool:
        """
        Check that the generated text satisfies the poem constraints.
        
        Args:
            text: The generated text
            
        Returns:
            Boolean indicating whether the text satisfies the constraints
        """
        # Split the text into lines
        lines = text.strip().split('\n')
        
        # Check the line count
        if len(lines) != self.size:
            return False
        
        # Check each line's word count
        for line in lines:
            words = line.split()
            if len(words) != self.size:
                return False
        
        return True


class BudgetConstrainedListModel(BaseModel):
    """
    Generates an ingredients list with a budget constraint.
    """
    
    def __init__(
        self,
        context,
        max_tokens: int = 512,
        budget: float = 20.0,
        max_items: int = 7
    ):
        """
        Initialize the model.
        
        Args:
            context: The context for the model
            max_tokens: Maximum number of tokens to generate
            budget: Maximum total cost of ingredients
            max_items: Maximum number of ingredients
        """
        super().__init__(context=context, max_tokens=max_tokens)
        self.budget = budget
        self.max_items = max_items
        self.current_item = 0
        self.total_cost = 0.0
        self.header_generated = False
    
    @classmethod
    def prior_prompt(cls):
        return "Write an ingredients list for a recipe."
    
    def extract_cost(self, text: str) -> float:
        """
        Extract the cost from an ingredient line.
        
        Args:
            text: The ingredient line
            
        Returns:
            The cost as a float, or None if no cost is found
        """
        match = re.search(r'\$(\d+(?:\.\d+)?)', text)
        if not match:
            return None
        
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    async def step(self):
        """
        Generation strategy:
        - First generate the header "Ingredients:"
        - Then generate each ingredient line by line
        - Each line should start with "- " and include a price
        - Keep track of the total cost and ensure it stays under budget
        - Continue until max_items is reached or the model decides to end
        """
        # Generate the header first
        if not self.header_generated:
            await self.extend_with("Ingredients:\n")
            self.header_generated = True
            return
        
        # Provide a hint about the remaining budget
        await self.hint(f"Remaining budget: ${self.budget - self.total_cost:.2f}")
        
        # Generate the next ingredient
        ingredient, eos = await self.extend(start="- ", stop=["\n"], allow_eos=True)
        
        # Extract the cost
        cost = self.extract_cost(ingredient)
        if cost is None:
            self.condition(False)
            return
        
        # Update the total cost
        self.total_cost += cost
        
        # Check if we've exceeded the budget
        if self.total_cost > self.budget:
            self.condition(False)
            return
        
        # Increment the item counter
        self.current_item += 1
        
        # If we've reached the max items or the model decided to end, end generation
        if self.current_item >= self.max_items or eos:
            await self.end()
    
    async def check(self, text: str) -> bool:
        """
        Check that the generated text satisfies the budget constraints.
        
        Args:
            text: The generated text
            
        Returns:
            Boolean indicating whether the text satisfies the constraints
        """
        # Check for the header
        if not text.startswith("Ingredients:"):
            return False
        
        # Split the text into lines
        lines = text.strip().split('\n')
        
        # Remove the header
        lines = lines[1:]
        
        # Check the item count
        if len(lines) > self.max_items:
            return False
        
        # Calculate the total cost
        total_cost = 0.0
        for line in lines:
            if not line.strip():
                continue
            
            # Check the line format
            if not line.startswith("- "):
                return False
            
            # Extract the cost
            cost = self.extract_cost(line)
            if cost is None:
                return False
            
            total_cost += cost
        
        # Check the budget constraint
        if total_cost > self.budget:
            return False
        
        return True
