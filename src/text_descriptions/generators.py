"""LLM-based text description generators.

This module provides functionality to automatically generate text descriptions
for dataset classes using large language models like GPT-4o and Claude.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import time


class TextDescriptionGenerator(ABC):
    """Base class for LLM-based description generation."""

    @abstractmethod
    def generate_class_descriptions(
        self,
        class_name: str,
        dataset_context: str,
        num_descriptions: int = 10,
        diversity: str = "medium"
    ) -> List[str]:
        """Generate diverse descriptions for a single class.

        Args:
            class_name: Name of the class (e.g., "airplane", "cat")
            dataset_context: Context about the dataset (e.g., "CIFAR10 natural images")
            num_descriptions: Number of descriptions to generate
            diversity: Level of variation ("low", "medium", "high")

        Returns:
            List of text descriptions
        """
        raise NotImplementedError

    def generate_dataset_descriptions(
        self,
        class_names: List[str],
        dataset_name: str,
        num_descriptions: int = 10,
        diversity: str = "medium"
    ) -> Dict[str, List[str]]:
        """Generate descriptions for all classes in a dataset.

        Args:
            class_names: List of class names
            dataset_name: Name of the dataset
            num_descriptions: Number of descriptions per class
            diversity: Level of variation

        Returns:
            Dictionary mapping class names to descriptions
        """
        descriptions = {}
        for class_name in class_names:
            print(f"Generating descriptions for '{class_name}'...")
            descriptions[class_name] = self.generate_class_descriptions(
                class_name=class_name,
                dataset_context=f"{dataset_name} dataset",
                num_descriptions=num_descriptions,
                diversity=diversity
            )
            time.sleep(0.5)  # Rate limiting

        return descriptions


class OpenAIDescriptionGenerator(TextDescriptionGenerator):
    """Use OpenAI API (GPT-4o) for description generation.

    GPT-4o pricing (2026):
    - Input: $2.50 per million tokens
    - Output: $10.00 per million tokens
    - ~$1 per dataset for description generation
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize OpenAI generator.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Model to use (default: "gpt-4o")
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

    def generate_class_descriptions(
        self,
        class_name: str,
        dataset_context: str,
        num_descriptions: int = 10,
        diversity: str = "medium"
    ) -> List[str]:
        """Generate descriptions using GPT-4o.

        Args:
            class_name: Name of the class
            dataset_context: Context about the dataset
            num_descriptions: Number of descriptions to generate
            diversity: Level of variation ("low", "medium", "high")

        Returns:
            List of text descriptions
        """
        # Set temperature based on diversity
        temperature_map = {"low": 0.3, "medium": 0.7, "high": 1.0}
        temperature = temperature_map.get(diversity, 0.7)

        # Construct prompt
        prompt = f"""Generate {num_descriptions} diverse text descriptions for the class "{class_name}" from the {dataset_context}.

Requirements:
1. Each description should be suitable for text-to-image generation
2. Descriptions should be specific and visual
3. Use different phrasings and perspectives
4. Keep descriptions concise (5-15 words each)
5. Return ONLY the descriptions, one per line, without numbering

Example format:
a photo of a {class_name}
a high-quality image of a {class_name}
...

Now generate {num_descriptions} descriptions for "{class_name}"."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates diverse text descriptions for image datasets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )

            # Parse response
            text = response.choices[0].message.content.strip()
            descriptions = [line.strip() for line in text.split('\n') if line.strip()]

            # Filter out any numbered lines or empty lines
            descriptions = [
                d for d in descriptions
                if d and not d[0].isdigit() and d[:2] not in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']
            ]

            # Ensure we have the requested number
            if len(descriptions) < num_descriptions:
                print(f"Warning: Generated {len(descriptions)} descriptions, expected {num_descriptions}")

            return descriptions[:num_descriptions]

        except Exception as e:
            print(f"Error generating descriptions for '{class_name}': {e}")
            # Fallback to simple templates
            return self._fallback_descriptions(class_name, num_descriptions)

    def _fallback_descriptions(self, class_name: str, num_descriptions: int) -> List[str]:
        """Generate fallback descriptions using templates.

        Args:
            class_name: Name of the class
            num_descriptions: Number of descriptions

        Returns:
            List of template-based descriptions
        """
        templates = [
            f"a photo of a {class_name}",
            f"a high-quality image of a {class_name}",
            f"a rendering of a {class_name}",
            f"a picture of a {class_name}",
            f"an image of a {class_name}",
            f"a clear photo of a {class_name}",
            f"a good photo of a {class_name}",
            f"a close-up photo of a {class_name}",
            f"a bright photo of a {class_name}",
            f"a cropped photo of a {class_name}",
        ]
        return templates[:num_descriptions]


class ClaudeDescriptionGenerator(TextDescriptionGenerator):
    """Use Anthropic Claude API for description generation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Claude generator.

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            model: Model to use (default: "claude-3-5-sonnet-20241022")
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

    def generate_class_descriptions(
        self,
        class_name: str,
        dataset_context: str,
        num_descriptions: int = 10,
        diversity: str = "medium"
    ) -> List[str]:
        """Generate descriptions using Claude.

        Args:
            class_name: Name of the class
            dataset_context: Context about the dataset
            num_descriptions: Number of descriptions
            diversity: Level of variation

        Returns:
            List of text descriptions
        """
        # Set temperature based on diversity
        temperature_map = {"low": 0.3, "medium": 0.7, "high": 1.0}
        temperature = temperature_map.get(diversity, 0.7)

        prompt = f"""Generate {num_descriptions} diverse text descriptions for the class "{class_name}" from the {dataset_context}.

Requirements:
1. Each description should be suitable for text-to-image generation
2. Descriptions should be specific and visual
3. Use different phrasings and perspectives
4. Keep descriptions concise (5-15 words each)
5. Return ONLY the descriptions, one per line, without numbering

Example format:
a photo of a {class_name}
a high-quality image of a {class_name}
...

Now generate {num_descriptions} descriptions for "{class_name}"."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response
            text = response.content[0].text.strip()
            descriptions = [line.strip() for line in text.split('\n') if line.strip()]

            # Filter out any numbered lines
            descriptions = [
                d for d in descriptions
                if d and not d[0].isdigit() and d[:2] not in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']
            ]

            return descriptions[:num_descriptions]

        except Exception as e:
            print(f"Error generating descriptions for '{class_name}': {e}")
            return self._fallback_descriptions(class_name, num_descriptions)

    def _fallback_descriptions(self, class_name: str, num_descriptions: int) -> List[str]:
        """Generate fallback descriptions using templates."""
        templates = [
            f"a photo of a {class_name}",
            f"a high-quality image of a {class_name}",
            f"a rendering of a {class_name}",
            f"a picture of a {class_name}",
            f"an image of a {class_name}",
            f"a clear photo of a {class_name}",
            f"a good photo of a {class_name}",
            f"a close-up photo of a {class_name}",
            f"a bright photo of a {class_name}",
            f"a cropped photo of a {class_name}",
        ]
        return templates[:num_descriptions]


# CLI interface for generating descriptions
if __name__ == "__main__":
    import argparse
    from src.text_descriptions.loaders import TextDescriptionLoader

    parser = argparse.ArgumentParser(description="Generate text descriptions for a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., CIFAR10)")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="List of class names")
    parser.add_argument("--llm", type=str, default="gpt4o", choices=["gpt4o", "claude"], help="LLM to use")
    parser.add_argument("--num-descriptions", type=int, default=15, help="Number of descriptions per class")
    parser.add_argument("--diversity", type=str, default="medium", choices=["low", "medium", "high"], help="Diversity level")
    parser.add_argument("--output-dir", type=str, default="data/text_descriptions", help="Output directory")

    args = parser.parse_args()

    # Initialize generator
    if args.llm == "gpt4o":
        generator = OpenAIDescriptionGenerator(model="gpt-4o")
        variant = "gpt4o"
    elif args.llm == "claude":
        generator = ClaudeDescriptionGenerator()
        variant = "claude"
    else:
        raise ValueError(f"Unknown LLM: {args.llm}")

    # Generate descriptions
    print(f"Generating descriptions for {args.dataset} using {args.llm}...")
    descriptions = generator.generate_dataset_descriptions(
        class_names=args.classes,
        dataset_name=args.dataset,
        num_descriptions=args.num_descriptions,
        diversity=args.diversity
    )

    # Save to file
    loader = TextDescriptionLoader(base_path=args.output_dir)
    metadata = {
        "llm": args.llm,
        "num_descriptions_per_class": args.num_descriptions,
        "diversity": args.diversity,
        "num_classes": len(args.classes)
    }
    loader.save_descriptions(
        descriptions=descriptions,
        dataset_name=args.dataset,
        source="generated",
        variant=variant,
        metadata=metadata
    )

    print(f"Descriptions saved to {args.output_dir}/generated/{args.dataset}_{variant}.json")
    print(f"Total classes: {len(descriptions)}")
    print(f"Total descriptions: {sum(len(v) for v in descriptions.values())}")
