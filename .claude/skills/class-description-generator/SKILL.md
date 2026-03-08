---
name: class-description-generator
description: >
  Generate diverse natural-language descriptions for dataset classes (e.g., ImageNet, SUN397, custom datasets).
  Produces 10 short descriptions (5-10 words each) per class in JSON format, explicitly avoiding CLIP-style
  templates like "a photo of a {class}". Supports single-class and batch mode with parallelization for large
  datasets. Use this skill whenever the user wants to generate text descriptions, captions, or labels for
  image classification classes, dataset categories, or visual concepts — even if they don't say "descriptions"
  explicitly. Trigger on phrases like "describe these classes", "generate captions for my dataset",
  "I need text labels for ImageNet/SUN397/CIFAR", "create descriptions for each category",
  "annotate my class list", or any request involving natural language descriptions tied to dataset class names.
---

# Class Description Generator

Generate diverse, natural-language descriptions for dataset classes. Each class gets ~10 descriptions that are 5-10 words long, output as JSON.

## Why this exists

Zero-shot and few-shot vision-language models often need text descriptions of classes. The standard approach uses rigid CLIP-style templates ("a photo of a {class}", "a blurry photo of a {class}") which are repetitive, lack semantic diversity, and don't capture the rich variety of how a class actually appears in the real world. This skill generates descriptions that are natural, diverse, and semantically rich — covering visual attributes, contextual settings, behaviours, and distinctive features.

## What makes a good description

Each description should be 5-10 words and read like a natural phrase a person might use to describe encountering that thing. The set of ~10 descriptions per class should collectively cover different facets:

- **Visual attributes**: colour, shape, texture, size, distinctive markings
- **Context/scene**: where you'd typically find or see it
- **Action/behaviour**: what it does, how it moves or functions (if applicable)
- **Distinctive features**: what sets it apart from similar things

Variety is the goal. If all 10 descriptions mention the same attribute, the set is poor. A good set paints a multi-dimensional picture of the class.

### What to avoid

- CLIP-style templates: "a photo of a {class}", "a picture of a {class}", "an image of a {class}", "a blurry/dark/bright photo of..."
- Starting every description with the class name
- Generic filler: "a nice {class}", "a good example of {class}"
- Descriptions that are essentially synonyms of each other
- Going below 5 words or above 10 words

### Examples

**Class: "golden retriever"**
```json
{
  "golden retriever": [
    "fluffy golden coat with a wagging tail",
    "loyal dog fetching a tennis ball",
    "large friendly breed with floppy ears",
    "wet retriever shaking off after swimming",
    "golden fur gleaming in afternoon sunlight",
    "gentle family dog sitting on grass",
    "muscular build with a broad happy face",
    "playful retriever bounding through a park",
    "thick double coat in honey gold shade",
    "obedient dog waiting by its owner's side"
  ]
}
```

**Class: "cliff"**
```json
{
  "cliff": [
    "steep rock face dropping to the sea",
    "jagged limestone edge above crashing waves",
    "towering sandstone wall with layered striations",
    "sheer vertical drop with sparse vegetation",
    "eroded coastal bluff under overcast skies",
    "narrow ledge along a dramatic precipice",
    "weathered granite escarpment above a valley",
    "chalky white cliffs overlooking the ocean",
    "rugged canyon wall in arid terrain",
    "mossy overhang on a forested mountain edge"
  ]
}
```

Notice how each description captures a different aspect — material, setting, weather, scale, vegetation, geometry — rather than just repeating "a tall cliff" in different words.

## Single-class mode

When the user provides one class (or a small handful), generate descriptions directly in conversation and output the JSON.

**Workflow:**
1. User provides the class name(s)
2. Generate ~10 descriptions per class following the guidelines above
3. Output as JSON: `{"class_name": ["desc1", ..., "desc10"]}`
4. If the user provides additional context (e.g., "these are fine-grained bird species" or "SUN397 scene categories"), use that context to make descriptions more specific and accurate

## Batch mode

For large datasets (tens to thousands of classes), use the bundled batch processing script to parallelise generation.

**Workflow:**
1. User provides a class list — either as a file (one class per line, `.txt` or `.csv`) or names a well-known dataset
2. For well-known datasets, generate the class list from common knowledge:
   - **ImageNet-1K**: 1,000 classes (use the standard ILSVRC 2012 synset names)
   - **SUN397**: 397 scene categories
   - **CIFAR-10 / CIFAR-100**: 10 / 100 classes
   - **Others**: Ask the user for the class list file
3. Run the batch script: `python scripts/batch_generate.py`

### Batch script usage

The batch script at `scripts/batch_generate.py` handles parallelisation and merging. Read the script before running it to understand the flags:

```bash
python scripts/batch_generate.py \
  --class-file classes.txt \
  --output descriptions.json \
  --descriptions-per-class 10 \
  --max-parallel 10
```

**Key flags:**
- `--class-file`: Path to a text file with one class name per line
- `--output`: Path for the merged JSON output (default: `descriptions.json`)
- `--descriptions-per-class`: Number of descriptions per class (default: 10)
- `--max-parallel`: Maximum concurrent subagent tasks (default: 10). Tune based on your rate limits and system resources. Higher values finish faster but may hit API limits.
- `--context`: Optional string providing domain context, e.g., `"fine-grained bird species"` or `"indoor scene categories"`. This gets passed to each subagent to improve description quality.

The script works by:
1. Reading the class list
2. Splitting classes into batches
3. Spawning parallel Claude Code subagents, each generating descriptions for a batch of classes
4. Merging all results into a single JSON file
5. Validating that every class has the right number of descriptions and they meet length requirements

### Output format

The final merged JSON is a flat dictionary:

```json
{
  "golden retriever": ["desc1", "desc2", ...],
  "cliff": ["desc1", "desc2", ...],
  "bedroom": ["desc1", "desc2", ...]
}
```

This is easy to load in Python (`json.load()`), merge with other files, or feed into training pipelines.

## Tips for best results

- **Provide domain context** when available. "SUN397 scene categories" produces better results than just a list of words, because it helps disambiguate (e.g., "bank" as a river bank vs. a financial institution).
- **Fine-grained datasets** benefit from more specific guidance. For CUB-200 bird species, mention that descriptions should include plumage, beak shape, habitat, and behaviour.
- **Review a sample** before running the full batch. Generate descriptions for 5-10 classes first, check quality, then scale up.
- **Adjust count** if needed. 10 is a good default, but some use cases benefit from more (20-30) or fewer (5).