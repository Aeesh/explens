"""
All prompt templates for ExpLens.
Separated from generation logic for clarity and modularity.
"""

NARRATOR_SYSTEM = """You are an expert ML engineer writing a concise, accurate
experiment report. You receive structured facts about a training run.
Your job is to convert them into clear, readable prose that a researcher
or engineer can immediately act on.

RULES:
1. Only state things that are supported by the facts provided.
2. Use precise numbers from the facts — do not round or approximate.
3. Do not invent explanations that aren't supported by the data.
4. If something is ambiguous, say so rather than guessing.
5. Write in past tense. Be direct. No filler phrases like "it is worth noting".
6. Length: 3-5 sentences per section. Concise is better than comprehensive.
"""

OVERVIEW_PROMPT = """
{facts_context}

Write the Overview section of the experiment report.
Cover: what was trained, for how long, and the headline result.
Do not mention recommendations yet.
"""

TRAINING_DYNAMICS_PROMPT = """
{facts_context}

Write the Training Dynamics section.
Cover: how loss evolved, whether training was stable, whether the model
converged and when. Mention specific epoch numbers and loss values from the facts.
"""

GENERALISATION_PROMPT = """
{facts_context}

Write the Generalisation section.
Cover: how validation metrics compare to training metrics, whether
overfitting occurred, and how severe it was.
Only include this section if validation metrics are available.
"""

RECOMMENDATIONS_PROMPT = """
{facts_context}

DETECTED PATTERNS:
{patterns}

Write the Recommendations section.
Give 2-4 specific, actionable recommendations based on the patterns above.
Each recommendation should reference specific hyperparameter values or
techniques. Prioritise critical issues first.
"""

WHAT_NEXT_PROMPT = """
{facts_context}

DETECTED PATTERNS:
{patterns}

Given these results, what are the 3 most important experiments to run next?
Be specific — mention hyperparameter ranges, techniques, or ablations.
Format as a numbered list.
"""