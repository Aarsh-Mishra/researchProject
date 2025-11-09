# sas_test.py
import os
import asyncio
import json
import time
from json import JSONDecoder
from autogen_agentchat.agents import (AssistantAgent)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image as AGImage
from PIL import Image

# --- 1. DEFINE THE SINGLE AGENT (SAS) ---

# This prompt is from the projectSuggestion.txt
SAS_SYSTEM_PROMPT = """
You are an expert visual analyst. Your task is to analyze a potentially blurred or ambiguous image and provide a detailed, structured analysis in a single JSON object.
Follow these steps in your internal reasoning process:
    1.  **Initial Observation:** Begin by describing what you see in the image. Note fundamental visual elements like shapes, colors, light, shadow, and any recognizable patterns or objects, even if your identification is uncertain.
    2.  **Hypothesis Generation:** Based on your initial observations, formulate at least two distinct and plausible interpretations of the image's content. For each interpretation, describe the scene it implies.
    3.  **Critical Evaluation:** For each interpretation you generated, critically assess its likelihood. Identify specific visual evidence within the image that supports the interpretation. Also, identify evidence that contradicts it or sources of ambiguity that weaken its certainty.
    4.  **Confidence Assessment:** Assign a confidence score from 1 (a wild guess with very little evidence) to 10 (absolute certainty with clear, unambiguous evidence) to each interpretation. You MUST be cautious and critical in your scoring. Default to a lower score if there is any significant ambiguity. You must provide a brief justification for each score, linking it to your critical evaluation.
    5.  **Synthesis:** After evaluating all interpretations, provide a final, synthesized conclusion that represents the most probable content of the image.

Your final output MUST be ONLY a valid JSON object that strictly adheres to the schema provided below.
Do not include any introductory text, concluding remarks, markdown formatting such as `json`, or any other explanations outside of the JSON structure itself.
{
  "possible_interpretations": [
    {
      "description": "A detailed textual description of one possible interpretation of the image.",
      "confidence_score": 10, # 0-10 scale
      "reasoning_for_score": "A brief justification for the assigned confidence score, mentioning specific visual evidence or lack thereof."
    }
  ],
  "final_conclusion": "A single, synthesized conclusion about the most likely content of the image.",
  "word_count": number_of_words_in_final_conclusion
}
"""

def create_sas_agent(model_client):
    """Creates the Single-Agent System (SAS) agent."""
    singleAgent = AssistantAgent(
        name='sas_analyst',
        model_client=model_client,
        system_message=SAS_SYSTEM_PROMPT
    )
    return singleAgent

# --- 2. DEFINE THE SAS TEST RUNNER ---

async def run_sas_test(image_path: str, model_client):
    """
    Runs a single test for the SAS, captures the time,
    and parses the final JSON output.
    """
    print(f"  Running SAS test for: {image_path}")
    
    sas_agent = create_sas_agent(model_client)
    
    # 1. Load the image
    pil_image = Image.open(image_path)
    ag_image = AGImage(pil_image)

    multi_modal_msg = MultiModalMessage(
        content=['You explain what an image may contain, even if blurred and strictly use the system message for guidance', ag_image],
        source='user'
    )
    
    # 2. Run the test and record time
    start_time = time.time()
    
    # This is a simple, single call. No RoundRobinChat.
    reply = await sas_agent.run(task=multi_modal_msg)
    
    end_time = time.time()
    processing_time = end_time - start_time

    print("\n--- SAS 'Conversation' (Input & Output) ---")
    for msg in reply.messages:
        print(f"[{msg.source}]: {msg.content}\n")
    print("--- End SAS 'Conversation' ---\n")

    # 3. Parse the final JSON output (using our robust parser)
    try:
        raw_output = reply.messages[-1].content
        
        json_start_pos = raw_output.find('{')
        if json_start_pos == -1:
            print("  ERROR: '{' not found in last message.")
            return None

        json_string = raw_output[json_start_pos:]
        
        decoder = JSONDecoder()
        data, index = decoder.raw_decode(json_string) 
        
        data['processing_time'] = processing_time
        
        print(f"  Success (SAS). Time: {processing_time:.2f}s")
        return data

    except Exception as e:
        print(f"  ERROR parsing JSON (SAS): {e}")
        print(f"  Raw output was: {raw_output}")
        return None