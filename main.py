import os
import asyncio
import json
import time
from json import JSONDecoder
import csv
from dotenv import load_dotenv
from autogen_agentchat.agents import (AssistantAgent)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination,TextMentionTermination

from autogen_core import Image as AGImage
from PIL import Image

from sas_test import run_sas_test

# --- 1. SETUP (Same as your file) ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=api_key,
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family="gemini"
    )
)

# --- 2. AGENT DEFINITIONS (With the new prompts) ---

imageExplainerAgent = AssistantAgent(
        name = 'explainer',
        model_client= model_client,
        system_message=
        """
        You are an expert visual Explainer agent.
        You will receive a message containing a blurred image.
        Your job is to analyze the image and propose one or more plausible interpretations.

        - Describe what you see, even if you are uncertain.
        - Your goal is to have a collaborative, round-robin discussion with the 'evaluator' agent.
        - Engage with the 'evaluator's' critique. Re-examine the image based on their feedback and refine your ideas.
        
        **IMPORTANT:**
        - When you and the evaluator have reached a final conclusion and you are satisfied, your **VERY LAST** message must be **ONLY** the following exact phrase:
        
        `I am satisfied with our conclusion. Please compile the final report.`
        
        - Do not say *anything* else in that final message.
        """
        )

imageEvaluatorAgent = AssistantAgent(
        name = 'evaluator',
        model_client= model_client,
        system_message=
        """
        You are an expert Evaluator agent. You have two critical roles.

        **ROLE 1: COLLABORATE & CRITIQUE**
        - You will receive the Explainer's analysis of a blurred image.
        - Your job is to critique and refine this explanation through a round-robin discussion.
        - Ask for clarification, check for consistency, and suggest alternative interpretations.
        - Work collaboratively with the Explainer until you are both satisfied.
        - **DO NOT** compile the final report until the Explainer tells you to.

        **ROLE 2: COMPILE THE FINAL REPORT (WHEN TOLD)**
        - You must continue the discussion until the `explainer` agent's message is **EXACTLY**:
        `I am satisfied with our conclusion. Please compile the final report.`
        
        - **When, and only when, you see that exact phrase**, it is your turn to reply.
        - Your reply MUST be your final message and MUST start with the word **DONE** followed immediately by the single, valid JSON object.
        - To build this JSON, you must **review the entire conversation history** and summarize it.

        **JSON SCHEMA:**
        {
          "possible_interpretations": [
            {
              "description": "A detailed description of one key hypothesis discussed.",
              "confidence_score": scale_of_0_to_10,
              "reasoning_for_score": "Justification for this score, referencing visual evidence and the discussion."
            }
          ],
          "final_conclusion": "The single, synthesized conclusion that you and the explainer agreed on.",
          "word_count": number_of_words_in_final_conclusion
        }

        **INSTRUCTIONS FOR JSON:**
        - The `possible_interpretations` array MUST summarize the main ideas from your discussion. 
        - **CRITICAL:** This array **MUST** include both the initial (weaker/rejected) hypotheses and the final (stronger/accepted) hypothesis.
        - The `reasoning_for_score` for a rejected idea should explain WHY it was rejected.
        - `word_count` is the word count of the `final_conclusion` string.
        
        """
    )


# --- 3. NEW TEST FUNCTION (This is what you asked for) ---
async def run_mas_test(image_path: str):
    """
    Runs a single test for the MAS, captures the time,
    and parses the final JSON output.
    """
    print(f"  Running MAS test for: {image_path}")
    
    # 1. Load the image
    pil_image = Image.open(image_path)
    ag_image = AGImage(pil_image)

    multi_modal_msg = MultiModalMessage(
        content = ['You explain what an image may contain, even if blurred and strictly use the system message for guidance', ag_image],
        source='user'
    )
    
    # 2. Set up the team (same as your original)
    termination_condition = MaxMessageTermination(20) | TextMentionTermination('DONE')
    team = RoundRobinGroupChat(
        participants= [imageExplainerAgent, imageEvaluatorAgent],
        termination_condition=termination_condition,
        max_turns=20
    )

    # 3. Run the test and record time
    start_time = time.time()
    
    result = await team.run(task=multi_modal_msg)
    
    end_time = time.time()
    processing_time = end_time - start_time


   # --- ADD THESE 3 LINES TO PRINT THE CONVERSATION ---
    print("\n--- Full Conversation ---")
    for msg in result.messages:
        print(f"[{msg.source}]: {msg.content}\n")
    print("--- End Conversation ---\n")
    # --- END OF ADDED LINES ---

    # 4. Parse the final JSON output (This is the FINAL, 100% ROBUST logic)
    try:
        # --- NEW ROBUST FINDER ---
        # Loop backwards from the end of the chat to find the JSON
        last_message_content = ""
        for msg in reversed(result.messages):
            # Find the first message from 'evaluator' that has a '{'
            if msg.source == 'evaluator' and msg.content and "{" in msg.content:
                last_message_content = msg.content
                break  # Found it!
        
        if not last_message_content:
            # If we looped and found nothing
            print("  ERROR: Could not find a JSON message from 'evaluator' in the chat history.")
            return None
        # --- END OF ROBUST FINDER ---

        # Find the start of the JSON object
        json_start_pos = last_message_content.find('{')
        
        # Extract the JSON string (starting from the first '{')
        json_string = last_message_content[json_start_pos:]
        
        # Use JSONDecoder.raw_decode()
        decoder = JSONDecoder()
        data, index = decoder.raw_decode(json_string) 
        
        # Add the processing time to our data
        data['processing_time'] = processing_time
        
        print(f"  Success (MAS). Time: {processing_time:.2f}s")
        return data

    except Exception as e:
        print(f"  ERROR parsing JSON (MAS): {e}")
        print(f"  Raw output was: {last_message_content}")
        return None

# --- 4. NEW MAIN FUNCTION (The "Test Harness") ---
async def main():
    
    IMAGE_FILES_TO_TEST = [
        "blurred_dataset\\secondTestImage_blur_0.jpg",
        "blurred_dataset\\secondTestImage_blur_2.jpg",
        "blurred_dataset\\secondTestImage_blur_5.jpg",
        "blurred_dataset\\secondTestImage_blur_10.jpg",
        "blurred_dataset\\secondTestImage_blur_15.jpg",
        "blurred_dataset\\secondTestImage_blur_20.jpg"
    ]
    NUM_RUNS = 2 # Set to 3-5 for real experiment
    
    # One CSV file for ALL results
    CSV_OUTPUT_FILE = "comparative_results.csv"

    print(f"Starting experiment... Appending results to {CSV_OUTPUT_FILE}")
    file_exists = os.path.exists(CSV_OUTPUT_FILE)

    with open(CSV_OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "system_type", # NEW: "SAS" or "MAS"
                "image_file", 
                "blur_radius", 
                "run_num", 
                "final_conclusion", 
                "final_confidence", 
                "word_count", 
                "processing_time"
            ])
            print(f"Created new file and wrote header: {CSV_OUTPUT_FILE}")
        else:
            print(f"File exists. Appending new results...")
            
        for img_file in IMAGE_FILES_TO_TEST:
            if not os.path.exists(img_file):
                print(f"WARNING: Image not found '{img_file}'. Skipping.")
                continue
            
            base_name = os.path.basename(img_file)
            file_name_no_ext = os.path.splitext(base_name)[0]
            blur_radius = file_name_no_ext.split('_')[-1]
            
            print(f"--- Processing Image: {img_file} (Blur: {blur_radius}) ---")

            for i in range(1, NUM_RUNS + 1):
                print(f"\n  Starting Run {i}/{NUM_RUNS}...")
                
                # --- RUN MAS TEST ---
                print("  (Running MAS...)")
                mas_data = await run_mas_test(img_file)
                if mas_data:
                    try:
                        final_conf = mas_data['possible_interpretations'][-1]['confidence_score']
                        writer.writerow([
                            "MAS", img_file, blur_radius, i,
                            mas_data['final_conclusion'], final_conf,
                            mas_data['word_count'], mas_data['processing_time']
                        ])
                    except Exception as e:
                        print(f"  ERROR: Could not save MAS data. {e}")


                # --- RUN SAS TEST ---
                print("  (Running SAS...)")
                sas_data = await run_sas_test(img_file, model_client)
                if sas_data:
                        try:
                            final_conf = sas_data['possible_interpretations'][-1]['confidence_score']
                                # --- THIS IS THE FIX ---
                            word_count = sas_data.get('word_count', 0) 
                                # --- END OF FIX ---
                            writer.writerow([
                                "SAS", img_file, blur_radius, i,
                                sas_data['final_conclusion'], final_conf,
                                word_count, sas_data['processing_time']
                            ])
                        except Exception as e:
                            print(f"  ERROR: Could not save SAS data. {e}")

    print("--- Experiment Complete ---")
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

