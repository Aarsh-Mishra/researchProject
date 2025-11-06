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
        - Be explicit about your level of confidence (e..g, "likely", "possibly", "a wild guess").
        - Include relevant details such as shapes, colors, context, or patterns.
        - Your goal is to have a collaborative, round-robin discussion with the 'evaluator' agent.
        - Engage with the 'evaluator's' critique. Re-examine the image based on their feedback and refine your ideas.
        - Do not try to finalize the conclusion. Just work with the 'evaluator' to find the best possible explanation.
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
        - Work collaboratively with the Explainer until you are both satisfied and have reached a consensus on the most plausible interpretation.

        **ROLE 2: COMPILE THE FINAL REPORT**
        - Once the discussion is complete, your **VERY LAST MESSAGE** must be the final report.
        - This message MUST start valid JSON object followed by the word "DONE".
        - To build this JSON, you must **review the entire conversation history** and summarize it.

        **JSON SCHEMA:**
        Your final JSON output MUST strictly follow this schema:
        {
          "possible_interpretations": [
            {
              "description": "A detailed description of one key hypothesis discussed.",
              "confidence_score": 10,  # 0-10 scale
              "reasoning_for_score": "Justification for this score, referencing visual evidence and the discussion."
            }
          ],
          "final_conclusion": "The single, synthesized conclusion that you and the explainer agreed on.",
          "word_count": 0
        }

        **INSTRUCTIONS FOR JSON:**
        - The `possible_interpretations` array MUST summarize the main ideas from your discussion (e.g., the initial "wrong" idea and the final "correct" one).
        - The `reasoning_for_score` should explain WHY an idea was kept or rejected.
        - `word_count` is the word count of the `final_conclusion` string.
        - **DO NOT** include any other text, apologies, or explanations like "Here is the JSON:" in your final message. Your *entire* final message must start with `{...}` JSON object followed by the word "DONE".
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
    termination_condition = MaxMessageTermination(10) | TextMentionTermination('DONE')
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

    # 4. Parse the final JSON output (This is the NEW, ROBUST logic)
    try:
        # Get the content of the very last message
        last_message_content = result.messages[-1].content
        
        # Find the start of the JSON object
        json_start_pos = last_message_content.find('{')
        
        if json_start_pos == -1 or "DONE" not in last_message_content:
            print("  ERROR: 'DONE' or '{' not found in last message.")
            return None

        # Extract the JSON string (starting from the first '{')
        json_string = last_message_content[json_start_pos:]
        
        # --- THIS IS THE FIX ---
        # Use JSONDecoder.raw_decode()
        # This reads ONE valid JSON object and stops, ignoring any extra "garbage" data
        decoder = JSONDecoder()
        data, index = decoder.raw_decode(json_string) 
        # 'data' is now our clean Python dictionary
        
        # Add the processing time to our data
        data['processing_time'] = processing_time
        
        print(f"  Success (and ignored garbage data). Time: {processing_time:.2f}s")
        return data

    except Exception as e:
        print(f"  ERROR parsing JSON: {e}")
        print(f"  Raw output was: {last_message_content}")
        return None

# --- 4. NEW MAIN FUNCTION (The "Test Harness") ---
async def main():
    
    # !!! --- IMPORTANT --- !!!
    # Update this list with your image paths
    IMAGE_FILES_TO_TEST = [
        "C:\\Users\\hp\\Downloads\\test_image2.jpg" # Use your full list here
        # "test_image_blur_0.jpg",
        # "test_image_blur_2.jpg",
        # "test_image_blur_5.jpg",
        # "test_image_blur_10.jpg",
        # "test_image_blur_15.jpg",
        # "test_image_blur_20.jpg",
    ]
    
    # Run each test 3 times as discussed
    NUM_RUNS = 1 # Set this back to 3 or 5 for the real experiment
    
    # This is the output file for your data
    CSV_OUTPUT_FILE = "mas_results.csv"

    print(f"Starting MAS experiment... Appending results to {CSV_OUTPUT_FILE}")

    # --- THIS IS THE NEW LOGIC ---
    # 1. Check if the file already exists to decide on writing the header
    file_exists = os.path.exists(CSV_OUTPUT_FILE)

    # 2. Open the file in 'append' mode ('a')
    #    'newline' and 'encoding' are important for CSV
    with open(CSV_OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 3. If the file is new, write the header row
        if not file_exists:
            writer.writerow([
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
            
        # --- THE REST OF THE FUNCTION IS THE SAME ---

        # Loop 1: Iterate through each image
        for img_file in IMAGE_FILES_TO_TEST:
            
            if not os.path.exists(img_file):
                print(f"WARNING: Image not found '{img_file}'. Skipping.")
                continue

            try:
                # This logic will fail on your "test_image2.jpg".
                # A safer version:
                base_name = os.path.basename(img_file) # "test_image2.jpg"
                file_name_no_ext = os.path.splitext(base_name)[0] # "test_image2"
                blur_radius = file_name_no_ext.split('_')[-1] # "image2"
            except:
                blur_radius = "unknown"

            print(f"--- Processing Image: {img_file} (Blur: {blur_radius}) ---")

            # Loop 2: Run the test 3-5 times
            for i in range(1, NUM_RUNS + 1):
                print(f"  Starting Run {i}/{NUM_RUNS}...")
                
                data = await run_mas_test(img_file)
                
                if data:
                    try:
                        # Get the confidence of the FINAL interpretation
                        final_confidence = data['possible_interpretations'][-1]['confidence_score']
                        
                        # Write the data to our CSV file
                        writer.writerow([
                            img_file,
                            blur_radius,
                            i,
                            data['final_conclusion'],
                            final_confidence,
                            data['word_count'],
                            data['processing_time']
                        ])
                    except Exception as e:
                        print(f"  ERROR: Could not extract data from JSON. {e}")

    print("--- Experiment Complete ---")
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())