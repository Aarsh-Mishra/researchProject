import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import (AssistantAgent)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination,TextMentionTermination


from autogen_core import Image as AGImage

from PIL import Image
from io import BytesIO
import requests

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv('SERPER_API_KEY')

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
        - This message MUST start with the word **DONE** followed immediately by a single, valid JSON object.
        - To build this JSON, you must **review the entire conversation history** and summarize it.

        **JSON SCHEMA:**
        Your final JSON output MUST strictly follow this schema:
        {
          "possible_interpretations": [
            {
              "description": "A detailed description of one key hypothesis discussed.",
              "confidence_score": 10,
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
        - **DO NOT** include any other text, apologies, or explanations like "Here is the JSON:" in your final message. Your *entire* final message must be ONLY the word `DONE` followed by the `{...}` JSON object.
        
        """
    )



async def test_multi_modal():
        
        # response = requests.get('https://picsum.photos/400/300/?blur=2')
        # pil_image = Image.open(BytesIO(response.content))
        pil_image = Image.open("C:\\Users\\hp\\Downloads\\test_image2.jpg")  # Use a local image file
        ag_image = AGImage(pil_image)

        multi_modal_msg = MultiModalMessage(
            content = ['You explain what an image may contain, even if blurred and strictly use the system message for guidance', ag_image],
            source='user'
        )


        
        task = TextMessage(
            content='Image is given as input, it may be blurred. You need to explain what the image may contain, even if blurred and strictly use the system message for guidance.',
            source='user'
        )

        termination_condition = MaxMessageTermination(10) | TextMentionTermination('DONE')

        team = RoundRobinGroupChat(
        participants= [imageExplainerAgent, imageEvaluatorAgent],
        termination_condition=termination_condition,
        max_turns=10
        )

        # result = await team.run(task=multi_modal_msg)
        # for each_agent_message in result.messages:
        #     print(f'{each_agent_message.source} : {each_agent_message.content}')

        async for each_agent_message in team.run_stream(task=multi_modal_msg):
            print(f'{each_agent_message.source}: {each_agent_message.content}')





async def main():
    await test_multi_modal()
    await model_client.close()

    


if __name__ == "__main__":
    asyncio.run(main())
