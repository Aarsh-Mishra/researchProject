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
        You are an Explainer agent.  
        You receive a blurred image from the user.  
        Your job is to analyze the image carefully.  

        - Describe what you see, even if uncertain.  
        - Provide multiple possible interpretations if the image is unclear.  
        - Be explicit about your level of confidence (e.g., "likely", "possibly").  
        - Include relevant details such as shapes, colors, context, or patterns.  
        - Do not finalize the conclusion — instead, provide a draft explanation for the Evaluator to critique.  
        """
        )

imageEvaluatorAgent = AssistantAgent(
        name = 'evaluator',
        model_client= model_client,
        system_message=
        """
        You are an Evaluator agent.  
        You receive the Explainer’s description of a blurred image.  
        Your job is to critique and refine the explanation.  

        - Check for consistency, plausibility, and clarity.  
        - If the description is too vague, ask for more details or clarification.  
        - Suggest corrections or alternative interpretations.  
        - If you also have access to the image, cross-check it against the explanation.  
        - Work collaboratively with the Explainer until the final explanation is accurate and well-structured. 
        - You have to come with some realistic situations based on the explanation provided by the Explainer. 
        - Mark the final answer explicitly with "Final Answer:" and conclude about the image what it explains.  
        - If you are satisfied then say "DONE".
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
