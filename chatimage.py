import streamlit as st
import base64

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import  Field
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import chain
from PIL import Image


# Define the Schema
class Meal(BaseModel):
    Name: str = Field(..., examples=["Noodel", "Bread", "Steak"], description="The name of the food.")
    Origin: str = Field(..., examples = ["German", "Austrian", "Asian", "Italian"], description="The Origin of the food.")
    Where: str = Field(...,  description="The restaurants or places we can find this food, provide up to 5 places.")
    Information: str = Field(..., description="Further information about this food within 20 words.")
    

# Helper functions
def image_encoding(inputs):
    """Load and convert image to base64 encoding"""
    with open(inputs["image_path"], "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        inputs['image'] = image_base64
    return inputs


# JSON Output Parser
parser = JsonOutputParser(pydantic_object=Meal)
instructions = parser.get_format_instructions()


# Define the Prompt Generation component
@chain
def prompt(inputs):
    """Create the prompt"""
    prompt = [
        SystemMessage(content="""You are an AI assistant whose job is to inspect an image and provide the desired information from the image. If the desired field is not clear or not well detected, return None for this field. Do not try to guess."""),
        HumanMessage(content=[
            {"type": "text", "text": """Examine the name of the food, the origin, places or restaurants we can find this food, and a short further information of this food."""},
            {"type": "text", "text": instructions},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}", "detail": "low", }}])
             ]
    if 'user_question' in inputs:
        prompt = [
        SystemMessage(content="""Answer the question related to the image provided. Answer consisely in less than 50 words."""),
        HumanMessage(content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}", "detail": "low", }}, 
                                {"type": "text", "text": f"Question: {inputs['user_question']}"}])
                ]
    
    return prompt


# Define the multimodal LLM component
@chain
def MLLM_response(inputs):
    """Invoke GPT model to extract information from the image"""
    # Set up OpenAI API key
    with open("key.txt", "r") as file:
        api_key = file.read().strip() 
    model: ChatOpenAI = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1024,
        api_key=api_key
    )
    output = model.invoke(inputs)
    return output.content



# Streamlit Web Application
def run_streamlit_app():
    st.title("Chat with Your Meal")
    st.subheader("Upload an image of your food or meal.")
    uploaded_image = st.file_uploader("Choose an image...", type="jpeg")
    

    if uploaded_image is not None:
        # Save the uploaded image temporarily
        with open("uploaded_image.jpeg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Display the uploaded image
        img = Image.open("uploaded_image.jpeg")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        inputs = {"image_path": "uploaded_image.jpeg"}
        pipeline = image_encoding | prompt | MLLM_response | parser
        output = pipeline.invoke(inputs)
        # Display the results
        st.subheader("Basic Information")
        st.json(output)
    
        st.subheader("Further question about this food or meal.")
        user_question = st.text_input("")
        if user_question:   
            inputs["user_question"] =  user_question

            # Run the pipeline to take care further questions
            pipeline = image_encoding | prompt | MLLM_response
            response = pipeline.invoke(inputs)
            st.write("Answer:", response)


if __name__ == "__main__":
    run_streamlit_app()
