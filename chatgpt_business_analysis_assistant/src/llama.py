# # load the large language model file
# from llama_cpp import Llama
# LLM = Llama(model_path="llama-2-7b-chat.ggmlv3.q8_0.bin")


# # # create a text prompt
# # prompt = "Q: What are the names of the days of the week? A:"

# # # generate a response (takes several seconds)
# # output = LLM(prompt)

# # # display the response
# # print(output["choices"][0]["text"])


# # create a text prompt
# prompt = '''system_message=""" You are a market research analyst to help answer scenario based business questions. Your purpose is to analyse the related predictors and provide trend, CAGR, growth rate % etc against the geolocation and time. It should be as quantitative and accurate as possible. """ few_shot_examples=””” <<Template>> Question: User Question, for example: What impact will it have in Azure Cloud storage adoption growth in EMEA in 2024? Provide quantitative value for the growth rate. Thought: Location and time duration will be important to provide specific answer. Action: First, I need to extract the geographic location, EMEA and time duration (2024) from the question. Observation: I get the dependent information required to analyze the metric under consideration Thought: Now I can start my market research work. Action: I should estimate the metric under consideration and provide an honest explaination regarding the same. ... (this Thought/Action/Observation can repeat N times) Final Answer: Your final answer and comment for the question <<Template>>'''


# # generate a response (takes several seconds)
# output = LLM(prompt)

# # display the response
# print(output["choices"][0]["text"])
import os

def llama_response(question, replicate_key):
    os.system(f'SET REPLICATE_API_TOKEN={replicate_key}')
    if question != "":
        prompt = f'''
        system_message="""
You are a market research analyst to help answer scenario based business questions. Your purpose is to analyse the related predictors and provide 
trend, CAGR, growth rate % etc against the geolocation and time. It should be as quantitative and accurate as possible.
"""

 

few_shot_examples="""
<<Template>>
Question: User Question, for example: What impact will it have in Azure Cloud storage adoption growth in EMEA in 2024? Provide quantitative value for the growth rate.
Thought: Location and time duration will be important to provide specific answer. 
Action: First, I need to extract the geographic location, EMEA and time duration (2024) from the question.

 

Observation: I get the dependent information required to analyze the metric under consideration
Thought: Now I can start my market research work. 
Action: I should estimate the metric under consideration and provide an honest explaination regarding the same.

 

... (this Thought/Action/Observation can repeat N times)
Final Answer: Your final answer and comment for the question
<<Template>>
<<Example-1>>
Question: What impact will Ukrain-Russia War have in Azure Cloud storage adoption growth in EMEA in 2024? Provide quantitative value for the growth rate.
Thought: Location and time duration will be important to provide a specific answer.
Action: First, I need to extract the geographic location, EMEA, and time duration (2024) from the question.
Observation: I get the dependent information required to analyze the metric under consideration.
Thought: Now I can start my market research work.
Action: I should estimate the metric under consideration and provide an honest explanation regarding the same.
Estimation: Based on historical data and industry trends, I estimate that Azure Cloud storage adoption in EMEA will grow at a CAGR of 15% between 2020 and 2024. This means that the market size will increase from approximately $X in 2020 to $Y in 2024.
Explanation: The growth in Azure Cloud storage adoption in EMEA can be attributed to factors such as increased cloud computing adoption, growing demand for digital storage, and Microsoft's strong presence in the region. Additionally, the COVID-19 pandemic has accelerated the shift towards cloud computing and remote work, which is also expected to contribute to the growth of Azure Cloud storage adoption in EMEA.
Observation:In the previous response, $X and $Y are placeholder values that represent the starting and ending points of the estimated market size for Azure Cloud storage adoption in EMEA between 2020 and 2024. These values are not actual numbers and were used only for illustrative purposes.
To provide a more accurate estimate, we would need to conduct further research and analysis based on available data and industry trends. Here's an updated version of the response with more realistic values:
Estimation: Based on historical data and industry trends, I estimate that Azure Cloud storage adoption in EMEA will grow at a CAGR of 15% between 2020 and 2024. This means that the market size will increase from approximately $6.7 billion in 2020 to $13.8 billion in 2024.
Explanation: The growth in Azure Cloud storage adoption in EMEA can be attributed to factors such as increased cloud computing adoption, growing demand for digital storage, and Microsoft's strong presence in the region. Additionally, the COVID-19 pandemic has accelerated the shift towards cloud computing and remote work, which is also expected to contribute to the growth of Azure Cloud storage adoption in EMEA.
Final Answer: The estimated CAGR of Azure Cloud storage adoption in EMEA between 2020 and 2024 is 15%. The market size is expected to increase from approximately $6.7 billion in 2020 to $13.8 billion in 2024.
<<Example-1>>"""
Question: {question}
        '''
        print(prompt)
        import replicate
        output = replicate.run(
            "meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
            input={"prompt":prompt}
        )
        # The meta/llama-2-70b-chat model can stream output as it's running.
        # The predict method returns an iterator, and you can iterate over that output.

        result = ""
        for item in output:
            result += f"{item}"
        # print(result)
        return result