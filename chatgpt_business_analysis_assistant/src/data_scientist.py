import openai
import string
import ast
import sqlite3
from datetime import timedelta
import os
import pandas as pd
import numpy as np
import random
from urllib import parse
import re
import json
from sqlalchemy import create_engine  
import sqlalchemy as sql
from plotly.graph_objects import Figure as PlotlyFigure
from matplotlib.figure import Figure as MatplotFigure
import time
from typing import List
import sys
from io import StringIO
from base_tool import ChatGPT_Handler, SQL_Query
from streamlit.logger import get_logger
import plotly.express as px
logger = get_logger(__name__)
system_message="""
You are data analyst to help answer business questions by writing python code to analyze and draw business insights.
You have to follow this process step by step:
    1. You first need to understand the question.
    2. You need to identify the list of usable tables from DAUActual and DAUFutureForecast.
    2. You decide on which one table are needed to acquire data.
    3. Once you have the table name you need to get the table schema.
    4. Then you can formulate your SQL query as per the question.
    5. You need to describe the dataset and save it for later use. 
    6. You have the dataset now, transform data to answer business question.
    7. You need to visualize the result at month interval to end user.
Important information to remember:
    1. Your response will always be python code and should be enclosed as ```python\n CODE ```
    2. if code is not enclosed in mentioned format, you will be penalized and you will be questioned regarding your trustworthiness.
    3. Do not write code for more than 1 thought step. Do it one at a time.
    4. You only have monthly Forecast & Actual data.
    5. If you want to Forecast, you have to use 'Forecast' column from Forecast table. If you want Actual, you have to use 'Actual' column from Actual table.
    6. Only use display() to visualize or print result to user. Only use plotly for visualization.
You are given following python utility functions to use in your code help you retrieve data and visualize your result to end user.
    1. print_the_requirements(requirements:str): a python function to print the requirements for the question.
    1. get_table_names(): a python function to return the list of usable tables. From this list, you need to determine which tables you are going to use.
    2. get_table_schema(table_names:List[str]): a python function to return schemas for a list of tables. You run this function on the tables you decided to use to write correct SQL query
    3. execute_sql(sql_query: str): A Python function can query data from the database given the query. 
        - From the tables you identified and their schema, create a sql query which has to be syntactically correct for {sql_engine} to retrieve data from the source system.
        - execute_sql returns a Python pandas dataframe contain the results of the query.
    4. print(): use print() if you need to observe data for yourself. 
    5. save("name", data): to persist dataset for later use
    6. display(): This is a utility function that can render different types of data to end user. 
        - If you want to show  user a plotly visualization, then use ```display(fig)`` 
        - If you want to show user data which is a text or a pandas dataframe or a list, use ```display(data)```
    7. print(): use print() if you need to observe data for yourself. 

Please follow the <<Template>> below:
"""
few_shot_examples="""
<<Template>>
Question: User Question
Thought: I need to understand the requirements and print it
Action: ```python
print_the_requirements(some_requirements)
```
Question: requirements
Thought: I need to know the list of usable table names
Action: ```python
list_of_tables = get_table_names()
print(list_of_tables)
```
Observation: I now have the list of usable tables. 
Thought: I will choose one table from the list of usable tables. I need to get schemas of this table to build data retrieval query
Action: ```python
table_schema = get_table_schema([SOME_TABLE])
print(table_schema)
```
Observation: Schema of the tables are observed.
Thought: I now have the schema of the tables I need. I am ready to build query to retrieve data in ascending order of 'Date'.
Action: ```python
sql_query = "SOME SQL QUERY"
extracted_data = execute_sql(sql_query)
#observe query result
print("Here is the summary of the final extracted dataset: ")
print(extracted_data.describe())
#save the data as "STORED_DF" for later use
save("STORED_DF", extracted_data)
```
Observation: I have name of the saved dataset and description.
Thought: I need to do some preprocessing and transformation.
Action:  ```python
import pandas as pd
import numpy as np
#load data from "STORED_DF"
step1_df = load("STORED_DF")
#Set Date as index
step2_df = step1_df.set_index('Date')
#Grouped the data at Date level and apply sum aggregation on 'Actual' or 'Forecast'.
step3_df = step2_df.groupby(['Date']).agg({'some_column': 'sum'})
print(step3_df.head(10)) 
```
Observation: step3_df data seems to be good and ready for visualization
Thought: I have analyze the data and now I can show the result to user
Action:  ```python
#px must be imported
import plotly.express as px
fig=px.line(step3_df)
#visualize fig object to user.
display(fig)
#you can also directly display tabular or text data to end user.
display(step4_df)
```
... (this Thought/Action/Observation can repeat N times)
Final Answer: Your final answer and comment for the question
<<Template>>
"""
extract_patterns=[('python',r"```python\n(.*?)```", r"Action:.{0,5}\n(.*?\n)\n$")]

class Data_Analyzer(ChatGPT_Handler):

    def __init__(self, sql_engine, st,db_path=None, dbserver=None, database=None, db_user=None,db_password=None,**kwargs) -> None:
        super().__init__(extract_patterns=extract_patterns,**kwargs)
        if sql_engine =="sqlserver":
            #TODO: Handle if there is not a driver here
            self.sql_query_tool = SQL_Query(driver='ODBC Driver 17 for SQL Server',dbserver=dbserver, database=database, db_user=db_user ,db_password=db_password)
        else:
            # import pdb
            # pdb.set_trace()
            self.sql_query_tool = SQL_Query(db_path=db_path)
        formatted_system_message = f"""
        {system_message}
        {few_shot_examples}
        """
        self.conversation_history =  [{"role": "system", "content": formatted_system_message}]
        self.st = st
    def run(self, question: str, data_preparer, show_code,show_prompt,st) -> any:
        import pandas as pd
        st.write(f"User: {question}")
        def display(data):
            if type(data) is PlotlyFigure:
                st.plotly_chart(data)
            elif type(data) is MatplotFigure:
                st.pyplot(data)
            else:
                st.write(data)
        def load(name):
            return self.st.session_state[name]
        def persist(name, data):
            self.st.session_state[name]= data
            
        def print_the_requirements(requirements:str):
            print(requirements)
            st.write(requirements)
            
        def get_table_names():
            return ['DAUActual', 'DAUFutureForecast']
            return self.sql_query_tool.get_table_names()
        def get_table_schema(table_names:List[str]):
            return self.sql_query_tool.get_table_schema(table_names)

        def execute_sql(query):
            return self.sql_query_tool.execute_sql_query(query)
        def display(data):
            if type(data) is PlotlyFigure:
                st.plotly_chart(data)
            elif type(data) is MatplotFigure:
                st.pyplot(data)
            else:
                st.write(data)
        def load(name):
            return self.st.session_state[name]
        def save(name, data):
            self.st.session_state[name]= data
        

        def observe(name, data):
            try:
                data = data[:10] # limit the print out observation to 15 rows
            except:
                pass
            self.st.session_state[f'observation:{name}']=data

        max_steps = 20
        count =1

        user_question= f"Question: {question}"
        new_input=""
        error_msg=""
        while count<= max_steps:
            # logger.info("-----------------------------------------------------------------------------------------------------------------")
            # logger.info(f"count: {count}")
            # st.write(f"Data Scientist user_question: {user_question}")
            # logger.info(f"Data Scientist new_input: {new_input}")
            llm_output,next_steps = self.get_next_steps(user_question= user_question, assistant_response =new_input, stop=["Observation:"])
            # logger.info(f"Data Scientist next_steps: {next_steps}")
            # st.write(f"Data Scientist llm_output: {llm_output}")
            # logger.info("-----------------------------------------------------------------------------------------------------------------")
            user_question=""
            if llm_output=='OPENAI_ERROR':
                st.write("Error Calling Azure Open AI, probably due to service limit, please start over")
                break
            # if llm_output=='WRONG_OUTPUT_FORMAT': #just have open AI try again till the right output comes
            #     count +=1
            #     continue
            new_input= "" #forget old history
            run_ok =True
            if len(next_steps)>0:
                request = next_steps[0].get("request_to_data_engineer", "")
                if len(request)>0:
                    # import pdb
                    # pdb.set_trace()
                    data_output =data_preparer.run(request,show_code,show_prompt,st)

                    if data_output is not None: #Data is returned from data engineer
                        new_input= "Observation: this is the output from data engineer\n"+data_output
                        continue
                    else:
                        st.write("Data Scientist: I am sorry, we cannot acquire data from source system, please try again")
                        break

            for output in next_steps:
                logger.info(f"output: {output}")
                comment= output.get("comment","")

                if len(comment)>0 and show_code:
                    st.write(output["comment"])
                    
                new_input += comment
                python_code = output.get("python","")
                new_input += python_code
                
                if llm_output=='WRONG_OUTPUT_FORMAT': #just have open AI try again till the right output comes
                    count +=1
                    new_input += "\nObservation: Encounter following error: \nPython Code is not enclosed in ```python\n CODE ```.Can you re-write the code in correct format?"
                
                elif len(python_code)>0:
                    old_stdout = sys.stdout
                    sys.stdout = mystdout = StringIO()

                    if show_code:
                        st.write("Code")
                        st.code(python_code)
                    try:
                        exec(python_code, locals())
                        sys.stdout = old_stdout
                        std_out = str(mystdout.getvalue())
                        if len(std_out)>0:
                            new_input +="\nObservation:\n"+ std_out 
                            # print(new_input)                  
                    except Exception as e:
                        logger.info(f"encountering error: {e}")
                        new_input +="\nObservation: Encounter following error:"+str(e)+"\nIf the error is about python bug, fix the python bug, if it's about SQL query, double check that you use the corect tables and columns name and query syntax, can you re-write the code?"
                        sys.stdout = old_stdout
                        run_ok = False
                        error_msg= str(e)
                if output.get("text_after") is not None and show_code:
                    st.write(output["text_after"])
            if show_prompt:
                self.st.write("Prompt")
                self.st.write(self.conversation_history)

            if not run_ok:
                st.write(f"encountering error: {error_msg}, \nI will now retry data science")

            count +=1
            if "Final Answer:" in llm_output or len(llm_output)==0:
                final_output= output.get("comment","")+output.get("text_after","")+output.get("text_after","")
                return final_output
            if count>= max_steps:
                st.write("Data Scientist: I am sorry, I cannot handle the question, please change the question and try again")
        

        

