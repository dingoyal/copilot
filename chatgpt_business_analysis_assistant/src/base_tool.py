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
import pyodbc
# from streamlit.logger import get_logger
# logger = get_logger(__name__)

import logging as logger

# Configure the logger
logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def validate_output( llm_output,extracted_output):
    logger.info(f"extracted_output={extracted_output}")
    valid = False
    if "Final Answer:" in llm_output:
        return True
    for output in extracted_output:
        if len(output.get("python",""))!=0 or len(output.get("request_to_data_engineer",""))!=0:
            return True
    if (llm_output == "OPENAI_ERROR"):
        valid = True
    

    return valid
class ChatGPT_Handler: #designed for chatcompletion API
    def __init__(self, extract_patterns=None,gpt_deployment=None,max_response_tokens=None,token_limit=None,temperature=None ) -> None:
        self.max_response_tokens = max_response_tokens
        self.token_limit= token_limit
        self.gpt_deployment=gpt_deployment
        self.temperature=temperature
        self.extract_patterns = extract_patterns
        # self.conversation_history = []
    def _call_llm(self,prompt, stop):
        response = openai.ChatCompletion.create(
        engine=self.gpt_deployment, 
        messages = prompt,
        temperature=self.temperature,
        max_tokens=self.max_response_tokens,
        stop= stop
        )
        # import pdb
        # pdb.set_trace()
        try:
            llm_output = response['choices'][0]['message']['content']
        except:
            llm_output=""

        return llm_output
    def extract_code_and_comment(self,entire_input, python_codes):
        # import pdb;pdb.set_trace()
        # print("entire_input: \n", entire_input)
        remaing_input = entire_input
        comments=[]
        logger.info(f"python_codes={python_codes}")
        for python_code in python_codes:
            try:
                temp_python_code = "```python\n"+python_code+"```"
                logger.info(f"temp_python_code={temp_python_code}")
                text_before = remaing_input.split(temp_python_code)[0]
                logger.info(f"text_before={text_before}")
                remaing_input = remaing_input.split(temp_python_code)[1]
                comments.append(text_before)
                logger.info(f"remaing_input={remaing_input}")
            except Exception as e:
                logger.error(str(e))
                temp_python_code = "\n"+python_code+"\n"
                logger.info(f"temp_python_code={temp_python_code}")
                text_before = remaing_input.split(temp_python_code)[0]
                logger.info(f"text_before={text_before}")
                comments.append(text_before)
                remaing_input = remaing_input.split(temp_python_code)[1]
                logger.info(f"remaing_input={remaing_input}")
        return comments, remaing_input
    
    def extract_output(self, text_input):
            # print("text_input\n",text_input)
            # import pdb
            # pdb.set_trace()
            logger.info(f"text_input={text_input}")
            outputs=[]
            logger.info(f"extract_patterns={self.extract_patterns}")
            for pattern in self.extract_patterns: 
                logger.info(f"pattern={pattern}")
                if "python" in pattern[1]:
                    python_codes = re.findall(pattern[1], text_input, re.DOTALL)
                    logger.info(f"python_codes={python_codes}")
                    if len(python_codes)==0:
                        python_codes = re.findall(pattern[2], text_input, re.DOTALL)
                    logger.info(f"python_codes={python_codes}")
                    comments, text_after= self.extract_code_and_comment(text_input, python_codes)
                    # print("text_after ", text_after)
                    for comment, code in zip(comments, python_codes):
                        outputs.append({"python":code, "comment":comment})
                    outputs.append({"text_after":text_after})
            return outputs
    def get_next_steps(self, user_question, assistant_response, stop):
        # st.write(f"assistant_response: {assistant_response}")
        if len(self.conversation_history)>2:
            self.conversation_history.pop() #removing old history

        if len(user_question)>0:
            self.conversation_history.append({"role": "user", "content": user_question})
        if len(assistant_response)>0:
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
        n=0
        # import pdb
        # pdb.set_trace()
        logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info("conversation_history %r",self.conversation_history)
        logger.info(f"stop={stop}")
        # import pdb
        # pdb.set_trace()
        try:
            llm_output = self._call_llm(self.conversation_history, stop)
            # st.write(f"llm_output={llm_output}")
            logger.info("llm_output %r",llm_output)
            logger.info("************ **************************************************************************************")

        except Exception as e:
            logger.error(str(e))
            if "maximum context length" in str(e):
                print(f"Context length exceeded")
                return "OPENAI_ERROR",""  
            time.sleep(8) #sleep for 8 seconds
            while n<2:
                try:
                    llm_output = self._call_llm(self.conversation_history, stop)
                except Exception as e:
                    logger.error(str(e))
                    n +=1

                    print(f"error calling open AI, I am retrying 5 attempts , attempt {n}")
                    time.sleep(8) #sleep for 8 seconds
                    print(str(e))

            llm_output = "OPENAI_ERROR"     
             
    
        outputs = self.extract_output(llm_output)
        if len(llm_output)==0:
            return "",[]
        if not validate_output(llm_output, outputs): #wrong output format
            logger.info("======================WRONG_OUTPUT_FORMAT==============================")
            llm_output = "WRONG_OUTPUT_FORMAT"
        # import pdb
        # pdb.set_trace()
        return llm_output,outputs

class SQL_Query(ChatGPT_Handler):
    def __init__(self, system_message="",data_sources="",db_path=None,driver=None,dbserver=None, database=None, db_user=None ,db_password=None, **kwargs):
        super().__init__(**kwargs)
        if len(system_message)>0:
            self.system_message = f"""
            {data_sources}
            {system_message}
            """
        self.sql_engine =os.environ.get("SQL_ENGINE","sqlite")
        self.database=database
        self.dbserver=dbserver
        self.db_user = db_user
        self.db_password = db_password
        self.db_path= db_path #This is the built-in demo using SQLite
        
        self.driver= driver
        
    def execute_sql_query(self, query, limit=10000):  
        if self.sql_engine == 'sqlserver': 
            # import pdb
            # pdb.set_trace()
            # logger.info("inside sqlserver engine")
            # logger.info(query)
            try:
                # connecting_string = f'Driver={{ODBC Driver 17 for SQL Server}};Server=tcp:{self.dbserver};database={self.database};UID={self.db_user};PWD={self.db_password};MultipleActiveResultSets=true;'
                # connecting_string = f"Driver={{ODBC Driver 17 for SQL Server}};Server=tcp:{self.dbserver},1433;Database={self.database};Uid={self.db_user};Pwd={self.db_password}"
                # params = parse.quote_plus(connecting_string)

                # engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, connect_args={'connect_timeout': 200})
                engine = pyodbc.connect(f'Driver={{ODBC Driver 17 for SQL Server}};'
                        f'Server=tcp:{self.dbserver};'
                        f'database={self.database};'
                        f'UID={self.db_user};'
                        f'PWD={self.db_password};'
                        'MultipleActiveResultSets=true;')
                logger.info(engine)
            except Exception as e:
                logger.info("error_while_making_sql_conn", str(e))
                # params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                #                  "SERVER=dagger;"
                #                  "DATABASE=test;"
                #                  "UID=user;"
                #                  "PWD=password")

                engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params), connect_args={'connect_timeout': 200})
                # logger.info(query)
                # logger.info("error", str(e))
                # logger.info(engine)
                # raise e
        else:
            # logger.info("inside sqlite engine")
            engine = create_engine(f'sqlite:///{self.db_path}')  


        result = pd.read_sql_query(query, engine)
        result = result.infer_objects()
        # logger.info(result)
        for col in result.columns:  
            if 'date' in col.lower():  
                result[col] = pd.to_datetime(result[col], errors="ignore")  
  
        if limit is not None:  
            result = result.head(limit)  # limit to save memory  
  
        # session.close()  
        return result  
    def get_table_schema(self, table_names:List[str]):

        # Create a comma-separated string of table names for the IN operator  
        table_names_str = ','.join(f"'{name}'" for name in table_names)  
        # print("table_names_str: ", table_names_str)
        
        # Define the SQL query to retrieve table and column information 
        if self.sql_engine== 'sqlserver': 
            sql_query = f"""  
            SELECT C.TABLE_NAME, C.COLUMN_NAME, C.DATA_TYPE, T.TABLE_TYPE, T.TABLE_SCHEMA  
            FROM INFORMATION_SCHEMA.COLUMNS C  
            JOIN INFORMATION_SCHEMA.TABLES T ON C.TABLE_NAME = T.TABLE_NAME AND C.TABLE_SCHEMA = T.TABLE_SCHEMA  
            WHERE T.TABLE_TYPE = 'BASE TABLE'  AND C.TABLE_NAME IN ({table_names_str})  
            """  
        elif self.sql_engine=='sqlite':
            sql_query = f"""    
            SELECT m.name AS TABLE_NAME, p.name AS COLUMN_NAME, p.type AS DATA_TYPE  
            FROM sqlite_master AS m  
            JOIN pragma_table_info(m.name) AS p  
            WHERE m.type = 'table'  AND m.name IN ({table_names_str}) 
            """  
        else:
            raise Exception("unsupported SQL engine, please manually update code to retrieve database schema")

        # Execute the SQL query and store the results in a DataFrame  
        df = self.execute_sql_query(sql_query, limit=None)  
        output=[]
        # Initialize variables to store table and column information  
        current_table = ''  
        columns = []  
        
        # Loop through the query results and output the table and column information  
        for index, row in df.iterrows():
            if self.sql_engine== 'sqlserver': 
                table_name = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"  
            else:
                table_name = f"{row['TABLE_NAME']}" 

            column_name = row['COLUMN_NAME']  
            data_type = row['DATA_TYPE']   
            if " " in table_name:
                table_name= f"[{table_name}]" 
            column_name = row['COLUMN_NAME']  
            if " " in column_name:
                column_name= f"[{column_name}]" 

            # If the table name has changed, output the previous table's information  
            if current_table != table_name and current_table != '':  
                output.append(f"table: {current_table}, columns: {', '.join(columns)}")  
                columns = []  
            
            # Add the current column information to the list of columns for the current table  
            columns.append(f"{column_name} {data_type}")  
            
            # Update the current table name  
            current_table = table_name  
        
        # Output the last table's information  
        output.append(f"table: {current_table}, columns: {', '.join(columns)}")
        output = "\n ".join(output)
        return output
    def get_table_names(self):
        
        # return ['DAUReport', 'UGBReport']
        
        # Define the SQL query to retrieve table and column information 
        if self.sql_engine== 'sqlserver': 
            sql_query = """  
            SELECT DISTINCT C.TABLE_NAME  
            FROM INFORMATION_SCHEMA.COLUMNS C  
            JOIN INFORMATION_SCHEMA.TABLES T ON C.TABLE_NAME = T.TABLE_NAME AND C.TABLE_SCHEMA = T.TABLE_SCHEMA  
            WHERE T.TABLE_TYPE = 'BASE TABLE' 
            """  
        elif self.sql_engine=='sqlite':
            sql_query = """    
            SELECT DISTINCT m.name AS TABLE_NAME 
            FROM sqlite_master AS m  
            JOIN pragma_table_info(m.name) AS p  
            WHERE m.type = 'table' 
            """  
        else:
            raise Exception("unsupported SQL engine, please manually update code to retrieve database schema")

        df = self.execute_sql_query(sql_query, limit=None)  
        
        output=[]
        
        # Loop through the query results and output the table and column information  
        for index, row in df.iterrows():
            if self.sql_engine== 'sqlserver': 
                # table_name = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"  
                table_name = f"{row['TABLE_NAME']}" 
            else:
                table_name = f"{row['TABLE_NAME']}" 

            if " " in table_name:
                table_name= f"[{table_name}]" 
            output.append(table_name)  
            # logger.info(output)
        return output








    