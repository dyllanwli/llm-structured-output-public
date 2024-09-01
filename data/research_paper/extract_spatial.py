from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List
import json
import os


def setup_api_key(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    api_key = data["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key


setup_api_key(file_path="../../../config.json")


EXTRACT_PROMPT = """
Given data products, if a data product is clearly related to spatial data,
return the UUID. If the data product is not related to spatial data, return na.
texts: {input_texts}
Output format:
{{"uuid": ["********-****-****-****-************", "na", ...]}}
{format_instructions}
"""

model = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")


class ExtractDataResult(BaseModel):
    uuid: List[str] = Field(
        None,
        description="The id of the spatial data, if the data is related to spatial data. Otherwise, na.",
    )


def extract_data(model, input_texts):
    """Given a text, if the data is related to spatial data, extract the uuid of the spatial data.

    Args:
        input_text (str): _description_

    Returns:
        Dict:
    """
    parser = JsonOutputParser(pydantic_object=ExtractDataResult)
    prompt = PromptTemplate(
        template=EXTRACT_PROMPT,
        input_variables=["input_texts"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )
    chain = prompt | model | parser

    response = chain.invoke({"input_texts": input_texts})
    return response


# read the data from ./train.csv, put each line as input_text
import csv

from tqdm import tqdm


def save_result(file_path, result):
    with open("spatial_data.txt", "a", encoding="utf-8") as output_file:
        for item in result:
            output_file.write(f"{item}\n")

def save_data_from_csv(file_path, batch_size=20):
    with open(file_path, mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        batch = []
        # Skip the first 399 rows
        for _ in range(500):
            next(csv_reader)
        
        for row in tqdm(csv_reader, desc="Processing rows"):
            batch.append(row)
            if len(batch) == batch_size:
                try:
                    result = extract_data(model, batch)
                    result = [r for r in result["uuid"] if r != "na"]
                    if len(result) > 0:
                        save_result("spatial_data.txt", result)
                    batch = []
                except Exception as e:
                    print(e)
                    continue
        if batch:  # Process any remaining rows in the last batch
            result = extract_data(model, batch)
            result = [r for r in result["uuid"] if r != "na"]
            if len(result) > 0:
                save_result("spatial_data.txt", result)


save_data_from_csv("./train.csv")
