import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon

from rtree import index
from shapely.geometry import shape

# Load the GeoJSON file
counties = gpd.read_file('../data/geojson/counties.geojson')
print(counties.head())


idx = index.Index()

for i, county in counties.iterrows():
    idx.insert(i, county.geometry.bounds)


print("Querying the point")
query_point = Point(-90.0, 38.0)  # Example coordinates

possible_matches = list(idx.intersection(query_point.bounds))

matching_counties = counties.iloc[possible_matches].loc[counties.iloc[possible_matches].intersects(query_point)]

print(matching_counties)


print("Querying the polygon")
query_polygon = Polygon([(-91, 37), (-89, 37), (-89, 39), (-91, 39), (-91, 37)])

possible_matches = list(idx.intersection(query_polygon.bounds))

matching_counties = counties.iloc[possible_matches].loc[counties.iloc[possible_matches].intersects(query_polygon)]

print(matching_counties)


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# Define the prompt for generating GeoJSON around a point
GEOJSON_PROMPT = """
Given a point with latitude and longitude, generate a GeoJSON object representing a polygon around the point.
latitude: {latitude}
longitude: {longitude}
Output format:
{{"type": "FeatureCollection", "features": [{{"type": "Feature", "geometry": {{"type": "Polygon", "coordinates": [[...]]}}, "properties": {{}}}}]}}
{format_instructions}
"""

# Define the model
model = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

class GeoJSONResult(BaseModel):
    type: str
    features: List[dict] = Field(
        None,
        description="The GeoJSON features generated around the point.",
    )

def generate_geojson_around_point(model, latitude, longitude):
    """Generate a GeoJSON object around a given point using an LLM.

    Args:
        model: The language model to use for generation.
        latitude (float): The latitude of the point.
        longitude (float): The longitude of the point.

    Returns:
        dict: The generated GeoJSON object.
    """
    parser = JsonOutputParser(pydantic_object=GeoJSONResult)
    prompt = PromptTemplate(
        template=GEOJSON_PROMPT,
        input_variables=["latitude", "longitude"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )
    chain = prompt | model | parser

    response = chain.invoke({"latitude": latitude, "longitude": longitude})
    return response

# Example usage
latitude = 38.0
longitude = -90.0
geojson_result = generate_geojson_around_point(model, latitude, longitude)
print(geojson_result)

